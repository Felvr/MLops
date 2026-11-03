"""Prefect: real schema/token preparation -> existing artifact -> gate -> publish.
Only URI/hash descriptions cross task boundaries; never weights or tensors.
"""
import argparse
import json
import os
from pathlib import Path
import uuid

HOME=Path(__file__).resolve().parent
os.environ.setdefault('PREFECT_HOME',str(HOME/'.prefect'))
os.environ.setdefault('PREFECT_SERVER_ANALYTICS_ENABLED','false')
os.environ.setdefault('PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS','90')
os.environ.setdefault('PREFECT_LOCAL_STORAGE_PATH',str(HOME/'.state/prefect-results'))
from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE
from release import cache_key, file_digest, canonical, digest, verify_files, write_json, publish_event
from provenance import make_report, verify_sources


def read(path):
    return json.loads(Path(path).read_text())


def preparation_config(root, synthetic):
    if synthetic:
        config=read(root/'config.json')
    else:
        manifest=read(root/'candidate-manifest.json')
        runtime=manifest['runtime']
        config={key:runtime[key] for key in ('tokenizer_revision','tokenizer_sha256','template_sha256')}
        config.update(max_length=runtime['max_input_tokens'],
                      system_sha256=digest(runtime['generation']['system'].encode()),
                      environment_sha256=file_digest(HOME/'uv.lock'),
                      code_revision=digest(canonical({name:file_digest(HOME/name)
                                                      for name in ('pipeline.py','release.py')})))
    return config


def prepare_key(context, parameters):
    root=Path(parameters['root'])
    source=root/('input.txt' if parameters['synthetic'] else 'evidence/golden.jsonl')
    # Workspace namespace prevents reuse of descriptors pointing into another workspace.
    return digest(canonical({'input':cache_key(source,parameters['config']),'workspace':str(root.resolve())}))


@task(persist_result=True,cache_key_fn=prepare_key,result_serializer='json',log_prints=True)
def prepare(root: str, synthetic: bool, config: dict):
    root=Path(root)
    source=root/('input.txt' if synthetic else 'evidence/golden.jsonl')
    key=cache_key(source,config)
    entry=root/'cache'/(key+'.json') if synthetic else root/'.state/prepared'/(key+'.json')
    if entry.exists():
        result=read(entry)
        print('CACHE_HIT',key)
    else:
        payload=source.read_text()
        if synthetic:
            lengths=[len(line) for line in payload.splitlines()]
        else:
            from transformers import AutoTokenizer
            tokenizer_uri=read(root/'candidate-manifest.json')['tokenizer_uri']
            tokenizer=AutoTokenizer.from_pretrained(root/tokenizer_uri,local_files_only=True)
            rows=[json.loads(line) for line in payload.splitlines()]
            ids=set(); lengths=[]
            system=read(root/'evidence/params.json')['system']
            for row in rows:
                if not isinstance(row['id'],str) or row['id'] in ids or not isinstance(row['input'],str) or not row['input'].strip():
                    raise ValueError('Invalid or duplicate evaluation row')
                ids.add(row['id'])
                tokens=tokenizer.apply_chat_template([{'role':'system','content':system},
                    {'role':'user','content':row['input']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
                if len(tokens)>config['max_length']: raise ValueError('Input too long')
                lengths.append(len(tokens))
        result={'input_sha256':file_digest(source),'rows':len(lengths),'lengths':lengths,'config':config}
        write_json(entry,result)
        print('PREPARED',key)
    return {'uri':str(entry),'sha256':file_digest(entry),'input_sha256':result['input_sha256']}


@task(cache_policy=NO_CACHE)
def obtain_artifact(root: str, synthetic: bool, manifest_name: str):
    root=Path(root)
    manifest=read(root/manifest_name)
    verify_files(manifest,root)
    if not synthetic: verify_sources(manifest,root)
    get_run_logger().info('Read existing artifact; NO training: %s',manifest['model_id'])
    return {'uri':str(root/manifest_name),'sha256':file_digest(root/manifest_name),
            'model_id':manifest['model_id'],'model_sha256':manifest['model_sha256']}


@task(cache_policy=NO_CACHE)
def evaluate_gate(root: str, synthetic: bool, artifact: dict, report_name: str):
    root=Path(root)
    if file_digest(artifact['uri'])!=artifact['sha256']: raise ValueError('Manifest changed during flow')
    manifest=read(artifact['uri'])
    report=read(root/report_name) if synthetic else make_report(manifest,root)
    destination=root/('flow-report.json' if synthetic else 'release-report.json')
    write_json(destination,report)
    get_run_logger().info('Gate passed=%s decision=%s',report.get('passed'),report.get('decision'))
    return {'uri':str(destination),'sha256':file_digest(destination)}


@task(cache_policy=NO_CACHE)
def publication(root: str, artifact: dict, report: dict, fail_after_pointer: bool):
    for descriptor in (artifact,report):
        if file_digest(descriptor['uri'])!=descriptor['sha256']: raise ValueError('Stage result changed')
    manifest=read(artifact['uri']); evaluation=read(report['uri'])
    identity=publish_event(Path(root),manifest,evaluation,fail_after_pointer)
    return {'release_id':identity,'model_id':manifest['model_id']}


@flow(name='hw8910-release',log_prints=True)
def release_flow(root: str, synthetic=False, fail_after_prepare=False, fail_after_pointer=False,
                 manifest_name=None, report_name='report-B.json'):
    root=Path(root).resolve()
    if manifest_name is None: manifest_name='manifest-B.json' if synthetic else 'candidate-manifest.json'
    attempts=root/'attempts.jsonl' if synthetic else root/'.state/attempts.jsonl'
    attempts.parent.mkdir(exist_ok=True,parents=True)
    with attempts.open('a') as f:
        f.write(json.dumps({'attempt_id':uuid.uuid4().hex,'manifest':manifest_name})+'\n')
    descriptor=prepare(str(root),synthetic,preparation_config(root,synthetic),return_state=True)
    print('PREPARATION_STATE',descriptor.name,descriptor.state_details.task_run_id)
    prepared=descriptor.result()
    # Check a cached descriptor still points to durable, unmodified bytes.
    if file_digest(prepared['uri'])!=prepared['sha256']: raise ValueError('Cached preparation changed')
    if fail_after_prepare: raise RuntimeError('Injected failure after preparation')
    artifact=obtain_artifact(str(root),synthetic,manifest_name)
    report=evaluate_gate(str(root),synthetic,artifact,report_name)
    return publication(str(root),artifact,report,fail_after_pointer)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,default=HOME)
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--fail-after-prepare',action='store_true')
    p.add_argument('--fail-after-pointer',action='store_true')
    p.add_argument('--manifest')
    p.add_argument('--report',default='report-B.json')
    a=p.parse_args()
    release_flow(str(a.workspace),a.synthetic,a.fail_after_prepare,a.fail_after_pointer,a.manifest,a.report)


if __name__=='__main__': main()
