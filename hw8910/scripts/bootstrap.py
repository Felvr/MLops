"""Build portable manifests from preserved HW67 evidence and local HF/MLflow files.
No training and no hand-written PASS. Baseline retention is explicitly NOT promotion.
"""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from release import canonical, digest, file_digest, verify_files, write_json, tokenizer_hash
from provenance import make_report, verify_sources

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'hw67'


def main():
    evidence=ROOT/'evidence'
    evidence.mkdir(exist_ok=True)
    for name in ['baseline-eval.json','candidate-eval.json','baseline-outputs.json','candidate-outputs.json','candidate-model.json']:
        shutil.copy2(OLD/'metrics'/name,evidence/name)
    for source,name in [('evaluation/params.json','params.json'),('evaluation/baseline.json','baseline-spec.json'),
                        ('data/eval/golden.jsonl','golden.jsonl'),('data/eval/behavioral.jsonl','behavioral.jsonl'),
                        ('data/eval/training-index.jsonl','training-index.jsonl')]:
        shutil.copy2(OLD/source,evidence/name)
    shutil.copytree(OLD/'evaluation',evidence/'evaluation-code',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    base_spec=json.loads((evidence/'baseline-spec.json').read_text())
    from huggingface_hub import snapshot_download
    snapshot=Path(snapshot_download(base_spec['name'],revision=base_spec['revision'],local_files_only=True))
    (ROOT/'artifacts').mkdir(exist_ok=True)
    base=ROOT/'artifacts/base'
    if not base.exists(): base.symlink_to(snapshot,target_is_directory=True)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    params=json.loads((evidence/'params.json').read_text())
    runtime={'base_name':base_spec['name'],'base_revision':base_spec['revision'],
             'tokenizer_sha256':tokenizer_hash(base), 'tokenizer_revision':base_spec['revision'],
             'template_sha256':digest(tokenizer.chat_template.encode()),
             'generation':{key:params[key] for key in ('system','max_new_tokens','enable_thinking','seed')},
             'dtype':'float32','device':'cpu','max_input_tokens':4096}
    runtime['generation']['do_sample']=False
    sources={str(p.relative_to(ROOT)):file_digest(p) for p in sorted(evidence.rglob('*')) if p.is_file()}
    old=json.loads((evidence/'baseline-eval.json').read_text())
    baseline={'schema_version':1,'model_id':'Qwen3-0.6B-base-'+base_spec['revision'][:12],
              'artifact_uri':'artifacts/base','tokenizer_uri':'artifacts/base','artifact_root':'artifacts','runtime':runtime,
              'runtime_sha256':digest(canonical(runtime)),'protocol_sha256':old['protocol_sha256'],
              'golden_sha256':old['golden_sha256'],'source_hashes':sources,'model_spec':base_spec,
              'acceptance':'retained HW67 baseline; NOT a newly gate-approved model',
              'known_limitation':'behavior-refusal failed in HW67 (2/3 behavioral cases)'}
    files={'base/'+p.name:file_digest(p) for p in sorted(base.iterdir()) if p.is_file() and p.suffix in ('.json','.txt','.safetensors')}
    baseline.update(file_hashes=files,model_sha256=digest(canonical(files)))
    write_json(ROOT/'baseline-manifest.json',baseline)
    candidate=json.loads(canonical(baseline))
    spec=json.loads((evidence/'candidate-model.json').read_text())
    candidate.update(model_id='Qwen3-LoRA-'+spec['run_id'],model_spec=spec,
                     adapter_uri='artifacts/adapter',acceptance='requires quality gate')
    adapter=ROOT/'artifacts/adapter'
    if not (adapter/'adapter_config.json').exists():
        raise ValueError('First run scripts/export_hw67.py using HW67 environment and live MLflow')
    for p in sorted(adapter.rglob('*')):
        if p.is_file(): candidate['file_hashes']['adapter/'+str(p.relative_to(adapter))]=file_digest(p)
    candidate['model_sha256']=digest(canonical(candidate['file_hashes']))
    candidate['tokenizer_uri']='artifacts/adapter'
    candidate['runtime']['tokenizer_sha256']=tokenizer_hash(adapter)
    candidate['runtime_sha256']=digest(canonical(candidate['runtime']))
    write_json(ROOT/'candidate-manifest.json',candidate)
    for manifest in (baseline,candidate): verify_files(manifest,ROOT);verify_sources(manifest,ROOT)
    write_json(ROOT/'release-report.json',make_report(candidate,ROOT))
    # Initial pointer retains an existing baseline, even when both behavioral gates fail.
    # Never label this decision passed=True or publish candidate around its gate.
    if not (ROOT/'accepted.json').exists(): write_json(ROOT/'accepted.json',baseline)
    write_json(ROOT/'baseline-retention.json',{'action':'retain previously pinned baseline','passed':False,
               'reason':'Candidate failed mandatory behavioral gate; no promotion',
               'baseline_evaluation_run_id':old['evaluation_run_id'],
               'manifest_sha256':digest(canonical(baseline))})
    print('Prepared manifests; candidate gate:',make_report(candidate,ROOT)['decision'])


if __name__=='__main__': main()
