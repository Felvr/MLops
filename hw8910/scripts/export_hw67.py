"""Run with HW67 environment from HW67 cwd; read existing MLflow artifact only."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path.cwd()))
from evaluation.cli import current_reports, candidate_model, manifest_for
current_reports()  # existing provenance and leakage checks; NO training/evaluation
spec=candidate_model()
manifest,adapter=manifest_for(spec)
destination=Path('../hw8910/artifacts/adapter')
destination.parent.mkdir(exist_ok=True)
shutil.copytree(adapter,destination,dirs_exist_ok=True)
print(json.dumps({'training_run_id':spec['run_id'],'model_spec':spec,'artifact_manifest':manifest},ensure_ascii=False,indent=2))
