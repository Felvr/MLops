"""Public baseline-only restoration; never needs private candidate weights."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from release import verify_files
import json

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'accepted.json').read_text())
if manifest['model_spec']['kind']!='base': raise SystemExit('This helper restores only baseline')
from huggingface_hub import snapshot_download
snapshot=Path(snapshot_download(manifest['runtime']['base_name'],revision=manifest['runtime']['base_revision']))
base=ROOT/manifest['artifact_uri']
base.parent.mkdir(parents=True,exist_ok=True)
if not base.exists(): base.symlink_to(snapshot,target_is_directory=True)
verify_files(manifest,ROOT)
print('Exact baseline files verified; run make serve')
