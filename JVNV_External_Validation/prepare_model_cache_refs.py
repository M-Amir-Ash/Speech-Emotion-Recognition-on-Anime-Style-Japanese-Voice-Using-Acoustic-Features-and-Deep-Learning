"""Create Hugging Face refs/main files pointing only to audited pinned snapshots."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent; base=HERE/"model_cache/huggingface/hub"
for model,revision in json.loads((HERE/"model_revisions.json").read_text(encoding="utf-8")).items():
    repo=base/("models--"+model.replace("/","--")); snapshot=repo/"snapshots"/revision
    if snapshot.is_dir():
        refs=repo/"refs"; refs.mkdir(parents=True,exist_ok=True); (refs/"main").write_text(revision,encoding="utf-8")
        print(model,revision)
