"""MedQuAD/GHR XML → тематический chat JSONL с происхождением каждой пары."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from src.config import load_params
from src.schema import Example, dump


def text(element):
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def main():
    params = load_params()
    cfg, paths = params["collect"], params["paths"]
    version = cfg["version"]
    rows, provenance = [], []
    counts = Counter()
    qtypes = Counter()
    ids = set()
    with zipfile.ZipFile(params["source"]["path"]) as archive:
        xml_files = [name for name in archive.namelist() if name.endswith(".xml")]
        selected = [name for name in xml_files if f"/{cfg['subset']}/" in name]
        selected.sort(key=lambda name: hashlib.sha256(name.split("/", 1)[1].encode()).hexdigest())
        counts["source_xml_files"] = len(xml_files)
        counts["domain_xml_files"] = len(selected)
        counts["excluded_other_domains"] = len(xml_files) - len(selected)
        selected = selected[:cfg["documents"][version]]
        counts["selected_documents"] = len(selected)
        for name in selected:
            root = ET.fromstring(archive.read(name))
            focus = text(root.find("Focus"))
            if not focus:
                raise ValueError(f"{name}: пустой Focus")
            for pair in root.findall("QAPairs/QAPair"):
                counts["pairs_scanned"] += 1
                question = pair.find("Question")
                if question is None:
                    raise ValueError(f"{name}: Question отсутствует")
                qtype = question.get("qtype")
                if qtype not in cfg["question_types"]:
                    counts["excluded_question_type"] += 1
                    continue
                user, answer = text(question), text(pair.find("Answer"))
                if not user or not answer:
                    counts["excluded_empty"] += 1
                    continue
                qid = question.get("qid")
                if not qid:
                    raise ValueError(f"{name}: нет qid")
                row_id = f"ghr_{qid}"
                if row_id in ids:
                    raise ValueError(f"повторный id: {row_id}")
                ids.add(row_id)
                prompt_index = int(hashlib.sha256(row_id.encode()).hexdigest(), 16) % len(cfg["system_prompts"])
                row = Example.model_validate({
                    "id": row_id, "topic": focus,
                    "messages": [
                        {"role": "system", "content": cfg["system_prompts"][prompt_index]},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": answer},
                    ],
                })
                rows.append(row)
                qtypes[qtype] += 1
                provenance.append({"id": row_id, "xml": name.split("/", 1)[1],
                                   "source_url": root.get("url"), "qid": qid,
                                   "question_type": qtype, "source_revision": params["source"]["revision"]})
    for key, values in (("raw", [dump(row) for row in rows]),
                        ("provenance", [json.dumps(row, ensure_ascii=False) for row in provenance])):
        path = Path(paths[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(values) + "\n", encoding="utf-8")
    metrics = {"version": version, **counts, "rows_out": len(rows),
               "questions_by_type": dict(qtypes), "system_prompts": len(cfg["system_prompts"]),
               "enriched_rows": len(rows), "source_revision": params["source"]["revision"],
               "excluded_empty": counts["excluded_empty"]}
    mpath = Path(paths["metrics_collect"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"collect {version}: {counts['selected_documents']} документов, {len(rows)} QA, "
          f"отсечено по типу {counts['excluded_question_type']}")


if __name__ == "__main__":
    main()
