import json
import re
from pathlib import Path

from mlops.rag.config import corpus_files, load
from mlops.rag.llm_client import chat

cfg = load()
Path("eval").mkdir(exist_ok=True)
rows = []
for f in corpus_files(cfg):
    r = chat(cfg["judge"], (
        "Below is a moderation guideline document. Write 2 different questions a moderator might ask whose "
        "answers are in it, with short reference answers. The questions must NOT contain the document's title "
        "or any distinctive phrase from it: describe the situation by its properties instead. "
        'Respond with only a JSON array: [{"question": "...", "reference_answer": "..."}]\n\n' + f.read_text()))
    m = re.search(r"\[.*\]", r["text"], re.S)
    if not m:
        print("skip", f.name)
        continue
    for item in json.loads(m.group(0)):
        rows.append({"id": f"q{len(rows) + 1:03d}", "question": item["question"],
                     "relevant_docs": [f.stem], "reference_answer": item["reference_answer"]})
Path(cfg["eval"]["questions"]).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print(len(rows), "questions written; now read them and delete bad ones")