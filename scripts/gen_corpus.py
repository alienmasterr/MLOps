from pathlib import Path

from mlops.rag.config import load
from mlops.rag.llm_client import chat

TOPICS = {
    "insults": "personal insults aimed at another user", "profanity_not_targeted": "swearing not aimed at anyone",
    "identity_attacks": "attacks on a group by race, religion, gender, orientation or disability",
    "threats": "threats of violence", "harassment_repeated": "repeated unwanted contact with one user",
    "sexual_content": "sexually explicit comments", "quoting_toxic_content": "quoting toxic text to criticise it",
    "sarcasm_and_irony": "sarcasm that looks toxic out of context", "reclaimed_slurs": "reclaimed slurs used by community members",
    "political_criticism": "harsh criticism of policies and parties", "criticism_of_public_figures": "criticising named public figures",
    "satire": "satire and jokes", "spam_and_links": "spam and promotional links", "doxxing_personal_info": "publishing private personal information",
    "self_harm_mentions": "mentions of self-harm", "hate_symbols_and_dogwhistles": "coded hateful language",
}

cfg = load()
out = Path(cfg["corpus"]["dir"])
out.mkdir(exist_ok=True)
for slug, desc in TOPICS.items():
    f = out / f"{slug}.md"
    if f.exists():
        continue
    r = chat(cfg["judge"], (
        "Write a short internal moderation-guideline document (110-200 words) for a news-comment platform "
        f"with three outcomes: publish, send to human review, hide. Topic: {desc}. "
        "Start with a first line '# <Title>'. Include one concrete rule with a number or condition and one "
        "example described in words (never quote slurs). Plain text, no other headings."))
    f.write_text(r["text"].strip() + "\n")
    print("wrote", f)