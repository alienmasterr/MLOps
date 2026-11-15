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
    "all_caps_and_shouting": "all-caps comments", "off_topic_comments": "off-topic comments",
    "trolling_and_baiting": "deliberate provocation", "misinformation_claims": "false factual claims",
    "appeals_process": "how an author appeals a hidden comment", "human_review_queue": "which comments go to human review and in what order",
    "escalation_to_legal": "when moderators escalate to the legal team", "repeat_offenders": "authors with many hidden comments",
    "auto_hide_threshold": "score above which a comment is hidden automatically", "publish_threshold": "score below which a comment is published immediately",
    "label_delay_and_rater_disagreement": "crowd rater labels arrive late and raters disagree", "non_english_comments": "comments not in English",
    "replies_and_context": "judging a reply using the parent comment", "mentions_of_identity_neutral": "neutral mentions of identity words",
    "borderline_cases": "borderline cases between review and publish", "moderator_notes_and_audit": "audit log of moderator decisions",
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