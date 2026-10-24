import re

BAD_WORDS = frozenset("""
idiot idiots stupid moron morons dumb dumbass imbecile loser losers jerk scum trash garbage pathetic
ignorant liar liars fool fools clown clowns sucks suck crap damn bastard bitch bullshit shit shitty
fuck fucking fucked asshole ass dick douche retard retarded racist pig pigs nazi fascist coward cowards
hypocrite traitor thug thugs filthy disgusting worthless hate hateful
""".split())

_TOKEN = re.compile(r"[a-z']+")


def rule_score(text: str) -> int:
    return sum(tok in BAD_WORDS for tok in _TOKEN.findall((text or "").lower()))