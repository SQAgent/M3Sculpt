"""Relevance screening for retrieved papers; pure text rules, no database access."""
import re

# Each entry: the property the user asks for, and the wordings that contradict it.
CONTRADICTIONS = [
    {
        "name": "high_stiffness",
        "wanted": r"\u9ad8\u521a\u5ea6|\u9ad8\u521a\u6027|\u5927\u521a\u5ea6|\u521a\u5ea6\u9ad8|high[\s-]*stiff(?:ness)?|stiff(?:ness)?[\s-]*maximi|ultra[\s-]*stiff|high[\s-]*modulus|stiffest",
        "forbidden": r"\u8d1f\u521a\u5ea6|\u96f6\u521a\u5ea6|\u51c6\u96f6\u521a\u5ea6|negative[\s-]*stiff(?:ness)?|zero[\s-]*stiff(?:ness)?|quasi[\s-]*zero[\s-]*stiff(?:ness)?|snap[\s-]*through|multistable|bistable",
        "reason": "The user requests high stiffness, but the paper studies negative/zero stiffness or bistable buckling, which conflicts with that objective",
    },
    {
        "name": "stiffness_vs_damping",
        "wanted": r"\u9ad8\u521a\u5ea6|\u9ad8\u521a\u6027|\u5927\u521a\u5ea6|\u521a\u5ea6\u9ad8|high[\s-]*stiff(?:ness)?|stiff(?:ness)?[\s-]*maximi|ultra[\s-]*stiff|high[\s-]*modulus|stiffest",
        "forbidden": r"\u51cf\u632f|\u9694\u632f|\u963b\u5c3c|\u5438\u58f0|\u9694\u58f0|\u566a\u58f0|vibration[\s-]*damping|damping[\s-]*(?:propert|behavio|capacit|performance)|vibration[\s-]*(?:isolation|attenuation|mitigation|suppression|absorption)|sound[\s-]*(?:insulation|absorption|transmission)|acoustic[\s-]*(?:absorption|attenuation)|noise[\s-]*reduction",
        "reason": "The user requests high stiffness, but the paper targets vibration damping or acoustic performance; stiffness is only an incidental measurement",
    },
    {
        "name": "negative_poisson",
        "wanted": r"\u8d1f\u6cca\u677e\u6bd4|\u62c9\u80c0|auxetic|negative[\s-]*poisson",
        "forbidden": r"positive[\s-]*poisson[\s'’]*s?[\s-]*ratio|\u6b63\u6cca\u677e\u6bd4",
        "reason": "The user requests a negative Poisson ratio, but the paper explicitly studies a positive Poisson ratio",
    },
    {
        "name": "lightweight",
        "wanted": r"\u8f7b\u8d28|\u8f7b\u91cf\u5316|light[\s-]*weight|lightweight|low[\s-]*density",
        "forbidden": r"fully[\s-]*dense|solid[\s-]*bulk[\s-]*material",
        "reason": "The user requests a lightweight porous structure, but the paper studies fully dense bulk material",
    },
]


def _hits(pattern, text):
    return len(re.findall(pattern, text or "", re.I))


def contradiction(query, title, abstract=""):
    """Reject only on a title-level conflict; abstract mentions stay advisory."""
    for rule in CONTRADICTIONS:
        if not _hits(rule["wanted"], query):
            continue
        if not _hits(rule["forbidden"], title):
            continue
        if _hits(rule["wanted"], title):
            continue  # The title claims both, so text alone cannot settle it.
        return {"rule": rule["name"], "reason": rule["reason"]}
    return None


def verdict_from_json(payload):
    """Normalise the screening model output; unusable output is not a rejection."""
    if not isinstance(payload, dict):
        return None
    value = payload.get("is_relevant")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "\u76f8\u5173"}:
            value = True
        elif lowered in {"false", "no", "\u4e0d\u76f8\u5173"}:
            value = False
    if not isinstance(value, bool):
        return None
    reason = payload.get("reason")
    reason = reason.strip() if isinstance(reason, str) and reason.strip() else "No reason provided"
    return {"is_relevant": value, "reason": reason[:200]}
