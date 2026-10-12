"""Extract explicitly named constituent parameters from the natural-language box."""

import math
import re

from m3s.computation import validate_config
from m3s.messages import ai, last_human_text

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
PREFIX = r"(?:\u57fa\u4f53|\u6bcd\u6750|constituent(?:\s+material)?|base\s+material)\s*"
YOUNG = re.compile(
    rf"{PREFIX}(?:\u6768\u6c0f\u6a21\u91cf|\u5f39\u6027\u6a21\u91cf|Young(?:'s)?\s+modulus|E)\s*[:=\uff1a]?\s*({NUMBER})\s*(GPa|MPa|kPa|Pa)(?![A-Za-z])",
    re.I,
)
POISSON = re.compile(
    rf"{PREFIX}(?:\u6cca\u677e\u6bd4|Poisson(?:'s)?\s+ratio|[vν])\s*[:=\uff1a]?\s*({NUMBER})",
    re.I,
)


class MaterialInputError(ValueError):
    """Identify constituent input that the user can correct in this conversation."""


def material_parameters(text, existing=None):
    """Remove only explicit constituent clauses; leave target parsing unchanged."""
    text = text.replace("−", "-").replace("－", "-")
    config = dict(existing or {})
    factors = {"gpa": 1.0, "mpa": 1e-3, "kpa": 1e-6, "pa": 1e-9}
    young = list(YOUNG.finditer(text))
    poisson = list(POISSON.finditer(text))
    if len(young) > 1 or len(poisson) > 1:
        raise MaterialInputError("Specify only one set of constituent Young modulus and Poisson ratio.")
    if young:
        config["young_gpa"] = float(young[0][1]) * factors[young[0][2].lower()]
    if poisson:
        if re.match(r"\s*(?:GPa|MPa|kPa|Pa)(?![A-Za-z])", text[poisson[0].end():], re.I):
            raise MaterialInputError("Constituent Poisson ratio is dimensionless and must not include pressure units.")
        config["poisson"] = float(poisson[0][1])
    cleaned = POISSON.sub("", YOUNG.sub("", text))
    # Fail visibly for incomplete explicit clauses instead of inventing parameters.
    if re.search(PREFIX + r"(?:\u6768\u6c0f\u6a21\u91cf|\u5f39\u6027\u6a21\u91cf|\u6cca\u677e\u6bd4|Young(?:'s)?\s+modulus|Poisson(?:'s)?\s+ratio|E\b|[vν]\b)", cleaned, re.I):
        raise MaterialInputError("Incomplete constituent parameters. Example: constituent Young modulus 1 GPa, constituent Poisson ratio 0.3.")
    if young and (not math.isfinite(config["young_gpa"]) or config["young_gpa"] <= 0):
        raise MaterialInputError("Constituent Young modulus must be a positive finite number.")
    if poisson and (not math.isfinite(config["poisson"]) or not -1 < config["poisson"] < .5):
        raise MaterialInputError("Constituent Poisson ratio must be a finite number within (-1, 0.5).")
    return validate_config(config), cleaned


def extract_parameters(state):
    """Keep constituent defaults unless the user explicitly supplies replacements."""
    text = last_human_text(state.get("messages", []))
    try:
        config, cleaned = material_parameters(text, state.get("material_config"))
        frozen = state.get("frozen_compute_config")
        if frozen and config != frozen:
            raise MaterialInputError("Constituent parameters are fixed once computation preparation starts. Start a new conversation to change them, or send Continue to retry the current computation.")
    except MaterialInputError as exc:
        return {"messages": [ai(str(exc))], "parameters_valid": False}
    return {"material_config": config, "requirement_text": cleaned, "parameters_valid": True}
