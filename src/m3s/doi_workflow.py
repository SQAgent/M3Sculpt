"""Pure contracts for DOI-first retrieval; no model or database dependencies."""
import math
import re

# Match principal engineering components and stored full-direction extrema.
FIELDS = {
    "poisson_ratio": ["v12", "v13", "v23", "v21", "v31", "v32", "vmin", "vmax"],
    "young_modulus": ["E1", "E2", "E3", "Emin", "Emax"],
    "shear_modulus": ["G12", "G23", "G31", "Gmin", "Gmax"],
}
STEPS = {"poisson_ratio": 0.1, "young_modulus": 0.1, "shear_modulus": 0.05}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def normalize_components(text):
    """Translate Cartesian component names and reject unsupported directions."""
    text = re.sub(r"([EGvν])_\{([^{}]+)\}", r"\1_\2", text, flags=re.I)
    pattern = r"(?<![A-Za-z0-9_])([EGvν])_?(max|min|[xyz]{1,2}|\d{1,2})(?![A-Za-z0-9_])"

    def replace(match):
        prefix = {"e": "E", "g": "G", "v": "v", "ν": "v"}[match[1].lower()]
        suffix = match[2].lower()
        if suffix not in {"max", "min"}:
            suffix = suffix.translate(str.maketrans("xyz", "123"))
        if prefix == "G":
            suffix = {"21": "12", "32": "23", "13": "31"}.get(suffix, suffix)
        field = prefix + suffix
        if field not in sum(FIELDS.values(), []):
            raise ValueError(f"Unsupported direction component {match[0]}; use principal components on x/y/z axes or min/max extrema.")
        return field

    text = re.sub(pattern, replace, text, flags=re.I)
    if re.search(r"(?<![A-Za-z0-9_])[EGvν]_[A-Za-z0-9\u0370-\u03ff]+|(?<![A-Za-z0-9_])[EGvν][xyzθφϕ]{1,3}(?![A-Za-z0-9_])", text, re.I):
        raise ValueError("Unrecognized direction notation; specify a principal component or min/max extremum.")
    return text


def parse_requirement(text):
    """Accept one explicit equality target, never infer a value or inequality."""
    text = text.replace("−", "-").replace("－", "-")
    text = normalize_components(text)
    names = {
        "poisson_ratio": r"\u6cca\u677e\u6bd4|poisson(?:[’']?s)?(?:\s+ratio)?|(?<!\w)[vν](?:max|min)?(?![A-Za-z])",
        "young_modulus": r"\u6768\u6c0f\u6a21\u91cf|\u5f39\u6027\u6a21\u91cf|young(?:[’']?s)?(?:\s+modulus)?|(?<!\w)E(?:max|min)?(?![A-Za-z])",
        "shear_modulus": r"\u526a\u5207\u6a21\u91cf|shear(?:\s+modulus)?|(?<!\w)G(?:max|min)?(?![A-Za-z])",
    }
    families = [key for key, pattern in names.items() if re.search(pattern, text, re.I)]
    if len(families) != 1 or re.search(r"\u81f3\u5c11|\u81f3\u591a|\u5927\u4e8e|\u5c0f\u4e8e|\u9ad8\u4e8e|\u4f4e\u4e8e|[<>≤≥]|at least|at most", text, re.I):
        raise ValueError("Specify one property target and tolerance, for example: Poisson ratio -1.2, tolerance 0.05. Matching uses target +/- tolerance.")
    family = families[0]
    directions = [field for field in FIELDS[family] if re.search(rf"(?<!\w){field}(?![A-Za-z0-9])", text, re.I)]
    if len(directions) > 1:
        raise ValueError("Specify one direction component at a time, or leave direction unspecified.")
    tol = re.search(rf"(?:\u5bb9\u5dee|tolerance|±|\+/-)\s*[:=\uff1a]?\s*({NUMBER})\s*(GPa|MPa|kPa|Pa)?", text, re.I)
    clean = text[:tol.start()] + text[tol.end():] if tol else text
    for field in sorted(FIELDS[family], key=len, reverse=True):
        clean = re.sub(rf"(?<!\w){field}(?![A-Za-z0-9])", "", clean, flags=re.I)
    numbers = list(re.finditer(rf"({NUMBER})\s*(GPa|MPa|kPa|Pa)?", clean, re.I))
    if len(numbers) != 1:
        raise ValueError("Specify one explicit target value; do not combine multiple targets or ranges in one requirement.")
    raw = numbers[0]
    target_unit = (raw.group(2) or "").lower()
    factors = {"gpa": 1, "mpa": .001, "kpa": .000001, "pa": .000000001}
    if family == "poisson_ratio" and (target_unit or (tol and tol.group(2))):
        raise ValueError("Poisson ratio is dimensionless and must not include pressure units.")
    if family != "poisson_ratio" and not target_unit:
        raise ValueError("Specify modulus units, for example: Young modulus 0.4 GPa, tolerance 0.05 GPa.")
    target = float(raw.group(1)) * factors.get(target_unit, 1)
    tolerance = float(tol.group(1)) * factors.get((tol.group(2) or target_unit).lower(), 1) if tol else STEPS[family] / 2
    if not math.isfinite(target) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("The target and tolerance must be finite numbers, and tolerance must be nonnegative.")
    if family != "poisson_ratio" and target < 0:
        raise ValueError("The target modulus of a stable structure cannot be negative.")
    return {"family": family, "target": target, "tolerance": tolerance,
            "direction": directions[0] if directions else "any", "default_tolerance": tol is None}


def source_paths(doi, paths):
    result = {}
    for path in paths:
        match = re.search(r"([a-f0-9]{64})/structure/(\d+)\.(?:png|jpe?g)$", path.replace("\\", "/"), re.I)
        if match:
            result[f"{doi}_{match[1]}_{match[2]}"] = path
    return result


def closest_row(rows, paths, requirement):
    """Find the nearest engineering component or directional extremum in stable DOI rows."""
    allowed = FIELDS[requirement["family"]]
    direction = requirement["direction"]
    if direction != "any" and direction not in allowed:
        raise ValueError(f"Unsupported target component {direction}; supported engineering components and directional extrema: {'/'.join(allowed)}.")
    fields = allowed if direction == "any" else [direction]
    best = None
    for row in rows:
        sid = row.get("structure_id", "")
        source = next((key for key in paths if sid == key or sid.startswith(key + "_")), None)
        if source is None:
            continue
        constants = row.get("elastic_constants")
        if not isinstance(constants, dict) or not isinstance(constants.get("is_stable"), bool):
            raise ValueError(f"Computation record {sid} lacks valid engineering constants or a stability flag")
        if not constants["is_stable"]:
            continue
        usable = False
        for field in fields:
            if field not in constants and direction == "any":
                continue
            value = constants.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Computation record {sid}: {field} is missing or is not a finite number")
            usable = True
            error = abs(value - requirement["target"])
            order = (error, sid, field)
            if best is None or order < best[0]:
                best = (order, {**row, "source_key": source, "image_path": paths[source],
                               "matched_field": field, "matched_value": value, "absolute_error": error})
        if not usable:
            raise ValueError(f"Stable computation record {sid} has no engineering constants for this property")
    return best[1] if best else None


def within_tolerance(error, tolerance):
    return error <= tolerance or math.isclose(error, tolerance, rel_tol=1e-12, abs_tol=1e-12)


def next_doi(ranked, processed):
    seen = {doi.lower().strip() for doi in processed}
    return next((doi for doi in ranked if doi and doi.lower().strip() not in seen), None)
