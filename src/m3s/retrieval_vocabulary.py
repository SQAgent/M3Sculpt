"""English retrieval terminology from SI S4.1; runtime data is self-contained."""

import re

PAPER_KEYWORDS = (
    'Metamaterial',
    'Origami',
    'Kirigami',
    'Lattice',
    'Chiral',
    'Tubular',
    'Multistable Structure',
    'Elasticity',
    'Dilatational',
    'Auxetic',
    "Negative Poisson's Ratio",
    'Programmable',
    'Ultra-light and High-strength',
    'Negative Stiffness',
    'Multistability',
    'Negative Thermal Expansion',
    'Zero Expansion',
    'Shape Memory',
    'Deformation Recovery',
    'Nonlinear Response',
    'Anisotropy',
    'Compression-torsion Coupling',
    'Impact Resistance',
    'Energy Absorption',
    'Buffering and Energy Absorption',
    'Low-frequency Vibration Isolation',
    'Magneto-mechanical',
    'Hashin-Shtrikman',
    'Gibson-Ashby',
    'Effective Medium Theory',
    'Parameterization',
    'Multiscale',
    'Homogenization',
    'Topology Optimization',
    'Inverse Design',
    'Digital Twin',
)


def _term_key(term):
    """Ignore case, apostrophes and separators when comparing paper terminology."""
    return re.sub(r"\s+", " ", re.sub(r"[\u2019'_-]+", " ", term.lower())).strip()


_TERMS = {_term_key(term): term for term in PAPER_KEYWORDS}
_TERMS["negative poisson"] = "Negative Poisson's Ratio"
_TERMS["negative poisson ratio"] = "Negative Poisson's Ratio"


def canonical_keywords(keywords):
    """Use SI spelling where applicable and retain additional specific English terms."""
    if not isinstance(keywords, list):
        raise ValueError("Retrieval keywords must be a list of English strings")
    result, seen = [], set()
    for term in keywords:
        if not isinstance(term, str) or not term.strip():
            raise ValueError("Retrieval keywords must be nonempty English strings")
        if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", term):
            raise ValueError("Retrieval keywords must be in English")
        key = _term_key(term.strip())
        canonical = _TERMS.get(key, term.strip())
        if canonical.casefold() not in seen:
            result.append(canonical)
            seen.add(canonical.casefold())
    return result
