"""Export the preserved source graphic for supplementary retrieval illustrations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared.asset_export import export_examples

if __name__ == "__main__":
    export_examples(Path(__file__).resolve().parent)
