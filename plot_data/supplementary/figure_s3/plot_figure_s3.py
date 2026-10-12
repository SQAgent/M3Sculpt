"""Export the preserved source graphic for Figure S3."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared.asset_export import export_asset

if __name__ == "__main__":
    export_asset(Path(__file__).resolve().parent, "figureS3")
