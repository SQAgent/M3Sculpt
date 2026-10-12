"""Render the current manuscript figures from this self-contained directory."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    **{str(n): ROOT / f"figure{n}" / f"plot_figure{n}.py" for n in range(1, 7)},
    **{
        f"S{n}": ROOT / "supplementary" / f"figure_s{n}" / f"plot_figure_s{n}.py"
        for n in range(1, 6)
    },
    "examples": ROOT / "supplementary/retrieval_examples/export_examples.py",
}


def main():
    """Run the requested plotting/export scripts in separate processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", nargs="+", choices=list(SCRIPTS), default=list(SCRIPTS)
    )
    parser.add_argument("--out", type=Path, default=ROOT / "rendered")
    args = parser.parse_args()
    output = args.out.resolve()
    for key in args.figures:
        destination = output / (
            "retrieval_examples" if key == "examples" else f"figure{key}"
        )
        print(f"Rendering {key}", flush=True)
        subprocess.run(
            [sys.executable, str(SCRIPTS[key]), "--out", str(destination)],
            check=True,
        )
    print(f"All requested figures exported to {output}", flush=True)


if __name__ == "__main__":
    main()
