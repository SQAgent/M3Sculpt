"""Export preserved source graphics without reconstructing missing drawing data."""

import argparse
import shutil
from pathlib import Path

from PIL import Image


def export_asset(data_dir, name):
    """Preserve PNG/SVG sources and produce a PDF from the supplied raster asset."""
    data_dir = Path(data_dir)
    parser = argparse.ArgumentParser(description="Export the preserved source graphic")
    parser.add_argument("--out", type=Path, default=data_dir / "rendered")
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = data_dir / "source.png"
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, out / f"{name}.png")
    svg = data_dir / "source.svg"
    if svg.is_file():
        shutil.copy2(svg, out / f"{name}.svg")
    with Image.open(source) as image:
        if image.mode == "RGBA":
            canvas = Image.new("RGBA", image.size, "white")
            canvas.alpha_composite(image)
            image = canvas
        image.convert("RGB").save(out / f"{name}.pdf", resolution=600)
    print(f"{name}: exported preserved source graphic to {out}", flush=True)


def export_examples(data_dir):
    """Export the four supplied retrieval illustrations to lossless PNG files."""
    data_dir = Path(data_dir)
    parser = argparse.ArgumentParser(
        description="Export supplementary retrieval images"
    )
    parser.add_argument("--out", type=Path, default=data_dir / "rendered")
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for source in sorted(data_dir.glob("example_*.jpeg")):
        with Image.open(source) as image:
            image.save(out / source.with_suffix(".png").name)
    print(f"Retrieval illustrations exported to {out}", flush=True)
