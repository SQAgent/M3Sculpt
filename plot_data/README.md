# Manuscript figure source bundle

This directory contains local plotting code and source data for the current manuscript's Figures 1–6, supplementary Figures S1–S5, and four supplementary retrieval illustrations. Figure numbers follow the current manuscript, rather than the historical names of the original plotting folders.

All code, documentation, editable-source page names and query labels are in English. Numeric values, stiffness matrices, voxel arrays and source-image pixels are retained. Query IDs and the original evaluation ordering remain unchanged when their labels are translated.

## Install and run

```bash
python -m pip install -r plot_data/requirements.txt
python plot_data/plot_all.py
```

The requirements record the package versions used for validation. Arial is used when installed; font availability and graphics-library versions can affect rendering appearance. VTK performs offscreen rendering for Figures 4 and 6. No database, `.env`, paper directory, reconstruction endpoint, language model or finite-element solver is needed.

Scripts resolve input paths relative to their own location. The entire directory can be copied elsewhere and run independently.

By default, the combined runner writes results to `plot_data/rendered/`, keeping them separate from source data. Select figures or choose another output directory:

```bash
python plot_data/plot_all.py --figures 3 4 5 6 --out output/figure_exports
python plot_data/figure4/plot_figure4.py --out output/figure4
python plot_data/figure5/plot_figure5.py --out output/figure5
```

Figures 3–6 export PNG, PDF and SVG with the original publication layout (600 DPI, Arial 10 pt). Figure 1 uses its original procedural drawing and exports PNG/SVG. Preserved supplementary and workflow graphics export their original PNG/SVG assets and a PDF made from the supplied raster source.

## Main figures

| Figure | Content | Code | Local inputs |
|---|---|---|---|
| 1 | Knowledge-driven design workflow | `figure1/plot_figure1.py` | `source_data.json`; deterministic FCC geometry and drawing coordinates in the copied script |
| 2 | M3Sculptor lifecycle | `figure2/plot_figure2.py` | Editable `source.drawio`, design sidecars, `source.svg`, `source.png` |
| 3 | Knowledge graph, mechanical MCQ evaluation and MMLU | `figure3/plot_figure3.py` | `knowledge_graph.svg`, `source_data.json` |
| 4 | Source images, preprocessing, reconstruction and elastic responses | `figure4/plot_figure4.py` | `source_data.json`, four pairs of source/processed images and four voxel arrays |
| 5 | Tolerance and candidate-set comparison | `figure5/plot_figure5.py` | `per_query.json` plus four source-data CSV tables |
| 6 | Initial/expanded structures and target-property examples | `figure6/plot_figure6.py` | `source_data.json`, four initial and four selected voxel arrays |

Shared publication layout, directional elasticity and rendering functions are copied into `_shared/`. The numerical and geometric calculations follow the original plotting functions. Generated images do not overwrite JSON, CSV or voxel inputs.

Figure 1 is a conceptual illustration. Its FCC geometry and morphological changes are procedural examples; its stiffness-matrix icon contains symbols rather than simulated property values.

Figure 2's original authoring source is Draw.io. Its Python adapter exports the preserved source graphics. Edit `source.drawio` in Draw.io to change the diagram; the adapter does not reconstruct its layout from measured numerical data.

Figure 3's mechanical epoch curves are recovered from the saved SVG coordinates used by the original publication script. Baselines, additional-model counts and MMLU records come from saved source audits. Full training/evaluation corpora are unnecessary for redrawing these plotted results, and missing raw epoch records are not invented.

Figure 5 keeps raw per-case error traces, which are required to reconstruct the violin distributions faithfully. The CSV tables supply success rates, mean papers examined, distribution summaries and property-specific success rates. These records belong to the original experiment; translating query labels does not rerun retrieval or change its results.

## Supplementary graphics

| Figure | Content | Code | Source type |
|---|---|---|---|
| S1 | Morphological erosion/dilation schematic | `supplementary/figure_s1/plot_figure_s1.py` | Preserved PNG |
| S2 | Volume-preserving Laplacian smoothing schematic | `supplementary/figure_s2/plot_figure_s2.py` | Preserved PNG |
| S3 | Cellular and lattice structure examples | `supplementary/figure_s3/plot_figure_s3.py` | Preserved SVG and PNG |
| S4 | Coupled-response and tubular structures | `supplementary/figure_s4/plot_figure_s4.py` | Preserved SVG and PNG |
| S5 | Auxetic structure examples | `supplementary/figure_s5/plot_figure_s5.py` | Preserved SVG and PNG |

Original drawing algorithms and numerical datasets for these supplementary illustrations are unavailable in the project. Their adapters preserve and export the supplied visual sources; they do not claim a numerical reconstruction. Figure S1 here is the morphology schematic from the SI, rather than the historical standalone MMLU plot that is now incorporated into main Figure 3(d).

`supplementary/retrieval_examples/` contains the four source JPEGs referenced in the SI examples and `export_examples.py`, which exports lossless PNG copies of their decoded pixels.

## Source integrity

Only this copied bundle is adapted for local paths, current figure numbers and English text. The manuscript, SI, original plotting scripts and original experiment files remain unchanged. Source images and voxel arrays are copied without resampling; the core directional-elasticity and mesh-rendering functions are retained.
