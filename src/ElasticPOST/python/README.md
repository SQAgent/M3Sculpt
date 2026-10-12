# ElasticPOST Python：Figure 5

This is a standalone Python rewrite of the directional Young, shear and Poisson
surface calculations used by MATLAB `ElasticPlot_3D.m`, `Young_3D.m`,
`Shear_3D.m`, and `Poisson_3D.m`. It does not recreate the MATLAB GUI.

## Run from repository root

```bash
python src/ElasticPOST/python/select_examples.py
python src/ElasticPOST/python/verify_elasticpost.py
python src/ElasticPOST/python/plot_figure5.py
```

Dependencies: numpy, scipy, matplotlib >= 3.7, scikit-image, pymongo,
python-dotenv, vtk (offscreen EGL rendering). Database credentials are read from the repository `.env` and are
never included in exported files. The selection is read-only.

## Selection and normalization

Search only mechanically stable StructureProperties records in DS3d_6. Join
Structures using the same design_space_id and structure_id, and reject geometry
paths outside /database/MetamaterialProject/DS3d_6.

Rank (a) by the largest of v12/v13/v23/v21/v31/v32, (b) by the smallest of these
six components, (c) by the largest of E1/f, E2/f, E3/f, and (d) by the largest of
G12/f, G23/f, G31/f. **vmax/vmin/Emax/Emin/Gmax/Gmin do not participate in
selection.** Every panel annotates the selected engineering component.
`f = count_nonzero(model.npy) / model.npy.size` is computed for each actual
binary voxel variant, including its symmetry and morphology. Moduli remain in
GPa because volume fraction is dimensionless. Do not substitute the original
variant's fraction for a modified variant. No printability filter is imposed.

The directional cloud is supplementary: nu_max(n) for (a), nu_min(n) for (b),
E(n)/f for (c), and G_max(n)/f for (d). Here max/min denotes the envelope over
the transverse direction at fixed loading n; it is not the selection criterion.
Database notation follows elastic_constants.py: v_ij = -S_ij / S_jj, so the
second index is the loading axis. E_i = 1/S_ii; G12 = 1/S66, G23 = 1/S44,
G31 = 1/S55 in engineering Voigt notation.

The source material is identified in every selected record. BM_1 is E = 1 GPa,
nu = 0.3. Figure values belong to computed structures, not the source papers.
The selector fails on missing geometry and nonbinary/empty voxel arrays.
The volume cache can be deleted to rescan changed database geometry.

## Mathematical conventions

Engineering Voigt order is xx, yy, zz, yz, xz, xy. The rendering uses
`C = (CH + CH.T)/2` to remove solver reciprocity noise, rejects nonpositive
stiffness eigenvalues and inverts to obtain S. It analytically diagonalizes
the two-dimensional transverse compliance for each loading direction. Thus
G_max(n) and nu_min/max(n) are exact with respect to the transverse angle;
loading directions are sampled on a 361 × 721 angular sphere. Figure annotations
report the selected database engineering component; recomputed directional
ranges are retained separately in render_audit.json. This improves on the MATLAB
finite chi grid. E has no transverse-angle dependence.

The surface radius is the absolute field value; the color retains the signed
Poisson value. This avoids folding negative-radius surfaces through the origin.
Surface colors have no additional lighting modulation, so they match their
colorbars exactly. Neutral geometry renderings are shaded independently.
Depth-buffered VTK rendering avoids painter-order artifacts in highly anisotropic
surfaces; scalar interpolation precedes the color lookup. The geometry is taken directly from the selected voxel file using marching cubes.

## Color and outputs

Use the paper palette: `BrBG_r` for Poisson (green negative / pale zero / brown
positive), and `PuBuGn` for both normalized moduli. The two Poisson panels share
a zero-centered symmetric range. E/f and G/f share one range starting at zero.
Colorbars have exactly three ticks. Symmetry labels use #7890AC; morphology
labels use #26768C; the computed geometry uses #8A939C.

Outputs go to achievements/paper/plt/Figure5: figure5.png/pdf/svg, individual
figure5a–d.png, selected_examples.json, volume_fractions.json, render_audit.json.
The PDF/SVG retain editable text; detailed meshes are rasterized for manageable
file sizes. selected_examples.json retains complete selected database metadata
and CH matrices, while render_audit.json records the actual plotted ranges.

Validation includes the isotropic closed-form E, nu, G solution and an independent
comparison with m3s.elastic.core on a positive-definite anisotropic matrix,
using 20,001 transverse angles (absolute agreement better than 1e-6).
