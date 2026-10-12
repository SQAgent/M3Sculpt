"""Depth-buffered offscreen rendering to avoid painter-order surface artifacts."""

import matplotlib.pyplot as plt
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


def render(vertices, faces, values=None, cmap=None, norm=None, size=1100):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices, dtype=float), deep=True))
    cells = vtk.vtkCellArray()
    packed = np.column_stack([np.full(len(faces), 3), faces]).astype(np.int64).ravel()
    cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(packed, deep=True))
    data = vtk.vtkPolyData()
    data.SetPoints(points)
    data.SetPolys(cells)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data)
    if values is not None:
        scalars = numpy_to_vtk(
            np.ascontiguousarray(norm(values), dtype=float), deep=True
        )
        data.GetPointData().SetScalars(scalars)
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(1024)
        lut.SetRange(0, 1)
        for i, color in enumerate(plt.get_cmap(cmap)(np.linspace(0, 1, 1024))):
            lut.SetTableValue(i, *color)
        lut.Build()
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0, 1)
        mapper.InterpolateScalarsBeforeMappingOn()
    else:
        mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if values is not None:
        actor.GetProperty().LightingOff()
    else:
        actor.GetProperty().SetColor(138 / 255, 147 / 255, 156 / 255)
        actor.GetProperty().SetAmbient(0.25)
        actor.GetProperty().SetDiffuse(0.75)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.AddActor(actor)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(size, size)
    window.SetMultiSamples(8)
    window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(3, 3, 2)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 0, 1)
    camera.ParallelProjectionOn()
    renderer.ResetCamera()
    camera.Zoom(1.0)
    renderer.ResetCameraClippingRange()
    window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.ReadFrontBufferOff()
    capture.Update()
    im = capture.GetOutput()
    result = (
        vtk_to_numpy(im.GetPointData().GetScalars()).reshape(size, size, 3)[::-1].copy()
    )
    window.Finalize()
    nonwhite = np.any(result < 248, axis=-1)
    y, x = np.where(nonwhite)
    if len(x):
        pad = 25
        result = result[
            max(0, y.min() - pad) : min(size, y.max() + pad),
            max(0, x.min() - pad) : min(size, x.max() + pad),
        ]
    return result


def surface(n, values, cmap, norm):
    vertices = (n * np.abs(values)[..., None]).reshape(-1, 3)
    rows, cols = values.shape
    a = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
    faces = np.concatenate(
        [
            np.stack([a, a + cols, a + 1], axis=-1),
            np.stack([a + 1, a + cols, a + cols + 1], axis=-1),
        ]
    )
    return render(vertices, faces, values.ravel(), cmap, norm)
