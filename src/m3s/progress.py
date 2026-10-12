"""Per-turn progress reporting from workflow nodes to the browser.

Nodes call `report()` without knowing who listens; the engine installs a sink for
the thread that runs one turn, so progress can never leak across sessions.
"""
from contextvars import ContextVar

_sink = ContextVar("m3s_progress_sink", default=None)

# Node-level labels; finer steps are reported by the nodes themselves.
NODE_LABELS = {
    "NODE_Preprocessing": "Understand the research topic",
    "EXTRACT_PARAMETERS": "Parse constituent parameters",
    "FIND_BY_TAGS": "Retrieve candidate papers",
    "RECEIVE_DOI_REQUIREMENT": "Parse the target and tolerance",
    "CHECK_DOI_PROPERTY": "Compare the target with computed results",
    "SELECT_RESULT": "Confirm the selected structure",
    "PREPARE_EVALUATION": "Prepare structure evaluation",
    "LOAD_DATABASE": "Load existing computation records",
    "IMAGE_PROCESS": "Image processing",
    "RECONSTRUCT": "3D reconstruction",
    "VOXELIZE": "Voxelize the initial structure",
    "SYMMETRIZE": "Reflect voxel octants",
    "MORPHOLOGY": "Generate erosion and dilation candidates",
    "HOMOGENIZE": "Voxel homogenization",
    "ENGINEERING_CONSTANTS": "Calculate engineering constants",
}


def use(sink):
    """Install a `sink(label, detail)` for the current turn; returns a reset token."""
    return _sink.set(sink)


def reset(token):
    """Remove the sink installed by `use()`."""
    _sink.reset(token)


def report(label, detail=""):
    """Announce the step now starting. Never raise: progress must not break a run."""
    sink = _sink.get()
    if sink is None:
        return
    try:
        sink(str(label), str(detail))
    except Exception as exc:  # A broken listener must not abort the workflow.
        print(f"Progress report failed: {exc}", flush=True)
