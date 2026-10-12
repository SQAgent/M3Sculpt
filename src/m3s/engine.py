"""LangGraph workflow and in-process conversation sessions."""

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from contextlib import nullcontext

from langgraph.graph import END as GRAPH_END
from langgraph.graph import START, StateGraph

from m3s import progress
from m3s.nodes import doi_property_loop as loop
from m3s.nodes import evaluation
from m3s.nodes.parameters import extract_parameters
from m3s.nodes.preprocessing import NODE_Preprocessing
from m3s.routers import route_after_input
from m3s.states import SPAgentState

END = "END"

NODES = {
    "NODE_Preprocessing": NODE_Preprocessing,
    "EXTRACT_PARAMETERS": extract_parameters,
    "FIND_BY_TAGS": loop.recommend_paper,
    "RECEIVE_DOI_REQUIREMENT": loop.receive_requirement,
    "CHECK_DOI_PROPERTY": loop.check_paper_property,
    "SELECT_RESULT": loop.select_result,
    "PREPARE_EVALUATION": evaluation.prepare_evaluation,
    "LOAD_DATABASE": evaluation.load_database,
    "IMAGE_PROCESS": evaluation.process_image,
    "RECONSTRUCT": evaluation.reconstruct,
    "VOXELIZE": evaluation.voxelize,
    "SYMMETRIZE": evaluation.symmetrize,
    "MORPHOLOGY": evaluation.morphology,
    "HOMOGENIZE": evaluation.homogenize,
    "ENGINEERING_CONSTANTS": evaluation.engineering_constants,
}

TERMINAL = {"SELECT_RESULT"}
STEP_LIMIT = 10000


def _next_node(name, state):
    if name == "NODE_Preprocessing":
        return "EXTRACT_PARAMETERS"
    if name == "EXTRACT_PARAMETERS":
        return route_after_input(state)
    if name in TERMINAL:
        return END
    if name in {"FIND_BY_TAGS", "RECEIVE_DOI_REQUIREMENT", "CHECK_DOI_PROPERTY"}:
        return loop.route_doi_step(state)
    return evaluation.route_evaluation(state)


def merge(state, updates):
    """Apply node updates; `messages` appends, everything else replaces."""
    if not updates:
        return state
    for key, value in updates.items():
        if key == "messages":
            state["messages"] = [*state.get("messages", []), *value]
        else:
            state[key] = value
    return state


def build_graph():
    """Compile the paper workflow, including DOI and geometric-candidate loops."""
    graph = StateGraph(SPAgentState)

    def tracked(name, node):
        def execute(state):
            progress.report(progress.NODE_LABELS.get(name, name))
            return node(state)

        return execute

    def route(name):
        def destination(state):
            target = _next_node(name, state)
            if target != END and target not in NODES:
                raise ValueError(f"Unknown node: {target}")
            return target

        return destination

    allowed = {
        "NODE_Preprocessing": {"EXTRACT_PARAMETERS"},
        "EXTRACT_PARAMETERS": {
            *evaluation.STAGES,
            "FIND_BY_TAGS",
            "RECEIVE_DOI_REQUIREMENT",
            "PREPARE_EVALUATION",
            "SELECT_RESULT",
            END,
        },
        "FIND_BY_TAGS": {"PREPARE_EVALUATION", END},
        "RECEIVE_DOI_REQUIREMENT": {"PREPARE_EVALUATION", END},
        "CHECK_DOI_PROPERTY": {"FIND_BY_TAGS", END},
        "SELECT_RESULT": {END},
        "PREPARE_EVALUATION": {"LOAD_DATABASE", "IMAGE_PROCESS", "CHECK_DOI_PROPERTY"},
        "LOAD_DATABASE": {"CHECK_DOI_PROPERTY"},
        "IMAGE_PROCESS": {"RECONSTRUCT"},
        "RECONSTRUCT": {"VOXELIZE"},
        "VOXELIZE": {"SYMMETRIZE"},
        "SYMMETRIZE": {"MORPHOLOGY"},
        "MORPHOLOGY": {"HOMOGENIZE"},
        "HOMOGENIZE": {"ENGINEERING_CONSTANTS"},
        "ENGINEERING_CONSTANTS": {"HOMOGENIZE", "IMAGE_PROCESS", "CHECK_DOI_PROPERTY"},
    }
    for name, node in NODES.items():
        graph.add_node(name, tracked(name, node))
        destinations = {
            target: GRAPH_END if target == END else target for target in allowed[name]
        }
        graph.add_conditional_edges(name, route(name), destinations)
    graph.add_edge(START, "NODE_Preprocessing")
    return graph.compile()


def run(state, on_progress=None, state_lock=None):
    """Stream successful node updates into the session, preserving retry state."""
    graph = build_graph()
    for event in graph.stream(
        state, {"recursion_limit": STEP_LIMIT}, stream_mode="updates"
    ):
        for name, updates in event.items():
            with state_lock if state_lock is not None else nullcontext():
                merge(state, updates)
            if on_progress:
                on_progress(name, state)
    return state


class Session:
    """One conversation: state plus the status of its latest turn."""

    def __init__(self, session_id, evaluation_mode):
        self.id = session_id
        self.lock = threading.Lock()
        self.state = {"messages": [], "evaluation_mode": evaluation_mode}
        self.turn_id = None
        self.status = "idle"
        self.error = ""
        self.steps = []  # Progress of the current turn only.

    def start_step(self, label, detail=""):
        """Close the previous step and open a new one; keeps the turn's full trail."""
        with self.lock:
            now = time.time()
            if self.steps and self.steps[-1]["state"] == "active":
                self.steps[-1].update(state="done", ended=now)
            self.steps.append(
                {
                    "label": label,
                    "detail": detail,
                    "state": "active",
                    "started": now,
                    "ended": None,
                }
            )

    def finish_steps(self, status):
        """Mark the last open step done or failed once the turn ends."""
        with self.lock:
            if self.steps and self.steps[-1]["state"] == "active":
                self.steps[-1].update(
                    state="done" if status == "success" else "failed", ended=time.time()
                )

    def snapshot(self):
        """Return a JSON-safe view of the state for the browser."""
        with self.lock:
            values = {
                key: value
                for key, value in self.state.items()
                if not key.startswith("_")
            }
            return {
                "session_id": self.id,
                "status": self.status,
                "error": self.error,
                "turn_id": self.turn_id,
                "values": values,
                "steps": [dict(step) for step in self.steps],
            }


class SessionCapacityError(RuntimeError):
    """Report capacity exhaustion when all existing sessions are running."""


class Engine:
    """Runs turns in background threads and keeps bounded session history."""

    def __init__(self, evaluation_mode="normal", max_sessions=64):
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 1:
            raise ValueError("Session capacity must be a positive integer")
        self.evaluation_mode = evaluation_mode
        self.max_sessions = max_sessions
        self.sessions = OrderedDict()
        self.guard = threading.Lock()

    def create(self):
        session = Session(uuid.uuid4().hex, self.evaluation_mode)
        with self.guard:
            if len(self.sessions) >= self.max_sessions:
                victim = None
                for session_id, candidate in self.sessions.items():
                    with candidate.lock:
                        if candidate.status != "running":
                            victim = session_id
                            break
                if victim is None:
                    raise SessionCapacityError("All sessions are running. Please create a new conversation later.")
                del self.sessions[victim]
            self.sessions[session.id] = session
        return session

    def get(self, session_id):
        with self.guard:
            session = self.sessions.get(session_id)
            if session is not None:
                self.sessions.move_to_end(session_id)
            return session

    def submit(self, session, text):
        """Start one turn; reject overlapping turns instead of queueing them."""
        with self.guard:
            if self.sessions.get(session.id) is not session:
                raise KeyError("This session does not exist or has expired. Please start a new conversation.")
            with session.lock:
                if session.status == "running":
                    return None
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Please enter a message before sending.")
                self.sessions.move_to_end(session.id)
                session.turn_id = uuid.uuid4().hex
                session.status = "running"
                session.error = ""
                session.steps = []
                session.state.setdefault("messages", [])
                session.state["messages"].append({"role": "user", "content": text.strip()})
                turn_id = session.turn_id
        thread = threading.Thread(
            target=self._execute, args=(session, turn_id), daemon=True
        )
        thread.start()
        return turn_id

    def _execute(self, session, turn_id):
        token = progress.use(session.start_step)
        try:
            run(session.state, state_lock=session.lock)
            status, error = "success", ""
        except Exception as exc:  # Surface the real failure; never fake a miss.
            traceback.print_exc()
            status, error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            progress.reset(token)
        session.finish_steps(status)
        with session.lock:
            if session.turn_id == turn_id:
                session.status = status
                session.error = error
