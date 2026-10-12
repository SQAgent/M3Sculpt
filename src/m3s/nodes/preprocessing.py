"""Keep the initial textual topic unchanged across property follow-up turns."""

from m3s.computation import mode_for
from m3s.messages import first_human_text


def NODE_Preprocessing(state):
    mode = mode_for(state)
    if state.get("frozen_mode") and state["frozen_mode"] != mode:
        raise ValueError("Start a new conversation to switch normal/test mode; computation sources cannot be mixed within a turn.")
    if state.get("query"):
        return {"evaluation_mode": mode, "frozen_mode": mode}
    content = first_human_text(state.get("messages", []))
    if not content.strip():
        raise ValueError("Please enter a structure name, geometric feature or research topic first.")
    return {"query": content.strip(), "evaluation_mode": mode, "frozen_mode": mode}
