"""Routing for the single supported DOI-first retrieval workflow."""


def route_after_input(state):
    if state.get("parameters_valid") is False:
        return "END"
    if state.get("pipeline_stage"):
        from m3s.nodes.evaluation import STAGES
        if state["pipeline_stage"] not in STAGES:
            raise ValueError("Unknown computation stage; it cannot be skipped automatically")
        return state["pipeline_stage"]
    if state.get("waiting_for_property"):
        return "RECEIVE_DOI_REQUIREMENT"
    if state.get("doi_phase") == "check" and state.get("doi_requirement"):
        return "PREPARE_EVALUATION"
    if state.get("waiting_for_selection"):
        return "SELECT_RESULT"
    if state.get("doi_phase") in {"matched", "selected", "exhausted"}:
        return "END"
    return "FIND_BY_TAGS"
