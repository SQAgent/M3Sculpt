"""Recommend one DOI, ask for a target, then check its SubImages only."""
import base64
import os
import re

from m3s.computation import mode_for
from m3s.configuration import create_database_config_from_env
from m3s.doi_workflow import FIELDS, next_doi, parse_requirement, within_tolerance
from m3s.llm import chat_json
from m3s.messages import ai, last_human_text
from m3s.nodes.evaluation import nearest_result
from m3s.progress import report
from m3s.prompts import PROMPTS
from m3s.screening import contradiction, verdict_from_json
from m3s.SubAgents.TagsQuery import TagsQueryHandler

SCREEN_LIMIT = 20  # Bounds the screening cost of one recommendation turn.


class ScreeningBudgetReached(Exception):
    """Carry remaining candidates so another user turn can resume screening."""

    def __init__(self, ranked, rejected, fresh, info):
        self.updates = {"doi_ranked": ranked, "doi_rejected": rejected,
                        "doi_paper_info": info, "doi_phase": "screening_pending",
                        "messages": [ai(_skipped_note(fresh) + "Rejected 20 papers in this turn. Remaining candidates have not been screened; send Continue to screen the next batch.") ]}
        super().__init__("The relevance screening budget for this turn is exhausted")


def handler_for(state):
    # The numeric requirement is deliberately not passed to the paper retriever.
    return TagsQueryHandler(create_database_config_from_env(), state["query"], strict_errors=True)


def _paper_texts(handler, info, doi):
    paper = info.get(doi)
    if paper is None:
        paper = handler._retrieve_doi_info_from_neo4j([doi]).get(doi, {})
        info[doi] = paper
    return paper.get("title", ""), paper.get("abstract", "")


def _rejection_reason(state, handler, info, doi):
    """Return why this paper contradicts the topic, or "" when it may be shown."""
    title, abstract = _paper_texts(handler, info, doi)
    clash = contradiction(state["query"], title, abstract)
    if clash:
        return clash["reason"]
    report("Screen paper relevance", f"{doi} · {title[:60]}")
    prompt = PROMPTS["paper_relevance_screen"].format(
        user_query=state["query"], title=title, abstract=abstract[:4000])
    for _ in range(3):
        verdict = verdict_from_json(chat_json(prompt, retries=1))
        if verdict is not None:
            return verdict["reason"] if not verdict["is_relevant"] else ""
    raise RuntimeError(f"Relevance screening for paper {doi} returned invalid JSON after 3 attempts; the paper cannot be recommended")


def _screen_ranked(state, handler, ranked, info):
    """Drop leading papers that contradict the topic; keep the rest for later rounds."""
    processed = set(state.get("doi_processed", []))
    rejected = list(state.get("doi_rejected", []))
    known = {item["doi"] for item in rejected}
    fresh = []
    kept = [doi for doi in ranked if doi not in known]
    for _ in range(SCREEN_LIMIT):
        doi = next_doi(kept, processed)
        if not doi:
            break
        reason = _rejection_reason(state, handler, info, doi)
        if not reason:
            break
        entry = {"doi": doi, "reason": reason, "round": len(processed) + 1}
        rejected.append(entry)
        fresh.append(entry)
        kept = [item for item in kept if item != doi]
    else:
        if next_doi(kept, processed):
            raise ScreeningBudgetReached(kept, rejected, fresh, info)
    return kept, rejected, fresh


def _skipped_note(fresh):
    if not fresh:
        return ""
    lines = "\n".join(f"- {item['doi']}:{item['reason']}" for item in fresh)
    return f"Rejected {len(fresh)} papers that conflict with the topic:\n{lines}\n\n"


def recommend_paper(state):
    handler = handler_for(state)
    try:
        try:
            return _recommend_paper(state, handler)
        except ScreeningBudgetReached as exc:
            return exc.updates
    finally:
        close = getattr(handler, "close", None)
        if close:
            close()


def _recommend_paper(state, handler):
    fresh = []
    if "doi_ranked" not in state:
        screened = {}

        def screen(ranked, info):
            kept, rejected, new = _screen_ranked(state, handler, ranked, info)
            screened.update({"kept": kept, "rejected": rejected, "fresh": new})
            return kept, rejected

        result = handler.run(top_k=1, screen=screen)
        ranked = screened.get("kept", result.get("ranked_dois", []))
        rejected = screened.get("rejected", state.get("doi_rejected", []))
        fresh = screened.get("fresh", [])
        info = result.get("doi_info", {})
        doi = next_doi(ranked, state.get("doi_processed", []))
        answer = result.get("answer", "")
        paths = result.get("subfigures", {}).get(doi, [])
    else:
        info = state.get("doi_paper_info", {})
        ranked, rejected, fresh = _screen_ranked(state, handler, state["doi_ranked"], info)
        doi = next_doi(ranked, state.get("doi_processed", []))
        if doi:
            report("Organize paper tags", doi)
            tags = handler._get_tags_content([doi])
            report("Explain paper relevance", doi)
            answer = handler._answer_with_ai(state["query"], [doi], info, tags)
            report("Read paper structure images", doi)
            paths = handler._get_subfigures([doi]).get(doi, [])
    note = _skipped_note(fresh)
    if not doi:
        return {"messages": [ai(f"{note}The text retrieval candidates are exhausted; no structure met the tolerance. Start a new conversation to change the topic or target. This does not imply that the property is absent from the entire database.")],
                "doi_ranked": ranked, "doi_paper_info": info, "doi_rejected": rejected,
                "doi_phase": "exhausted", "waiting_for_property": False, "waiting_for_selection": False}
    count = len(state.get("doi_processed", [])) + 1
    paper = info.get(doi, {})
    prompt = "\n\nWhich property do you need? Specify a target and tolerance for Poisson ratio, Young modulus or shear modulus; include units for moduli." if not state.get("doi_requirement") else "\n\nNext, evaluate this paper's structural images against your target and tolerance."
    return {"messages": [ai(f"{note}Recommended paper {count}\n{paper.get('title', '')}\nDOI: {doi}\n\n{answer}{prompt}")],
            "doi_ranked": ranked, "doi_paper_info": info, "doi_rejected": rejected,
            "doi_processed": [*state.get("doi_processed", []), doi],
            "doi_subimages": paths, "doi_round": count,
            "retrieved_doi": doi, "retrieved_title": paper.get("title", ""),
            "retrieved_abstract": paper.get("abstract", ""),
            "doi_phase": "check" if state.get("doi_requirement") else "awaiting_property",
            "waiting_for_property": not bool(state.get("doi_requirement")),
            "waiting_for_selection": False, "sub_images": []}


def receive_requirement(state):
    content = state.get("requirement_text", last_human_text(state["messages"]))
    try:
        requirement = parse_requirement(content)
    except ValueError as exc:
        return {"messages": [ai(str(exc))], "doi_phase": "awaiting_property", "waiting_for_property": True}
    unit = "" if requirement["family"] == "poisson_ratio" else " GPa"
    components = "/".join(FIELDS[requirement["family"]]) if requirement["direction"] == "any" else requirement["direction"]
    note = "(Tolerance omitted; using the default half-step tolerance.)" if requirement["default_tolerance"] else ""
    return {"doi_requirement": requirement, "waiting_for_property": False, "doi_phase": "check",
            "messages": [ai(f"Checking target {requirement['target']:g}{unit} with tolerance +/-{requirement['tolerance']:g}{unit}. Compare engineering components and directional extrema (only the requested component when specified): {components}. {note}")]}




def check_paper_property(state):
    # Database failures must fail the run, not be reported as a scientific miss.
    best = nearest_result(state)
    requirement = state["doi_requirement"]
    hit = best is not None and within_tolerance(best["absolute_error"], requirement["tolerance"])
    audit = {"round": state["doi_round"], "doi": state["retrieved_doi"], "matched": hit,
             "mode": mode_for(state), "source": state.get("result_origin"),
             "nearest_error": best["absolute_error"] if best else None,
             "structure_id": best["structure_id"] if best else None}
    updates = {"doi_history": [*state.get("doi_history", []), audit], "doi_phase": "matched" if hit else "next", "pipeline_stage": ""}
    unit = "" if requirement["family"] == "poisson_ratio" else " GPa"
    if not hit:
        detail = f"The nearest error is {best['absolute_error']:.6g}{unit}, exceeding tolerance {requirement['tolerance']:g}{unit}" if best else "No stable structure results are available for this target comparison"
        updates["messages"] = [ai(f"DOI {state['retrieved_doi']}: {detail}. Recommending the next unprocessed paper.")]
        return updates
    path = best["image_path"]
    if not os.path.isfile(path):
        path = path.replace("/home/shangqing/sqdata/", "/database/")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The source image for the matching structure does not exist: {path}")
    blocks = [{"type": "text", "text": f"Found the nearest image within tolerance in paper {state['doi_round']}.\nDOI: {state['retrieved_doi']}\nComputed structure: {best['structure_id']}\nComponent {best['matched_field']} = {best['matched_value']:.8g}{unit}; absolute error {best['absolute_error']:.6g}{unit}.\nThis value belongs to the specific computed structure or variant associated with the source image; it does not represent every structure in the image or values reported in the paper."}]
    blocks.insert(0, {"type": "text", "text": "Result source: " + ("Existing database results in test mode (no reconstruction or computation in this run)." if mode_for(state) == "test" else "Normal mode: image processing, reconstruction, reflection and homogenization performed in this run.")})
    with open(path, "rb") as file:
        data = base64.b64encode(file.read()).decode()
    blocks.append({"type": "image", "source_type": "base64", "mime_type": "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png", "data": data,
                   "metadata": {"selection_index": 1, "name": "matched_structure"}})
    blocks.append({"type": "text", "text": "Select image 1 to confirm it as the retrieval result."})
    record = {"DOI": state["retrieved_doi"], "structure_id": best["structure_id"], "source_key": best["source_key"],
              "display_images": [path], "structure_files": best.get("structure_files", []),
              "elastic_constants": best["elastic_constants"], "compute_config": best.get("compute_config"),
              "stiffness_matrix": (best.get("properties") or {}).get("CH"), "result_origin": state.get("result_origin"),
              "geometry": best.get("geometry"), "candidate_manifest_path": best.get("candidate_manifest_path"),
              "target_matches": {requirement["family"]: {"field": best["matched_field"], "value": best["matched_value"], "target": requirement["target"], "difference": best["absolute_error"]}}}
    updates.update({"messages": [ai(blocks=blocks)], "sub_images": [path],
                    "waiting_for_selection": True, "matched_result": record,
                    "waiting_for_property": False})
    return updates


def select_result(state):
    """Confirm the result without invoking old reconstruction or FEM pipelines."""
    content = last_human_text(state.get("messages", []))
    images = state.get("sub_images", [])
    if not re.fullmatch(r"\s*1\s*", content) or len(images) != 1:
        return {"messages": [ai("Select image 1 to confirm the current result. Start a new conversation to change the topic or target.")], "waiting_for_selection": True}
    return {"messages": [ai(f"Selected image 1. Retrieval complete. DOI: {state.get('retrieved_doi', '')}.")],
            "selected_image_index": 1, "waiting_for_selection": False, "doi_phase": "selected"}


def route_doi_step(state):
    return {"check": "PREPARE_EVALUATION", "next": "FIND_BY_TAGS"}.get(state.get("doi_phase"), "END")
