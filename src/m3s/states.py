"""Only the state required by DOI-first retrieval and result selection."""

from operator import add
from typing import Annotated, Any, TypedDict


class SPAgentState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], add]
    query: str
    waiting_for_property: bool
    waiting_for_selection: bool
    doi_phase: str
    doi_ranked: list[str]
    doi_processed: list[str]
    doi_rejected: list[dict[str, Any]]
    doi_paper_info: dict[str, Any]
    doi_subimages: list[str]
    doi_requirement: dict[str, Any]
    doi_round: int
    doi_history: list[dict[str, Any]]
    retrieved_doi: str
    retrieved_title: str
    retrieved_abstract: str
    sub_images: list[str]
    matched_result: dict[str, Any]
    selected_image_index: int
    evaluation_mode: str
    frozen_mode: str
    compute_config: dict[str, Any]
    evaluation_rows: list[dict[str, Any]]
    evaluation_paths: dict[str, str]
    evaluation_index: int
    result_origin: str
    pipeline_stage: str
    image_workdir: str
    current_source_key: str
    processed_image_path: str
    current_model_path: str
    current_voxel_path: str
    current_stiffness: list[list[float]]
    material_config: dict[str, Any]
    frozen_compute_config: dict[str, Any]
    parameters_valid: bool
    requirement_text: str
    initial_voxel_path: str
    symmetry_candidates: list[dict[str, Any]]
    voxel_candidates: list[dict[str, Any]]
    candidate_index: int
    candidate_manifest_path: str
