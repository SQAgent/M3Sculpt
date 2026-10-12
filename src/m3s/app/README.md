# M3Sculptor UI and workflow

This directory contains the sole application entry point. The UI and LangGraph retrieval/computation workflow run in one process.

Project components:

- `src/m3s/llm.py`: `chat`, `chat_json` and `embed`; chat defaults to the opencode CLI and can use an OpenAI-compatible HTTP endpoint.
- `src/m3s/messages.py`: plain dictionary messages with `role` and `content`.
- `src/m3s/engine.py`: the `NODES` registry, `build_graph()` and session management. Node functions accept state and return updates.
- `src/m3s/app/server.py`: standard-library `http.server`, calling `engine.Engine` directly.

## Start

Install development dependencies with `pip install -e '.[dev]'`. For normal mode, install `pip install -e '.[normal]'`, including image editing, voxel processing and image dependencies. Neo4j, MongoDB, Milvus and HTTP clients are base dependencies.

```bash
bash src/m3s/START.sh --port 8766 --mode test
```

The script runs `server.py` in the `m3s` conda environment. The server normally becomes available in about one second.

The default bind address is `0.0.0.0`, enabling LAN access. Startup prints local and LAN URLs:

```text
M3Sculptor started (mode=test), listening on 0.0.0.0:8766
  Local access: http://127.0.0.1:8766
  LAN access: http://192.168.31.7:8766
  LAN access is enabled. The API has no authentication; use it only on trusted networks.
```

The service has no authentication. Use `--host 127.0.0.1` for local access only.

Options: `--port`, `--host` (or `M3SCULPTOR_HOST`), `--mode normal|test` (or `M3SCULPTOR_MODE`), `--strict-origin` and `--allow-origin`.

If LAN clients cannot connect, check the host firewall, for example `sudo ufw allow 8766/tcp`.

## HTTP API

| Method and path | Purpose |
|---|---|
| `GET /api/config` | Return `evaluation_mode` |
| `POST /api/session` | Create a session and return `session_id` |
| `POST /api/session/{id}/message` | Submit natural-language input `{text}` for a background turn |
| `GET /api/session/{id}/state` | Poll `{status, error, values}`; status is `idle/running/success/error` |

A second submission to a running session returns 409; it is neither queued nor resent. The in-memory session limit is 64. Access updates LRU order; eviction removes only the least recently accessed nonrunning session. If all sessions are running, session creation returns 503. Restarting the service clears history. The UI reports expired sessions. Node exceptions produce `status=error` with the original failure rather than a scientific miss.

Origin checks are disabled by default. Enable `--strict-origin` or `M3SCULPTOR_STRICT_ORIGIN=1` to require matching Origin and Host names. Allow additional origins with repeated `--allow-origin https://example.com` options or comma-separated `M3SCULPTOR_ALLOW_ORIGIN`. Strict checks may reject port forwarding or SSH tunnels.

POST requires `Content-Type: application/json`. The UI sends `POST /api/session` during connection, so rejected requests appear immediately.

## Retrieval workflow

The first conversational round finds a relevant paper:

1. Rank by tags, semantics and keywords; save the candidate order and recommend one DOI at a time.
2. Apply title-level contradiction rules, then model-based relevance screening. Explain rejections and advance through candidates.
3. Analyze a relevant paper using its title, abstract and tags, then ask for a property target and tolerance.

Each recommendation screens at most 20 papers. If all are rejected and candidates remain, save progress and ask the user to send `Continue`. Every recommended paper must be screened. Invalid relevance JSON is retried at most three times, then reported as an error.

The second round searches the paper's subimages for a matching structure:

4. Test mode reads existing `StructureProperties` engineering constants for that DOI's SubImages. It skips editing, reconstruction and homogenization, and applies no target/tolerance filter to the database query.
5. Normal mode processes each distinct SubImage, reconstructs it, preserves initial 64³ voxels, independently reflects eight octants, generates erosion/dilation candidates, homogenizes each and computes engineering constants. It never queries `StructureProperties`.
6. Both modes compare supported components of stable structures and select the smallest absolute error within the DOI.
7. On a miss, recommend the next unprocessed DOI in the original text ranking. Numeric targets do not affect paper ranking.
8. On a hit, show the source image, computed variant, component, value, error and origin. Exhausted candidates produce a miss. Service failures, missing images and invalid values stop the workflow and remain errors; normal mode never falls back to stored properties.

`retrieval_vocabulary.py` contains the 36 English domain terms from SI S4.1. The keyword prompt uses their exact spelling when applicable while retaining specific terms such as honeycomb or star-shaped structures. Database taxonomy labels remain unchanged. Generated recommendations and screening reasons are requested in English.

## Modes and computation settings

Mode is set by `--mode normal|test` or `M3SCULPTOR_MODE`; the UI has no mode switch. Start a new conversation to change modes. Stored test results may use different reconstructions and constituent parameters.

The UI has one natural-language input box, without property, target, tolerance, constituent, symmetry or resolution controls. HTTP accepts `{text}` only, not `compute_config`.

Normal defaults are constituent `E = 1 GPa`, `v = 0.3`, resolution `64`, `orthorhombic` symmetry and the Python solver. Explicit natural-language constituent overrides are supported, for example:

```text
constituent Young modulus 2 GPa, constituent Poisson ratio 0.25; Young modulus 0.4 GPa, tolerance 0.01 GPa
```

Constituent and target clauses are parsed separately. Poisson ratio is dimensionless. Missing values, missing modulus units and invalid parameters prompt for clarification without advancing the current stage. Constituent parameters freeze when computation preparation begins, even if the first candidate fails. Start a new conversation to change them. Octant reflection supports only `orthorhombic`.

LangGraph uses a message reducer to append plain dictionary messages. For every distinct source image, normal mode preserves the initial reconstruction and independently reflects each of its eight octants. Initial and reflected candidates undergo six-neighbor erosion for six steps and dilation for eight steps with periodic unit-cell boundaries. Identical nonempty voxel arrays are computed once, retaining their generation aliases. Empty candidates are recorded in the manifest without solving. After all candidates are evaluated, the nearest stable result is chosen within the current DOI.

Image processing uses the existing Qwen editing interface (`AL_API_KEY`). Reconstruction uses `M3HunyuanClient` (`HUNYUAN_URL`, default `localhost:7860/generate`). Deterministic image retrieval and SubImage path derivation retain their original implementation. `M3S_WORKDIR` controls computation folders, defaulting to `src/m3s/log/doi_runs`; each image gets a separate folder.

`homogenization.py` defaults to Python periodic eight-node hexahedral FEM following the element integration, periodic node identification and strain-energy calculation in SI S8. Set `M3S_HOMOGENIZATION_BACKEND=chfem` to select the GPU adapter, and set `CHFEM_CONDA_ENV` to its installed environment, for example `chfem`. Both output GPa stiffness matrices and report failures without switching solvers or reading stored properties.

Every displayed value belongs to a specific reconstructed/reflected/morphological variant. It does not represent every structure in the source image or a value reported by the paper. Successful node updates remain in memory for retries; a failed computation does not mark the DOI as numerically infeasible. Target matching uses deterministic comparison rather than a model verdict.

Moduli use GPa, with MPa/kPa/Pa input conversion. Poisson ratio is dimensionless. Default half-step tolerances are 0.05 for Poisson ratio, 0.05 GPa for Young modulus and 0.025 GPa for shear modulus. Specify one property and one target. Ranges, multiple targets and ambiguous units prompt for clarification. Errors and tolerances include units where applicable.

Direction aliases include `E_z → E3`, `ν_xz → v13`, `G_xy → G12` and `G_xz → G31`. Unknown explicit directions prompt for clarification. Invalid stability flags, nonfinite values and missing requested components are errors. Missing matched images retain the comparison stage for retry after file repair. Chinese input aliases remain available as Unicode escapes in parser rules; application text and source documentation are English.

Loading-direction extrema use an axis-inclusive seed grid followed by multi-start continuous optimization. Transverse shear and Poisson extrema use eigenvalues of 2x2 forms. Loading-direction optimization is numerical and does not prove global optimality.

The workflow runs in a background thread inside the service process. The browser submits and polls sessions and displays successful node messages. Refresh restores the current session. Recovery checks state without resending. Image selection submits its number.

## Deployment and property matching

All imports use the `m3s` package under `src/m3s`. Put `src` on `PYTHONPATH` (the startup script does this) or install editable. Restart the service after code changes. Mock interface tests are not live retrieval results.

Supported matching fields:

- Young modulus: `E1`, `E2`, `E3`, `Emin`, `Emax`.
- Shear modulus: `G12`, `G23`, `G31`, `Gmin`, `Gmax`.
- Poisson ratio: `v12`, `v13`, `v23`, `v21`, `v31`, `v32`, `vmin`, `vmax`.

Without an explicit component, compare all supported fields for that property; otherwise compare only the requested field. Requests such as `Emax 0.7 GPa`, `Gmin 0.05 GPa`, `vmin -1.5`, `E_max` and `νmin` are supported. Matching recorded extrema does not search every intermediate value on a continuous directional surface.

`output/agent_principal_test_20260921` retains the earlier principal-component experiment. Existing cached evaluations, statistics and figures are not regenerated by source changes. Use a new `--out` directory for the English query set. Completed records with the same ID but a different query are rejected before metadata writes or database connections; `--no-cache` explicitly requests recomputation.

`eval/queries.py` copies the 16 phrase queries and 14 sentence queries from SI S5, including their order and wording. Additional query expansion produces English queries. A new test-mode paper-grid batch uses:

```bash
python -m m3s.eval.run_recall --target-grid paper --out <new-directory>
```

This evaluates 30 default queries against 42 targets with absolute tolerance 0.01. Summaries include `matching_fields` and `design_space_id`; round records identify structures, fields and errors. Configure a working model before running. `M3S_CHAT_BACKEND=http` selects an existing compatible endpoint. `M3S_CHAT_MAX_TOKENS` limits generation; `M3S_CHAT_ENABLE_THINKING=false` passes `chat_template_kwargs.enable_thinking=false` where supported. Without these settings, endpoint behavior is unchanged.

## Representative validation

```bash
PYTHONPATH=src python -m pytest -q tests
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CHFEM_CONDA_ENV=chfem python -m m3s.scripts.verify_langgraph_examples --solver chfem --resolution 64 --steps 1 --out output/langgraph_validation_20261012/gpu_64
```

Local solid, closed-cell and simple-cubic voxel examples save node traces, candidate manifests and engineering constants. GPU validation additionally compares axis ordering with the Python solver on an anisotropic cell. External retrieval, Qwen editing and Hunyuan reconstruction are outside this limited numerical verification.
