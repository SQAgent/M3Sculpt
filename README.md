# M3Sculptor / m3s

M3Sculptor is a literature retrieval-augmented generation (RAG) assistant for mechanical metamaterials. It uses LangGraph StateGraph to connect paper retrieval, relevance screening, structural reconstruction and physical evaluation. The browser interface and workflow run in a single Python process.

[Chinese README](docs/README.zh-CN.md) · [Application documentation](src/m3s/app/README.md) · [Environment template](.env.example)

## Workflow

1. Enter a research topic. The assistant retrieves papers through Neo4j tags, Milvus semantic search and keyword matching, screens their relevance, and analyzes one relevant paper.
2. Specify a property target and tolerance. The assistant evaluates structures associated with that paper's subimages, returns the nearest matching image for confirmation, or advances to the next paper.

Paper ranking combines semantic and keyword scores with equal weights. Numeric targets do not affect the paper order. Retrieval terminology uses the 36 English domain keywords from SI S4.1; the 16 phrase queries and 14 sentence queries in the evaluation benchmark match SI S5 exactly. Recommendations, progress messages, errors and interface text are English. Chinese input aliases remain supported by the property parser.

## Requirements

- Python 3.10 or newer; conda is needed by `START.sh` and the optional CHFEM adapter.
- Existing Neo4j and MongoDB databases, and a running Milvus service with the paper corpus and `entity` collection.
- A configured chat backend: the opencode CLI or an OpenAI-compatible HTTP endpoint.
- An HTTP embedding endpoint using the same model and vector dimension as the existing Milvus corpus.
- Normal mode additionally needs an image-editing API key and a Hunyuan3D reconstruction service.

The repository does not provision the databases or populate the literature corpus. Configure the services and corpus paths for your deployment.

## Install and configure

```bash
conda create -n m3s python=3.11
conda activate m3s
pip install -e '.[dev]'
cp -n .env.example .env
```

Edit `.env` to replace the example credentials, service addresses, model names and database names. `cp -n` preserves an existing `.env`. The application loads `.env` through `python-dotenv`; already exported variables take priority. The template contains placeholders only.

Install live-computation dependencies for normal mode:

```bash
pip install -e '.[normal]'
```

Both optional groups can be installed together with `pip install -e '.[normal,dev]'`.

### Model configuration

`M3S_CHAT_BACKEND=opencode` uses the opencode CLI. Set `M3S_OPENCODE_MODEL` to a provider/model configured in your installation. `M3S_OPENCODE_BIN`, `M3S_OPENCODE_AGENT` and `M3S_OPENCODE_WORKDIR` are optional overrides.

To use an HTTP chat endpoint, set `M3S_CHAT_BACKEND=http`, then configure `VLM_MODEL`, `VLM_API_KEY` and `VLM_URL`. The `RETRIEVAL_LLM_*` names are fallbacks when corresponding `VLM_*` settings are empty. HTTP URLs are base URLs such as `http://127.0.0.1:8000/v1`; the client appends `/chat/completions` or `/embeddings`.

`EMBED_MODEL`, `EMBED_API_KEY` and `EMBED_URL` are needed for semantic retrieval with either chat backend. Embeddings always use HTTP.

### Service configuration

| Variables | Purpose |
|---|---|
| `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | Paper knowledge graph |
| `MONGO_URL`, `MONGO_DATABASE` | MongoDB connection and database |
| `MONGO_PERFORMANCE_DATABASE`, `MONGO_PERFORMANCE_DESIGN_SPACE` | Stored property population for test mode; defaults are `MechanicalMetamaterialProjectDatabase` and `DS3d_6` |
| `MILVUS_URL`, `MILVUS_USERNAME`, `MILVUS_PASSWORD`, `MILVUS_DATABASE` | Semantic paper retrieval |
| `AL_API_KEY`, `HUNYUAN_URL` | Image editing and Hunyuan3D reconstruction in normal mode |
| `M3S_WORKDIR` | Output directory for computed structures |
| `M3S_HOMOGENIZATION_BACKEND` | `python` by default; `chfem` selects the GPU adapter |
| `CHFEM_CONDA_ENV`, `CHFEM_CONDA_TIMEOUT` | Environment and subprocess timeout for the optional CHFEM adapter |

For local and network deployment settings, see `.env.example` and the application documentation.

## Run

Start with existing property records:

```bash
bash src/m3s/START.sh --host 127.0.0.1 --port 8766 --mode test
```

Open `http://127.0.0.1:8766`. For live reconstruction and computation, replace `--mode test` with `--mode normal`.

The startup script uses the `m3s` conda environment. To select another environment, set `M3SCULPTOR_CONDA_ENV` in the shell before launching; placing it in `.env` alone does not affect the shell script:

```bash
M3SCULPTOR_CONDA_ENV=my-env bash src/m3s/START.sh --host 127.0.0.1 --port 8766 --mode test
```

Alternatively, run directly in an activated environment:

```bash
PYTHONPATH=src python -m m3s.app.server --host 127.0.0.1 --port 8766 --mode test
```

The server has no authentication. Use `--host 0.0.0.0` for LAN access only on a trusted network. Origin checks are disabled by default; `--strict-origin` enables them and `--allow-origin` adds exceptions. Restart the service after changing code or environment settings.

## Evaluation modes

| Mode | Source of engineering constants | Additional services |
|---|---|---|
| `test` | Existing MongoDB `StructureProperties` records for the current DOI's subimages | Chat, embeddings and retrieval databases |
| `normal` | Image editing → Hunyuan3D reconstruction → initial voxels → octant reflections → erosion/dilation candidates → periodic FEM → engineering constants | Image editing and Hunyuan3D; optional CHFEM |

Normal defaults are constituent Young modulus **1 GPa**, constituent Poisson ratio **0.3**, voxel resolution **64³**, orthorhombic symmetry and the Python solver. The pipeline preserves the initial structure, independently reflects eight octants, and generates six-neighbor periodic erosion/dilation candidates with six erosion and eight dilation steps. Identical nonempty candidates are computed once with generation aliases retained.

Constituent parameters can be specified explicitly in natural-language input and become fixed once computation preparation starts. The UI has one input box; HTTP accepts `{text}` only. Moduli use GPa, with MPa/kPa/Pa conversion; Poisson ratio is dimensionless. Default tolerances are 0.05 for Poisson ratio, 0.05 GPa for Young modulus and 0.025 GPa for shear modulus.

Example conversation:

```text
Honeycomb structures
Poisson ratio -0.5, tolerance 0.05
```

An explicit constituent and directional target example:

```text
constituent Young modulus 2 GPa; constituent Poisson ratio 0.25; E_z 0.4 GPa, tolerance 0.01 GPa
```

Values belong to a specific computed structure or variant associated with a source image, not every structure in the image or the paper's reported values. Service failures, missing images and invalid numbers are errors. Normal mode never substitutes stored properties for a failed computation. Switch modes in a new conversation.

## Verification and offline benchmarks

Run local tests and targeted lint checks:

```bash
PYTHONPATH=src python -m pytest -q tests
python -m ruff check --select F,E7,E9 src/m3s
node --check src/m3s/app/app.js
```

The JavaScript syntax check requires Node.js. With the normal extra installed, three small numerical examples exercise the real LangGraph calculation flow with the Python solver:

```bash
PYTHONPATH=src python -m m3s.scripts.verify_langgraph_examples --solver python --resolution 16 --steps 1 --out output/local_examples
```

This reduced validation starts from local voxels and does not call paper retrieval, image editing or reconstruction services. The production default remains 64³.

The offline retrieval benchmark reads stored properties in test mode:

```bash
PYTHONPATH=src python -m m3s.eval.run_recall --out output/recall_eval_english --rounds 5 --workers 6
```

Use a new output directory for the English query set. The runner rejects completed records whose query ID matches but whose query text differs. Existing cached statistics and paper figures are not regenerated by code changes. `--target-grid paper` selects the 42 manuscript targets; `--queries` supplies a custom list and `--expand` generates additional English queries.

## Project layout

```text
src/m3s/
  app/                     Browser UI and HTTP server
  engine.py                LangGraph workflow and sessions
  retrieval_vocabulary.py  Domain keywords from SI S4.1
  SubAgents/TagsQuery.py    Paper retrieval and ranking
  nodes/                   Relevance, target and evaluation nodes
  geometry.py              Voxelization, reflections and morphology
  homogenization.py        Python periodic hexahedral FEM
  elastic/                 Engineering constants and directional extrema
  db/                      Neo4j, MongoDB and Milvus adapters
  eval/                    Offline queries, metrics and plots
  scripts/                 Solver adapter and representative examples
  START.sh                 Single-process launcher
```

See [the application documentation](src/m3s/app/README.md) for the HTTP API, session behavior, supported directional components and retry semantics.
