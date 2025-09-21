# Repository Guidelines

## Project Structure & Module Organization
- `app/backend/`: FastAPI app (`main.py`), minimal auth, API endpoints.
- `app/frontend/`: Static demo UI (`index.html`). Run backend separately.
- `model/`: RAG and tuning code (`embed_and_index.py`, `vertex_rag_pipeline.py`, `tune_gemini.py`).
- `tools/`: Data prep utilities (e.g., `data_load.py`, `token_counter.py`).
- `lambdas/` + `artifacts/`: Deployment helpers and packaged functions used by infra.
- `templates/`: Infra templates (e.g., Cloud/CFN YAML templates).
- `data/`: Sample PDFs and prepared JSONL for demos.
- Root scripts: `setup.sh`, `run.sh`, `deploy.sh`, `create-rag.sh`; config in `config.py`, `params.json`.

## Build, Test, and Development Commands
- Install deps (root + app): `pip install -r requirements.txt && pip install -r app/requirements.txt`.
- Quickstart (prep data, build embeddings, basic query): `./setup.sh && ./run.sh`.
- Local API (from `app/`): `uvicorn backend.main:app --reload --port 8080`.
- GCP deploy (Container/Run): see `README.md` for `gcloud builds submit` and `gcloud run deploy` examples.
- Env config: export variables in `README.md` (e.g., `PROJECT_ID`, `GCS_PREFIX`, `MODEL_ID`). Avoid committing secrets.

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indent. Modules/files `snake_case.py`; functions `snake_case`; classes `CamelCase`.
- Keep functions small; isolate GCP calls behind helpers (e.g., in `model/` or `tools/`).
- Env vars UPPER_SNAKE (`MODEL_BASE`, `ALLOWED_USER_PW`). Prefer dependency injection over globals in new code.

## Testing Guidelines
- No formal suite yet. Add tests with `pytest` under `tests/` as `test_*.py`.
- Mock external services (Vertex AI, GCS, FAISS I/O). Run with: `pytest -q`.
- Target: cover core logic in `model/` and `tools/` (parsers, chunking, indexing) before modifying endpoints.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject, explanatory body (what/why), reference issues (`#123`).
- PRs: clear description, steps to validate, affected commands/env vars, screenshots for UI/API changes, link related issues.
- Keep diffs focused; update `README.md` when changing setup or deploy flow. Ensure `./run.sh` still works.

## Security & Configuration Tips
- Do not commit secrets or service account keys. Use local env vars or a `.env` (not tracked).
- Validate IAM roles before deploy; avoid logging PII; sanitize user inputs at API boundaries.
