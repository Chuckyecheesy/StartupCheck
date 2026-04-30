# Startup proposal — RAG judging layer

Python tooling that evaluates startup pitch PDFs against investor-style criteria using **RAG** (LangChain + OpenAI embeddings + FAISS), with an optional **Streamlit** UI for interactive review, scoring, and Gemini-powered rewrites. Analytics records can be written to **Snowflake** when configured.

## Repository layout

| Path | Role |
|------|------|
| `RAG-layer-of-startup-judging/` | Main application: Streamlit UI, RAG pipelines, criteria/scoring logic |
| `RAG-layer-of-startup-judging/frontend.py` | Streamlit entrypoint |
| `RAG-layer-of-startup-judging/extract_proposal.py` | CLI: analyze a proposal PDF vs criteria PDFs |
| `RAG-layer-of-startup-judging/scoring.py` | CLI: score analysis JSON and suggest refinements |
| `RAG-layer-of-startup-judging/startup_rag.py` | Startup-doc RAG + Snowflake insert helpers |
| `RAG-layer-of-startup-judging/criteria_rag.py` | Criteria PDF RAG helpers |
| `test_snowflake_connection.py` | Local Snowflake connectivity smoke test (adjust paths/credentials for your environment) |
| `requirements.txt` | Python dependencies (install from repository root) |

## Prerequisites

- Python 3.10+ recommended  
- [OpenAI API key](https://platform.openai.com/) for embeddings and baseline chat (`gpt-3.5-turbo` in RAG chains)  
- For the UI rewrite flows: [Gemini API key](https://ai.google.dev/) (`GEMINI_API_KEY`)  
- Optional: [ElevenLabs](https://elevenlabs.io/) API key for text-to-speech in the UI  
- Optional: Snowflake account + key-pair auth for `startup_rag` logging  

## Setup

1. Clone or copy this repo and go to the repository root (the directory that contains `requirements.txt`).

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Copy environment variables into `.env` in `RAG-layer-of-startup-judging/` (never commit real secrets; this repo’s `.gitignore` excludes `.env`). Example variables:

   | Variable | Purpose |
   |----------|---------|
   | `OPENAI_API_KEY` | Required for embeddings/RAG and proposal extraction |
   | `GEMINI_API_KEY` | Required for Gemini features in `frontend.py` |
   | `GEMINI_MODEL_REWRITE`, `GEMINI_MODEL_PROJECTION`, `GEMINI_MODEL_*_FALLBACKS` | Optional model overrides |
   | `ELEVENLAB_API_KEY`, `VOICE_ID` | Optional TTS |
   | `FAST_REWRITE` | Optional; set to `1` / `true` / `yes` for faster rewrite behavior |
   | `SNOWFLAKE_USER`, `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_PRIVATE_KEY_FILE`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_ROLE` | Optional Snowflake logging |

4. **Criteria PDFs** — The UI and `extract_proposal.py` expect criteria documents such as marketing, angel, technical, and VC judging PDFs in `RAG-layer-of-startup-judging/` (names referenced in `frontend.py` / `extract_proposal.py`). Add any missing files so default paths resolve.

5. **Snowflake key** — Point `SNOWFLAKE_PRIVATE_KEY_FILE` at your `.p8` private key file on disk.

When running the app or CLI scripts, use the app directory:

```bash
cd RAG-layer-of-startup-judging
```

## Run the frontend (Streamlit)

The web UI is launched with Streamlit. Activate your virtual environment (if you use one), go to the app folder, then start the app:

```bash
cd RAG-layer-of-startup-judging
streamlit run frontend.py
```

From the repository root in one line:

```bash
cd RAG-layer-of-startup-judging && streamlit run frontend.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) — open it in your browser. Ensure `OPENAI_API_KEY` and `GEMINI_API_KEY` are set (for example via `RAG-layer-of-startup-judging/.env`) for full functionality.

## CLI: analyze and score a proposal

Still from `RAG-layer-of-startup-judging/`:

```bash
# Produce structured analysis JSON
python extract_proposal.py --proposal-pdf /path/to/pitch.pdf --save-json analysis.json

# Score using saved analysis (avoids path quirks when chaining modules)
python scoring.py --analysis-json analysis.json --save-json score_output.json
```

For advanced options (multiple criteria PDFs, custom startup-failure PDF), see:

```bash
python extract_proposal.py --help
python scoring.py --help
```

## Other utilities

- **`upload.py`** — Upload a PDF to an HTTP endpoint (`python upload.py file.pdf https://example.com/upload`).
- **`commentaire.py`** — Gemini-backed commentary helpers (requires `GEMINI_API_KEY` and `google-generativeai`).
- **`test_elevenlabs.py`** — Quick ElevenLabs API check.

## Security notes

- Do not commit `.env`, private keys (`.p8`), or API keys. Use `.env.example` (without secrets) if you want to document variable names for collaborators.
