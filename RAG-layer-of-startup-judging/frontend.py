from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from extract_proposal import analyze_uploaded_pdf
from scoring import build_refinement_plan, compute_score

try:
    import google.generativeai as genai
except ImportError:
    genai = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CRITERIA_PDFS = [
    str(BASE_DIR / "MarketingCriteria.pdf"),
    str(BASE_DIR / "Angel_Investor_Evaluation.pdf"),
    str(BASE_DIR / "technical_evaluation_criteria.pdf"),
    str(BASE_DIR / "VCJudgingCriteria.pdf"),
]
DEFAULT_STARTUP_PDF = ""

HIGHLIGHT_COLORS = {
    "criteria": (1.0, 0.75, 0.0),  # amber
    "failure": (1.0, 0.75, 0.0),   # amber
}

GEMINI_MODEL_DEFAULT = "gemini-2.5-flash-lite"
# Default projection model uses a *different* model id so free-tier quotas (per model / per minute)
# apply to separate buckets. One "Rewrite" click still triggers many generate_content calls; 429 can
# happen from RPM or daily caps even when the AI Studio chart looks low (lag, other keys, or burst).
GEMINI_MODEL_PROJECTION_DEFAULT = "gemini-2.0-flash"


def _dedupe_model_list(names: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for name in names:
        n = (name or "").strip()
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _rewrite_model_candidates() -> List[str]:
    """Models for query plan, evidence pick, rewrite, repair, rewrite-delta (heavy rewrite path)."""
    primary = (os.environ.get("GEMINI_MODEL_REWRITE", "") or "").strip() or GEMINI_MODEL_DEFAULT
    extra = (os.environ.get("GEMINI_MODEL_REWRITE_FALLBACKS", "") or "").strip()
    if not extra:
        extra = (os.environ.get("GEMINI_MODEL_FALLBACKS", "") or "").strip()
    if extra:
        tail = [m.strip() for m in extra.split(",") if m.strip()]
        return _dedupe_model_list([primary, *tail])
    return _dedupe_model_list([primary, "gemini-2.0-flash", "gemini-2.5-flash"])


def _projection_model_candidates() -> List[str]:
    """Models for score projection + uplift copy (lighter follow-on; separate default model)."""
    primary = (os.environ.get("GEMINI_MODEL_PROJECTION", "") or "").strip() or GEMINI_MODEL_PROJECTION_DEFAULT
    extra = (os.environ.get("GEMINI_MODEL_PROJECTION_FALLBACKS", "") or "").strip()
    if extra:
        tail = [m.strip() for m in extra.split(",") if m.strip()]
        return _dedupe_model_list([primary, *tail])
    return _dedupe_model_list([primary, GEMINI_MODEL_DEFAULT, "gemini-2.5-flash"])

# Use AI Studio / Gemini API (API key), not Vertex AI (GCP project quotas).
_GEMINI_API_ENDPOINT = "generativelanguage.googleapis.com"


def _gemini_api_key() -> str:
    raw = str(os.environ.get("GEMINI_API_KEY", "") or "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1].strip()
    if not raw:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    return raw


def _configure_gemini() -> None:
    if genai is None:
        raise ImportError("Install google-generativeai to use Gemini.")
    from google.api_core.client_options import ClientOptions

    genai.configure(
        api_key=_gemini_api_key(),
        client_options=ClientOptions(api_endpoint=_GEMINI_API_ENDPOINT),
    )


PERSONA_GUIDANCE = {
    "vc": (
        "Prioritize venture-scale upside: market size, growth velocity, moat, "
        "fund-return potential, and scalability."
    ),
    "angel": (
        "Prioritize founder credibility, execution realism, early traction proof, "
        "and near-term de-risking milestones."
    ),
    "technical": (
        "Prioritize technical feasibility, architecture credibility, defensibility "
        "of technology, implementation risk, and roadmap realism."
    ),
    "market_analyst": (
        "Prioritize market structure, segmentation, GTM channel logic, "
        "competition/substitutes, timing, and unit economics coherence."
    ),
}

PERSONA_LABELS = {
    "vc": "VC",
    "angel": "Angel",
    "technical": "Technical reviewer",
    "market_analyst": "Market analyst",
}

INVESTOR_PERSONA_HINTS = {
    "vc": "vc",
    "angel": "angel",
    "technical": "technical",
    "market": "market_analyst",
    "marketing": "market_analyst",
}


def _clean_excerpt(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) > 2:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _analysis_cache_key(file_bytes: bytes, file_name: str) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()
    return f"{file_name}:{digest}"


@st.cache_data(show_spinner=False)
def run_analysis_cached(
    cache_key: str,
    file_bytes: bytes,
    file_name: str,
    criteria_pdf: str,
    startup_pdf: str,
    criteria_pdfs: List[str] | None = None,
) -> Dict[str, Any]:
    # cache_key is included so cache invalidates by content/hash.
    _ = cache_key
    suffix = Path(file_name).suffix if Path(file_name).suffix else ".pdf"
    try:
        # Prefer project directory for temp artifacts; some systems have constrained /tmp.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            dir=str(BASE_DIR),
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_pdf_path = temp_file.name
    except OSError as exc:
        raise RuntimeError(
            "Unable to create temporary upload file. Disk may be full. "
            "Use the local-path mode or free disk space and retry."
        ) from exc

    try:
        return analyze_uploaded_pdf(
            proposal_pdf=temp_pdf_path,
            criteria_pdf=criteria_pdf,
            startup_pdf=startup_pdf,
            criteria_pdfs=criteria_pdfs,
        )
    finally:
        try:
            os.remove(temp_pdf_path)
        except OSError:
            pass


@st.cache_data(show_spinner=False)
def run_analysis_from_path_cached(
    cache_key: str,
    proposal_pdf_path: str,
    criteria_pdf: str,
    startup_pdf: str,
    criteria_pdfs: List[str] | None = None,
) -> Dict[str, Any]:
    _ = cache_key
    return analyze_uploaded_pdf(
        proposal_pdf=proposal_pdf_path,
        criteria_pdf=criteria_pdf,
        startup_pdf=startup_pdf,
        criteria_pdfs=criteria_pdfs,
    )


def _collect_highlight_targets(analysis: Dict[str, Any]) -> List[Dict[str, str]]:
    targets: List[Dict[str, str]] = []
    for item in analysis.get("criteria_evaluation", []):
        status = str(item.get("status", "")).lower()
        if status in {"missing", "partial"}:
            excerpt = _clean_excerpt(item.get("highlighted_excerpt") or item.get("evidence"))
            if excerpt:
                reason = str(item.get("reason", "")).strip()
                improvement = str(item.get("improvement", "")).strip()
                note_parts = []
                if reason:
                    note_parts.append(f"Why weak: {reason}")
                if improvement:
                    note_parts.append(f"Missing detail: {improvement}")
                targets.append(
                    {
                        "type": "criteria",
                        "title": str(item.get("criterion", "Criteria issue")),
                        "excerpt": excerpt,
                        "note": "\n".join(note_parts).strip(),
                    }
                )

    for item in analysis.get("failure_mistakes_check", []):
        present = str(item.get("present", "")).lower()
        if present == "yes":
            excerpt = _clean_excerpt(item.get("highlighted_excerpt") or item.get("evidence"))
            if excerpt:
                risk = str(item.get("risk", "")).strip()
                fix = str(item.get("fix", "")).strip()
                note_parts = []
                if risk:
                    note_parts.append(f"Why risky: {risk}")
                if fix:
                    note_parts.append(f"Missing detail: {fix}")
                targets.append(
                    {
                        "type": "failure",
                        "title": str(item.get("mistake", "Common error")),
                        "excerpt": excerpt,
                        "note": "\n".join(note_parts).strip(),
                    }
                )
    return targets


def annotate_pdf(
    original_pdf_bytes: bytes,
    analysis: Dict[str, Any],
) -> Tuple[bytes, List[Dict[str, str]], List[Dict[str, Any]]]:
    doc = fitz.open(stream=original_pdf_bytes, filetype="pdf")
    unmatched: List[Dict[str, str]] = []
    note_markers: List[Dict[str, Any]] = []
    targets = _collect_highlight_targets(analysis)

    for target in targets:
        found = False
        note_added = False
        needle = target["excerpt"]
        if len(needle) > 280:
            needle = needle[:280]
        for page in doc:
            rects = page.search_for(needle, quads=False)
            for rect in rects:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=HIGHLIGHT_COLORS[target["type"]])
                annot.update()
                found = True

                # Add only one comment icon per highlighted issue block.
                # Place it near the first matched rect with a small offset for better clickability.
                if not note_added:
                    note_text = str(target.get("note", "")).strip()
                    if note_text:
                        page_rect = page.rect
                        icon_x = min(max(rect.x0 + 6, page_rect.x0 + 8), page_rect.x1 - 20)
                        icon_y = min(max(rect.y0 + 6, page_rect.y0 + 8), page_rect.y1 - 20)
                        icon_point = fitz.Point(icon_x, icon_y)
                        text_annot = page.add_text_annot(icon_point, note_text, icon="Comment")
                        text_annot.set_info(
                            title=target.get("title", "Review note"),
                            content=note_text,
                        )
                        try:
                            text_annot.set_open(True)
                        except Exception:
                            pass
                        text_annot.update()
                        width = max(float(page_rect.width), 1.0)
                        height = max(float(page_rect.height), 1.0)
                        note_markers.append(
                            {
                                "page_index": int(page.number),
                                "x_pct": ((icon_x - float(page_rect.x0)) / width) * 100.0,
                                "y_pct": ((icon_y - float(page_rect.y0)) / height) * 100.0,
                                "title": str(target.get("title", "Review note")),
                                "note": note_text,
                            }
                        )
                    note_added = True
        if not found:
            unmatched.append(target)

    return doc.tobytes(), unmatched, note_markers


def _render_pdf(
    pdf_bytes: bytes,
    height: int = 700,
    note_markers: List[Dict[str, Any]] | None = None,
) -> None:
    # Use a fully in-app scrollable viewer to avoid browser PDF plugin issues.
    with st.expander("Embedded PDF (scrollable)", expanded=True):
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            max_pages = min(len(doc), 20)
            if max_pages == 0:
                st.caption("No pages found in PDF.")
            pages_html: List[str] = []
            for i in range(max_pages):
                page = doc[i]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False)
                img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                page_notes = [
                    n for n in (note_markers or [])
                    if int(n.get("page_index", -1)) == i
                ]
                buttons_html = []
                for idx, n in enumerate(page_notes):
                    x_pct = float(n.get("x_pct", 5.0))
                    y_pct = float(n.get("y_pct", 5.0))
                    title = str(n.get("title", "Review note"))
                    note = str(n.get("note", ""))
                    title_payload = json.dumps(title)
                    note_payload = json.dumps(note)
                    buttons_html.append(
                        f"""
                        <button
                          onclick='openNote({title_payload}, {note_payload})'
                          title="{title.replace('"', '&quot;')}"
                          style="
                            position:absolute;
                            left:{x_pct:.2f}%;
                            top:{y_pct:.2f}%;
                            transform:translate(-50%,-50%);
                            border:1px solid #555;
                            border-radius:14px;
                            background:#ffe082;
                            width:28px;
                            height:28px;
                            cursor:pointer;
                            font-size:16px;
                            line-height:24px;
                          "
                        >💬</button>
                        """
                    )
                page_html = f"""
                <div style="position:relative; width:100%; max-width:920px; margin:0 auto 16px auto;">
                  <img src="data:image/png;base64,{img_b64}" style="width:100%; height:auto; display:block;" />
                  {''.join(buttons_html)}
                  <div style="font-size:12px; color:#666; margin-top:6px;">Page {i + 1}</div>
                </div>
                """
                pages_html.append(page_html)
            scroll_panel_html = f"""
            <style>
              .note-modal-backdrop {{
                display:none;
                position:fixed;
                inset:0;
                background:rgba(0,0,0,0.35);
                z-index:9999;
                align-items:center;
                justify-content:center;
              }}
              .note-modal {{
                width:min(560px,92vw);
                max-height:78vh;
                overflow-y:auto;
                background:#fff;
                border:1px solid #ddd;
                border-radius:10px;
                box-shadow:0 10px 30px rgba(0,0,0,0.25);
                padding:14px 16px;
                font-family:system-ui,-apple-system,sans-serif;
              }}
              .note-modal h4 {{
                margin:0 0 8px 0;
                font-size:16px;
              }}
              .note-modal p {{
                margin:0;
                white-space:pre-wrap;
                line-height:1.4;
                color:#222;
              }}
              .note-close {{
                margin-top:14px;
                padding:6px 10px;
                border:1px solid #bbb;
                border-radius:6px;
                background:#f5f5f5;
                cursor:pointer;
              }}
            </style>
            <div
              style="
                height:{height}px;
                overflow-y:auto;
                overflow-x:hidden;
                border:1px solid #dcdcdc;
                border-radius:8px;
                padding:10px;
                background:#fafafa;
              "
            >
              {''.join(pages_html)}
            </div>
            <div id="note-modal-backdrop" class="note-modal-backdrop" onclick="closeNote(event)">
              <div class="note-modal">
                <h4 id="note-modal-title">Annotation</h4>
                <p id="note-modal-body"></p>
                <button class="note-close" onclick="closeNote()">Close</button>
              </div>
            </div>
            <script>
              function openNote(title, note) {{
                const backdrop = document.getElementById("note-modal-backdrop");
                const titleEl = document.getElementById("note-modal-title");
                const bodyEl = document.getElementById("note-modal-body");
                if (!backdrop || !titleEl || !bodyEl) return;
                titleEl.textContent = title || "Annotation";
                bodyEl.textContent = note || "";
                backdrop.style.display = "flex";
              }}

              function closeNote(event) {{
                if (event && event.target && event.target.id !== "note-modal-backdrop") {{
                  return;
                }}
                const backdrop = document.getElementById("note-modal-backdrop");
                if (backdrop) backdrop.style.display = "none";
              }}
            </script>
            """
            components.html(scroll_panel_html, height=height + 18, scrolling=False)
            if len(doc) > max_pages:
                st.caption(f"Showing first {max_pages} pages in interactive view.")
        except Exception as exc:  # pragma: no cover - runtime UI safeguard
            st.caption(f"Interactive highlighted view unavailable: {exc}")

    # Text fallback for browsers that fail embedded PDF rendering.
    with st.expander("Extracted text (fallback)", expanded=False):
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            max_pages = min(len(doc), 20)
            text_blocks: List[str] = []
            for i in range(max_pages):
                page_text = doc[i].get_text("text").strip()
                if page_text:
                    text_blocks.append(f"--- Page {i + 1} ---\n{page_text}")
            if text_blocks:
                st.text_area(
                    "Uploaded PDF text",
                    value="\n\n".join(text_blocks),
                    height=420,
                    key="pdf_text_fallback",
                )
            else:
                st.caption("No extractable text found. The PDF may be image-only.")
            if len(doc) > max_pages:
                st.caption(f"Showing extracted text for first {max_pages} pages.")
        except Exception as exc:  # pragma: no cover - runtime UI safeguard
            st.caption(f"Text extraction fallback unavailable: {exc}")


def _verdict_text(score: float) -> str:
    if score >= 7.0:
        return "Ready for investor review."
    return "Needs extensive editing before investor review."


def _build_flagged_items(analysis: Dict[str, Any]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for item in analysis.get("criteria_evaluation", []):
        status = str(item.get("status", "")).lower()
        if status in {"missing", "partial"}:
            items.append(
                {
                    "kind": "criteria",
                    "name": str(item.get("criterion", "Unnamed criterion")),
                    "status": status,
                    "reason": str(item.get("reason", "")),
                    "evidence": str(item.get("highlighted_excerpt") or item.get("evidence") or ""),
                    "fix": str(item.get("improvement", "")),
                }
            )

    for item in analysis.get("failure_mistakes_check", []):
        present = str(item.get("present", "")).lower()
        if present == "yes":
            items.append(
                {
                    "kind": "common_error",
                    "name": str(item.get("mistake", "Unknown error")),
                    "status": "detected",
                    "reason": str(item.get("risk", "")),
                    "evidence": str(item.get("highlighted_excerpt") or item.get("evidence") or ""),
                    "fix": str(item.get("fix", "")),
                }
            )
    return items


def _extract_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object.")
    return parsed


def _infer_persona_from_investor_name(investor_name: str) -> str:
    lowered = str(investor_name or "").strip().lower()
    for key, persona in INVESTOR_PERSONA_HINTS.items():
        if key in lowered:
            return persona
    return "vc"


def _get_lowest_investor_score(score_result: Dict[str, Any]) -> Dict[str, Any] | None:
    scores = score_result.get("scores_by_investor", [])
    if not isinstance(scores, list) or not scores:
        return None
    valid = [entry for entry in scores if isinstance(entry, dict)]
    if not valid:
        return None
    return min(valid, key=lambda x: float(x.get("score_out_of_10", 0.0)))


def _sleep_after_gemini_429(exc: BaseException, attempt: int, initial_delay: int) -> None:
    """Honor server-suggested retry delay when present (RPM throttling)."""
    msg = str(exc)
    m = re.search(r"retry in ([\d.]+)\s*s", msg, re.IGNORECASE)
    if m:
        time.sleep(min(float(m.group(1)) + 0.5, 120.0))
    else:
        time.sleep(initial_delay * (attempt + 1))


def _call_gemini_with_retry(model_name: str, prompt: str, max_retries: int = 3, initial_delay: int = 5) -> str:
    _configure_gemini()

    errors: List[str] = []
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            answer = getattr(response, "text", "")
            if answer:
                return answer
        except Exception as exc:
            if "429" in str(exc):
                if attempt < max_retries - 1:
                    _sleep_after_gemini_429(exc, attempt, initial_delay)
                    continue
            errors.append(str(exc))
            break

    error_msg = f"Model {model_name} failed after {max_retries} attempts: {'; '.join(errors)}"
    raise RuntimeError(error_msg)


def generate_query_plan_with_gemini(
    persona: str,
    persona_guidance: str,
    criterion_name: str,
    reason: str,
    improvement: str,
    excerpt: str,
) -> List[str]:
    if genai is None:
        raise ImportError("Install google-generativeai to use query planning.")

    prompt = f"""
You are a retrieval query planner for startup proposal editing.

Inputs:
- Persona: {persona}
- Persona lens: {persona_guidance}
- Weak criterion: {criterion_name}
- Why weak: {reason}
- Required improvement: {improvement}
- Current excerpt: {excerpt}

Task:
Create 5 focused retrieval queries to find supporting evidence in the proposal.
Queries must target concrete facts (numbers, customers, metrics, mechanisms, timelines, channels, competitors, evidence statements), not generic wording.

Return JSON only:
{{
  "queries": [
    "query 1",
    "query 2",
    "query 3",
    "query 4",
    "query 5"
  ]
}}
""".strip()
    errors: List[str] = []
    for model_name in _rewrite_model_candidates():
        try:
            answer = _call_gemini_with_retry(model_name, prompt)
            parsed = _extract_json(answer)
            queries = parsed.get("queries", [])
            if isinstance(queries, list) and queries:
                cleaned = [str(q).strip() for q in queries if str(q).strip()]
                if cleaned:
                    return cleaned[:5]
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
    raise RuntimeError(f"Query planning failed: {'; '.join(errors[-3:])}")


def select_evidence_chunks_with_gemini(
    persona: str,
    criterion_name: str,
    reason: str,
    improvement: str,
    evidence_chunks_with_ids: str,
) -> Dict[str, Any]:
    if genai is None:
        raise ImportError("Install google-generativeai to use evidence selection.")

    prompt = f"""
You are an evidence selector for grounded startup proposal rewriting.

Goal:
Select the most relevant evidence chunks for rewriting one weak criterion.

Inputs:
- Persona: {persona}
- Weak criterion: {criterion_name}
- Why weak: {reason}
- Required improvement: {improvement}
- Candidate chunks: {evidence_chunks_with_ids}

Rules:
- Select 8 to 12 chunk_ids only.
- Prioritize chunks with concrete facts (metrics, process, customer segment, timeline, channel).
- Keep diversity: include at least 3 different evidence themes.
- Do not rewrite text.
- Ignore generic narrative chunks lacking concrete support.
- Prefer chunks with specific entities, numbers, or execution details.

Return JSON only:
{{
  "selected_chunk_ids": ["c3", "c8", "c12"],
  "selection_rationale": ["...", "...", "..."]
}}
""".strip()
    errors: List[str] = []
    for model_name in _rewrite_model_candidates():
        try:
            answer = _call_gemini_with_retry(model_name, prompt)
            parsed = _extract_json(answer)
            ids = parsed.get("selected_chunk_ids", [])
            if isinstance(ids, list):
                parsed["selected_chunk_ids"] = [str(x).strip() for x in ids if str(x).strip()]
                return parsed
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
    raise RuntimeError(f"Evidence selection failed: {'; '.join(errors[-3:])}")


def _normalize_for_contains(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return cleaned


def _get_or_build_proposal_vectorstore(
    proposal_pdf_bytes: bytes,
    cache_key: str,
):
    """
    Builds (or reuses) a FAISS vectorstore for the uploaded proposal.
    Stored in st.session_state to avoid repeated embedding work.
    """
    if "proposal_vectorstores" not in st.session_state:
        st.session_state["proposal_vectorstores"] = {}

    store_cache = st.session_state["proposal_vectorstores"]
    if cache_key in store_cache:
        return store_cache[cache_key]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
        dir=str(BASE_DIR),
    ) as temp_file:
        temp_file.write(proposal_pdf_bytes)
        temp_pdf_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
    finally:
        try:
            os.remove(temp_pdf_path)
        except OSError:
            pass

    store_cache[cache_key] = vectorstore
    return vectorstore


def _retrieve_evidence_chunks_with_ids(
    proposal_pdf_bytes: bytes,
    cache_key: str,
    queries: List[str],
    k_per_query: int = 5,
    max_chunks: int = 24,
) -> List[Dict[str, Any]]:
    """
    Retrieves top evidence chunks from the proposal for grounded rewrite generation.
    """
    vectorstore = _get_or_build_proposal_vectorstore(proposal_pdf_bytes, cache_key)

    results: List[Dict[str, Any]] = []
    seen_text_hashes: set[str] = set()

    for q in queries:
        try:
            docs = vectorstore.similarity_search(q, k=k_per_query)
        except Exception:
            docs = []
        for doc in docs:
            chunk_text = str(getattr(doc, "page_content", "")) or ""
            # Deduplicate by normalized chunk text.
            norm = _normalize_for_contains(chunk_text)
            text_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            if text_hash in seen_text_hashes:
                continue
            if not norm:
                continue
            seen_text_hashes.add(text_hash)
            page_num = ""
            try:
                page_num = doc.metadata.get("page", "")
            except Exception:
                page_num = ""
            results.append(
                {
                    "chunk_id": f"c{len(results)}",
                    "page": page_num,
                    "text": chunk_text,
                }
            )
            if len(results) >= max_chunks:
                return results

    return results[:max_chunks]


def _select_chunks_by_ids(
    evidence_chunks: List[Dict[str, Any]],
    selected_ids: List[str],
) -> List[Dict[str, Any]]:
    if not selected_ids:
        return evidence_chunks[:12]
    by_id = {str(ch.get("chunk_id", "")): ch for ch in evidence_chunks}
    selected = [by_id[cid] for cid in selected_ids if cid in by_id]
    return selected[:12] if selected else evidence_chunks[:12]


def _format_evidence_chunks_with_ids_for_prompt(
    evidence_chunks: List[Dict[str, Any]],
    text_max_chars: int = 900,
) -> str:
    blocks: List[str] = []
    for chunk in evidence_chunks:
        chunk_id = str(chunk.get("chunk_id", "unknown"))
        page = chunk.get("page", "")
        raw_text = str(chunk.get("text", ""))
        if len(raw_text) > text_max_chars:
            raw_text = raw_text[:text_max_chars] + "..."
        blocks.append(
            f"[{chunk_id}] (page {page})\n{raw_text}".strip()
        )
    return "\n---\n".join(blocks)


def _split_sentences(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = re.split(r"(?<=[.!?])\s+", raw)
    return [p.strip() for p in parts if p.strip()]


def _normalize_rewrite_schema(rewrite_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes rewrite output into a superset schema so downstream UI keeps working.
    Supports both legacy keys (rewrite_v1/diagnosis) and strict keys
    (recommended_rewrite/why_this_is_weak/sentence_to_claim_map).
    """
    if not isinstance(rewrite_json, dict):
        return rewrite_json

    normalized = dict(rewrite_json)
    if not normalized.get("why_this_is_weak"):
        normalized["why_this_is_weak"] = str(normalized.get("diagnosis", "")).strip()
    if not normalized.get("diagnosis"):
        normalized["diagnosis"] = str(normalized.get("why_this_is_weak", "")).strip()

    if not normalized.get("recommended_rewrite"):
        normalized["recommended_rewrite"] = str(normalized.get("rewrite_v1", "")).strip()
    if not normalized.get("rewrite_v1"):
        normalized["rewrite_v1"] = str(normalized.get("recommended_rewrite", "")).strip()

    if not normalized.get("alternative_rewrite"):
        normalized["alternative_rewrite"] = str(normalized.get("rewrite_v2", "")).strip()
    if not normalized.get("rewrite_v2"):
        normalized["rewrite_v2"] = str(normalized.get("alternative_rewrite", "")).strip()

    if not normalized.get("suggested_next_actions"):
        normalized["suggested_next_actions"] = normalized.get("founder_todo_next_7_days", [])
    if not normalized.get("founder_todo_next_7_days"):
        normalized["founder_todo_next_7_days"] = normalized.get("suggested_next_actions", [])

    claims = normalized.get("claims", [])
    if not isinstance(claims, list):
        claims = []
    normalized_claims: List[Dict[str, Any]] = []
    for idx, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id", "")).strip() or f"cl{idx}"
        claim_text = str(claim.get("claim_text", "")).strip()
        supported_by = claim.get("supported_by", [])
        if not isinstance(supported_by, list):
            supported_by = []
        normalized_claims.append(
            {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "supported_by": [str(x) for x in supported_by if str(x).strip()],
            }
        )
    normalized["claims"] = normalized_claims

    sentence_cards = normalized.get("sentence_cards", [])
    if isinstance(sentence_cards, list) and sentence_cards:
        converted_map: List[Dict[str, Any]] = []
        for entry in sentence_cards:
            if not isinstance(entry, dict):
                continue
            converted_map.append(
                {
                    "sentence_index": int(entry.get("sentence_index", 0) or 0),
                    "sentence_text": str(
                        entry.get("rewritten_sentence") or entry.get("sentence_text") or ""
                    ).strip(),
                    "claim_ids": (
                        [str(c).strip() for c in entry.get("claim_ids", []) if str(c).strip()]
                        if isinstance(entry.get("claim_ids", []), list)
                        else []
                    ),
                    "original_sentence": str(entry.get("original_sentence", "")).strip(),
                    "rewritten_sentence": str(
                        entry.get("rewritten_sentence") or entry.get("sentence_text") or ""
                    ).strip(),
                    "source_chunk_ids": (
                        [str(c).strip() for c in entry.get("source_chunk_ids", []) if str(c).strip()]
                        if isinstance(entry.get("source_chunk_ids", []), list)
                        else []
                    ),
                    "source_pages": (
                        [int(p) for p in entry.get("source_pages", []) if str(p).strip()]
                        if isinstance(entry.get("source_pages", []), list)
                        else []
                    ),
                }
            )
        if converted_map:
            normalized["sentence_to_claim_map"] = converted_map

    sentence_map = normalized.get("sentence_to_claim_map", [])
    if not isinstance(sentence_map, list) or not sentence_map:
        rewrite_text = str(normalized.get("recommended_rewrite", "")).strip()
        sentences = _split_sentences(rewrite_text)
        original_sentences = _split_sentences(str(normalized.get("original_excerpt", "")).strip())
        fallback_map: List[Dict[str, Any]] = []
        default_claim_ids = [c.get("claim_id", "") for c in normalized_claims if c.get("claim_id")]
        for i, sentence in enumerate(sentences, start=1):
            fallback_map.append(
                {
                    "sentence_index": i,
                    "sentence_text": sentence,
                    "claim_ids": default_claim_ids[:1] if default_claim_ids else [],
                    "original_sentence": original_sentences[i - 1] if i - 1 < len(original_sentences) else "",
                    "rewritten_sentence": sentence,
                    "source_chunk_ids": [],
                    "source_pages": [],
                }
            )
        normalized["sentence_to_claim_map"] = fallback_map
    normalized["sentence_cards"] = [
        {
            "sentence_index": int(entry.get("sentence_index", 0) or 0),
            "original_sentence": str(entry.get("original_sentence", "")).strip(),
            "rewritten_sentence": str(entry.get("rewritten_sentence") or entry.get("sentence_text") or "").strip(),
            "claim_ids": (
                [str(c).strip() for c in entry.get("claim_ids", []) if str(c).strip()]
                if isinstance(entry.get("claim_ids", []), list)
                else []
            ),
            "source_chunk_ids": (
                [str(c).strip() for c in entry.get("source_chunk_ids", []) if str(c).strip()]
                if isinstance(entry.get("source_chunk_ids", []), list)
                else []
            ),
            "source_pages": (
                [int(p) for p in entry.get("source_pages", []) if str(p).strip()]
                if isinstance(entry.get("source_pages", []), list)
                else []
            ),
        }
        for entry in normalized.get("sentence_to_claim_map", [])
        if isinstance(entry, dict)
    ]
    return normalized


def _quick_rewrite_json_checks(
    rewrite_json: Dict[str, Any],
    evidence_chunks: List[Dict[str, Any]],
) -> List[str]:
    checks: List[str] = []
    normalized = _normalize_rewrite_schema(rewrite_json)
    evidence_map: Dict[str, str] = {
        str(ch.get("chunk_id")): str(ch.get("text", ""))
        for ch in evidence_chunks
        if ch.get("chunk_id") is not None
    }
    claim_ids = {
        str(c.get("claim_id", "")).strip()
        for c in normalized.get("claims", [])
        if isinstance(c, dict) and str(c.get("claim_id", "")).strip()
    }

    # Check 1: claim_text exists verbatim in at least one cited chunk.
    for claim in normalized.get("claims", []):
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id", "")).strip()
        claim_text = str(claim.get("claim_text", "")).strip()
        supported_by = claim.get("supported_by", [])
        if not isinstance(supported_by, list):
            supported_by = []
        quoted = claim_text.strip('"').strip()
        has_verbatim = False
        for cid in supported_by:
            chunk_text = evidence_map.get(str(cid), "")
            if quoted and quoted in chunk_text:
                has_verbatim = True
                break
        if claim_id and not has_verbatim:
            checks.append(
                f"Claim {claim_id} is not verbatim in cited supported_by chunks."
            )

    # Check 2/3: sentence map claim_ids are valid and not only unsupported placeholders.
    for entry in normalized.get("sentence_to_claim_map", []):
        if not isinstance(entry, dict):
            continue
        sentence_index = int(entry.get("sentence_index", 0) or 0)
        mapped_ids = entry.get("claim_ids", [])
        if not isinstance(mapped_ids, list):
            mapped_ids = []
        mapped_ids = [str(cid).strip() for cid in mapped_ids if str(cid).strip()]
        if not mapped_ids:
            checks.append(f"Sentence {sentence_index} has no mapped claim_ids.")
            continue
        invalid = [cid for cid in mapped_ids if cid not in claim_ids]
        if invalid:
            checks.append(
                f"Sentence {sentence_index} references unknown claim_ids: {', '.join(invalid)}."
            )
    return checks


def _verify_rewrite_claims_locally(
    rewrite_json: Dict[str, Any],
    evidence_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Deterministic grounding check:
    - Every claim_text must appear (as normalized substring) in at least one supported_by chunk.
    """
    evidence_map: Dict[str, str] = {
        str(ch.get("chunk_id")): str(ch.get("text", ""))
        for ch in evidence_chunks
        if ch.get("chunk_id") is not None
    }

    normalized_rewrite = _normalize_rewrite_schema(rewrite_json)
    claims = normalized_rewrite.get("claims", [])
    if not isinstance(claims, list) or not claims:
        return {
            "verdict": "fail",
            "claim_checks": [],
            "sentence_checks": [],
            "unsupported_claims_count": 1,
            "fix_instructions": [
                "Rewrite must include claims[] with supported_by chunk_ids.",
            ],
        }

    claim_checks: List[Dict[str, Any]] = []
    claim_status_by_id: Dict[str, str] = {}
    unsupported_claims_count = 0

    for idx, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id", "")).strip() or f"cl{idx}"
        claim_text = str(claim.get("claim_text", "")).strip()
        supported_by = claim.get("supported_by", []) or []
        supported_by = [str(x) for x in supported_by if x is not None]

        status = "unsupported"
        evidence_ids_used: List[str] = []
        note = "No supported evidence id matched claim text."

        if supported_by:
            norm_claim = _normalize_for_contains(claim_text)
            matched = False
            for cid in supported_by:
                chunk_text = evidence_map.get(cid, "")
                if not chunk_text:
                    continue
                norm_chunk = _normalize_for_contains(chunk_text)
                if norm_claim and norm_claim in norm_chunk:
                    matched = True
                    evidence_ids_used.append(cid)
                    break
            if matched:
                status = "supported"
            else:
                # Allow a partial status when there is some lexical overlap but not strict containment.
                partial = False
                claim_tokens = set(norm_claim.split())
                for cid in supported_by:
                    chunk_text = evidence_map.get(cid, "")
                    norm_chunk = _normalize_for_contains(chunk_text)
                    chunk_tokens = set(norm_chunk.split())
                    if claim_tokens and chunk_tokens:
                        overlap = len(claim_tokens & chunk_tokens) / max(len(claim_tokens), 1)
                        if overlap >= 0.45:
                            partial = True
                            evidence_ids_used.append(cid)
                            break
                status = "partial" if partial else "unsupported"
        else:
            status = "unsupported"

        if status == "unsupported":
            unsupported_claims_count += 1

        claim_status_by_id[claim_id] = status
        claim_checks.append(
            {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "status": status,
                "valid_supported_by": evidence_ids_used or supported_by,
                "reason": (
                    note
                    if status == "unsupported"
                    else "Claim text partially overlaps evidence." if status == "partial"
                    else "Claim text matched an evidence chunk."
                ),
            }
        )

    sentence_checks: List[Dict[str, Any]] = []
    sentence_map = normalized_rewrite.get("sentence_to_claim_map", [])
    if isinstance(sentence_map, list):
        for entry in sentence_map:
            if not isinstance(entry, dict):
                continue
            sentence_index = int(entry.get("sentence_index", 0) or 0)
            claim_ids = entry.get("claim_ids", [])
            claim_ids = [str(c).strip() for c in claim_ids if str(c).strip()] if isinstance(claim_ids, list) else []
            if not claim_ids:
                sentence_checks.append(
                    {"sentence_index": sentence_index, "status": "unmapped"}
                )
                continue
            has_unsupported = any(
                claim_status_by_id.get(cid, "unsupported") == "unsupported"
                for cid in claim_ids
            )
            sentence_checks.append(
                {
                    "sentence_index": sentence_index,
                    "status": "mapped_to_unsupported_claim" if has_unsupported else "mapped",
                }
            )
            if has_unsupported:
                unsupported_claims_count += 1

    verdict = "pass" if unsupported_claims_count == 0 else "fail"
    quick_checks = _quick_rewrite_json_checks(normalized_rewrite, evidence_chunks)
    return {
        "verdict": verdict,
        "claim_checks": claim_checks,
        "sentence_checks": sentence_checks,
        "sentence_support": [
            {
                "sentence_index": item.get("sentence_index", 0),
                "status": (
                    "unsupported"
                    if item.get("status") == "mapped_to_unsupported_claim"
                    else "partially-supported"
                    if item.get("status") == "unmapped"
                    else "evidence-supported"
                ),
            }
            for item in sentence_checks
        ],
        "unsupported_claims_count": unsupported_claims_count,
        "quick_checks": quick_checks,
        "fix_instructions": [
            "Remove or rewrite any claim that is not present in the evidence chunks.",
            "Ensure each claims[].supported_by references chunk_ids that contain the claim_text.",
            "Ensure every sentence in recommended_rewrite is mapped to at least one supported claim_id.",
            "For each claim_text, copy exact wording from one cited chunk when possible; avoid paraphrase that changes wording.",
            "If a claim is unsupported, delete it rather than rephrase unless exact support exists.",
        ],
    }


def repair_rewrite_with_gemini(
    original_rewrite_json: Dict[str, Any],
    verifier_json: Dict[str, Any],
    persona_name: str,
    weakest_investor_name: str,
    evidence_chunks_for_prompt: str,
) -> Dict[str, Any]:
    """
    Repairs hallucinated claims by removing unsupported claims based on verifier output.
    Returns corrected rewrite JSON.
    """
    if genai is None:
        raise ImportError("Install google-generativeai to use the repair panel.")

    prompt = f"""
You are a rewrite repair assistant.

TASK
Repair the rewrite so all claims/sentences are evidence-supported.

HARD RULES
- Keep same JSON schema.
- Remove or soften unsupported claims.
- Do not add new facts.
- Preserve investor-facing clarity.
- If a claim is unsupported, delete it rather than rephrase unless exact support exists.
- Output valid JSON only. No markdown.

INPUTS
- original_rewrite_json: {json.dumps(_normalize_rewrite_schema(original_rewrite_json), indent=2)}
- audit_result_json: {json.dumps(verifier_json, indent=2)}
- selected_evidence_chunks: {evidence_chunks_for_prompt}

OUTPUT
Return corrected JSON only (same schema as input rewrite_json):
{{
  "persona": "{persona_name}",
  "weakest_investor_name": "{weakest_investor_name}",
  "why_this_is_weak": "...",
  "recommended_rewrite": "...",
  "alternative_rewrite": "...",
  "claims": [
    {{
      "claim_id": "cl1",
      "claim_text": "...",
      "supported_by": ["c0", "c1"]
    }}
  ],
  "sentence_cards": [
    {{
      "sentence_index": 1,
      "original_sentence": "...",
      "rewritten_sentence": "...",
      "claim_ids": ["cl1"],
      "source_chunk_ids": ["c0"],
      "source_pages": [1]
    }}
  ],
  "missing_proof_points": ["...", "..."],
  "suggested_next_actions": ["...", "...", "..."]
}}
""".strip()

    errors: List[str] = []
    for model_name in _rewrite_model_candidates():
        try:
            answer = _call_gemini_with_retry(model_name, prompt)
            return _normalize_rewrite_schema(_extract_json(answer))
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    error_text = "; ".join(errors[-3:]) if errors else "No Gemini repair attempts were made."
    raise RuntimeError(f"Repair failed. Details: {error_text}")


def generate_persona_rewrite_with_gemini(
    finding: Dict[str, str],
    persona_name: str,
    persona_guidance: str,
    overall_score: float,
    weakest_investor_score: float,
    weakest_investor_name: str,
    evidence_chunks_for_prompt: str,
) -> Tuple[Dict[str, Any], str]:
    if genai is None:
        raise ImportError("Install google-generativeai to use the why/rewrite panel.")

    if persona_name not in PERSONA_GUIDANCE:
        raise ValueError(f"Unsupported persona: {persona_name}")

    prompt = f"""
You are an investor-proposal rewriting assistant.

MISSION
Produce a targeted rewrite for the weakest investor lens.
Every factual statement must be grounded in the provided evidence chunks.

HARD RULES (NON-NEGOTIABLE)
1) Use ONLY facts explicitly present in SELECTED_EVIDENCE_CHUNKS.
2) No outside knowledge, no inferred numbers, no invented entities.
3) Every sentence in recommended_rewrite must map to at least one claim in claims[].
4) Each claim must include supported_by chunk IDs.
5) If evidence is missing, do not fabricate; add to missing_proof_points.
6) Keep tone concise, investor-grade, specific.
7) Output JSON only. No markdown.
8) For each claim_text, copy exact wording from one cited chunk when possible; avoid paraphrase that changes wording.
9) Ensure sentence_to_claim_map.claim_ids only reference existing claim_id values.
10) Do not map any sentence only to unsupported claims.

INPUTS
- persona: {persona_name}
- weakest_investor_name: {weakest_investor_name}
- weak_criterion: {finding.get("name", "")}
- why_weak: {finding.get("reason", "")}
- required_improvement: {finding.get("fix", "")}
- original_excerpt: {finding.get("evidence", "")}
- persona_lens_guidance: {persona_guidance}
- baseline_overall_score: {overall_score}
- baseline_weakest_investor_score: {weakest_investor_score}
- selected_evidence_chunks: {evidence_chunks_for_prompt}

OUTPUT FORMAT (JSON ONLY, NO MARKDOWN)
{{
  "persona": "{persona_name}",
  "weakest_investor_name": "{weakest_investor_name}",
  "why_this_is_weak": "1-2 sentences",
  "recommended_rewrite": "90-160 words",
  "alternative_rewrite": "90-160 words",
  "claims": [
    {{
      "claim_id": "cl1",
      "claim_text": "single atomic factual claim",
      "supported_by": ["c3"]
    }}
  ],
  "sentence_cards": [
    {{
      "sentence_index": 1,
      "original_sentence": "sentence from original_excerpt (or closest source sentence)",
      "rewritten_sentence": "exact sentence from recommended_rewrite",
      "claim_ids": ["cl1", "cl2"],
      "source_chunk_ids": ["c3"],
      "source_pages": [2]
    }}
  ],
  "missing_proof_points": ["...", "..."],
  "suggested_next_actions": ["...", "...", "..."],
  "score_uplift_hypothesis": [
    "specific criterion likely to improve and why",
    "specific criterion likely to improve and why"
  ]
}}
"""
    errors: List[str] = []
    for model_name in _rewrite_model_candidates():
        try:
            answer = _call_gemini_with_retry(model_name, prompt)
            return _normalize_rewrite_schema(_extract_json(answer)), model_name
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    error_text = "; ".join(errors[-3:]) if errors else "No Gemini model attempts were made."
    raise RuntimeError(f"No Gemini model succeeded. Details: {error_text}")


def generate_rewrite_change_summary_with_gemini(
    sentence_cards: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if genai is None:
        raise ImportError("Install google-generativeai to use rewrite-difference analysis.")

    prompt = f"""
You are a rewrite-difference analyst.
TASK
For each sentence card, compute how much wording changed from original_sentence to rewritten_sentence.
Return a judge-friendly % change metric.
RULES
- 0% means almost identical wording.
- 100% means fully rewritten wording.
- Base % on lexical/phrase overlap, not semantic opinion.
- Round to whole numbers.
- Keep output JSON only.
INPUT
- sentence_cards: {json.dumps(sentence_cards, indent=2)}
OUTPUT JSON ONLY
{{
  "rewrite_change_summary": {{
    "overall_percent_change": 0,
    "sentence_changes": [
      {{
        "sentence_index": 1,
        "percent_change": 0,
        "change_label": "low|medium|high"
      }}
    ]
  }}
}}
""".strip()
    errors: List[str] = []
    for model_name in _rewrite_model_candidates():
        try:
            answer = _call_gemini_with_retry(model_name, prompt)
            parsed = _extract_json(answer)
            summary = parsed.get("rewrite_change_summary", {})
            if isinstance(summary, dict):
                return summary
            raise ValueError("Missing rewrite_change_summary object.")
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
    raise RuntimeError(f"Rewrite-difference analysis failed: {'; '.join(errors[-3:])}")


def _filter_criteria_subset(
    criteria_eval: List[Dict[str, Any]],
    selected_item: Dict[str, str],
    limit: int = 8,
) -> List[Dict[str, Any]]:
    selected_name = str(selected_item.get("name", "")).strip().lower()
    if not selected_name:
        return criteria_eval[:limit]
    matched: List[Dict[str, Any]] = []
    for item in criteria_eval:
        criterion_name = str(item.get("criterion", "")).strip().lower()
        if selected_name in criterion_name or criterion_name in selected_name:
            matched.append(item)
    return (matched or criteria_eval)[:limit]


def _criteria_subsets_for_projection(
    analysis: Dict[str, Any],
    selected_item: Dict[str, str],
    weakest_investor_source: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    criteria_eval = analysis.get("criteria_evaluation", [])
    if not isinstance(criteria_eval, list):
        criteria_eval = []
    overall_subset = _filter_criteria_subset(criteria_eval, selected_item)

    investor_subset: List[Dict[str, Any]] = []
    by_rag = analysis.get("criteria_by_rag", [])
    if isinstance(by_rag, list):
        for entry in by_rag:
            if not isinstance(entry, dict):
                continue
            source = str(entry.get("criteria_source_pdf", ""))
            if source == weakest_investor_source:
                source_eval = entry.get("criteria_evaluation", [])
                if isinstance(source_eval, list):
                    investor_subset = _filter_criteria_subset(source_eval, selected_item)
                break
    return overall_subset, investor_subset


def generate_projection_with_gemini(
    persona: str,
    weakest_investor_name: str,
    baseline_overall: float,
    baseline_investor: float,
    selected_finding: Dict[str, str],
    persona_rewrite: Dict[str, Any],
    criteria_eval_subset: List[Dict[str, Any]],
    weakest_investor_criteria_subset: List[Dict[str, Any]],
    failure_check: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    if genai is None:
        raise ImportError("Install google-generativeai to use score projection.")

    prompt = f"""
You are an investment evaluation analyst estimating conservative score uplift from a rewrite.

Hard rules:
- Be conservative. If uncertain, keep status unchanged.
- Never assume facts outside the excerpt/rewrite.
- Only upgrade criterion status when rewrite adds concrete detail.
- Output valid JSON only. No markdown.

Scoring model:
- filled = 1.0
- partial = 0.5
- missing = 0.0
- overall score = completeness - failure penalty
- failure penalty unchanged unless rewrite explicitly mitigates listed failure risk

Weakest investor target:
- persona: {persona}
- investor_name: {weakest_investor_name}
- baseline_overall: {baseline_overall}
- baseline_investor: {baseline_investor}

Rewrite output:
{json.dumps(persona_rewrite, indent=2)}

Relevant criteria before rewrite (overall):
{json.dumps(criteria_eval_subset, indent=2)}

Relevant criteria before rewrite (weakest investor only):
{json.dumps(weakest_investor_criteria_subset, indent=2)}

Failure mistakes check:
{json.dumps(failure_check, indent=2)}

Task:
Estimate likely status transitions and projected uplift for:
1) overall score
2) weakest investor score

For each criterion change:
- criterion
- from_status (filled|partial|missing)
- to_status (filled|partial|missing)
- confidence (high|medium|low)
- applies_to (overall|investor|both)
- reason (<=25 words, evidence-grounded)

Return JSON only:
{{
  "projection": {{
    "target_investor": "{weakest_investor_name}",
    "criterion_changes": [
      {{
        "criterion": "...",
        "from_status": "partial",
        "to_status": "filled",
        "confidence": "medium",
        "applies_to": "both",
        "reason": "..."
      }}
    ],
    "projected_failure_penalty_change": 0.0,
    "projected_overall_delta": 0.0,
    "projected_target_investor_delta": 0.0,
    "assumptions": ["..."],
    "uncertainty_notes": ["..."]
  }}
}}
""".strip()

    errors: List[str] = []
    for model_name in _projection_model_candidates():
        try:
            answer = _call_gemini_with_retry(model_name, prompt)
            parsed = _extract_json(answer)
            projection = parsed.get("projection")
            if not isinstance(projection, dict):
                raise ValueError("Projection response missing 'projection' object.")
            return projection, model_name
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    error_text = "; ".join(errors[-3:]) if errors else "No Gemini model attempts were made."
    raise RuntimeError(f"No Gemini projection model succeeded. Details: {error_text}")


def generate_increase_summary_with_gemini(
    score_delta_json: Dict[str, Any],
    criterion_changes_json: List[Dict[str, Any]],
    target_investor: str,
) -> Tuple[Dict[str, Any], str]:
    if genai is None:
        raise ImportError("Install google-generativeai to explain score uplift.")

    prompt = f"""
You are explaining projected score uplift to a founder.

Rules:
- Keep it factual and concise.
- Use only the provided projection data.
- No hype language.
- Output valid JSON only.

Inputs:
- baseline_and_projected_scores: {json.dumps(score_delta_json, indent=2)}
- criterion_changes: {json.dumps(criterion_changes_json, indent=2)}
- target_investor: {target_investor}

Return JSON:
{{
  "increase_summary": {{
    "overall_increase_reason": "1-2 sentences",
    "target_investor_increase_reason": "1-2 sentences",
    "top_3_drivers": [
      "driver 1",
      "driver 2",
      "driver 3"
    ],
    "confidence_note": "short caveat about projection uncertainty"
  }}
}}
""".strip()
    errors: List[str] = []
    for model_name in _projection_model_candidates():
        try:
            answer = _call_gemini_with_retry(model_name, prompt)
            parsed = _extract_json(answer)
            summary = parsed.get("increase_summary")
            if not isinstance(summary, dict):
                raise ValueError("Increase summary response missing 'increase_summary' object.")
            return summary, model_name
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
    error_text = "; ".join(errors[-3:]) if errors else "No Gemini model attempts were made."
    raise RuntimeError(f"No Gemini increase-summary model succeeded. Details: {error_text}")


def _apply_status_change(
    criteria_list: List[Dict[str, Any]],
    target_name: str,
    to_status: str,
) -> bool:
    for item in criteria_list:
        criterion_name = str(item.get("criterion", "")).strip().lower()
        if target_name in criterion_name or criterion_name in target_name:
            item["status"] = to_status
            return True
    return False


def _project_score_from_projection(
    analysis: Dict[str, Any],
    score_result: Dict[str, Any],
    projection: Dict[str, Any],
    weakest_investor_source: str,
) -> Dict[str, Any]:
    criteria_eval = analysis.get("criteria_evaluation", [])
    if not isinstance(criteria_eval, list):
        criteria_eval = []
    updated_analysis = copy.deepcopy(analysis)
    updated_criteria = updated_analysis.get("criteria_evaluation", [])
    if not isinstance(updated_criteria, list):
        updated_criteria = []
        updated_analysis["criteria_evaluation"] = updated_criteria

    changes = projection.get("criterion_changes", [])
    if not isinstance(changes, list):
        changes = []

    valid_status = {"filled", "partial", "missing"}
    applied_changes: List[Dict[str, Any]] = []

    updated_by_rag = updated_analysis.get("criteria_by_rag", [])
    weakest_by_rag_eval: List[Dict[str, Any]] = []
    if isinstance(updated_by_rag, list):
        for entry in updated_by_rag:
            if not isinstance(entry, dict):
                continue
            source = str(entry.get("criteria_source_pdf", "")).strip()
            if source == weakest_investor_source:
                source_eval = entry.get("criteria_evaluation", [])
                if isinstance(source_eval, list):
                    weakest_by_rag_eval = source_eval
                break

    for change in changes:
        if not isinstance(change, dict):
            continue
        target_name = str(change.get("criterion", "")).strip().lower()
        to_status = str(change.get("to_status", "")).strip().lower()
        applies_to = str(change.get("applies_to", "both")).strip().lower()
        if not target_name or to_status not in valid_status:
            continue

        from_status = "missing"
        updated_any = False
        if applies_to in {"overall", "both"}:
            for item in updated_criteria:
                criterion_name = str(item.get("criterion", "")).strip().lower()
                if target_name in criterion_name or criterion_name in target_name:
                    from_status = str(item.get("status", "missing")).strip().lower()
                    if from_status not in valid_status:
                        from_status = "missing"
                    item["status"] = to_status
                    updated_any = True
                    break
        if applies_to in {"investor", "both"} and weakest_by_rag_eval:
            updated_any = _apply_status_change(weakest_by_rag_eval, target_name, to_status) or updated_any
        if not updated_any:
            continue

        applied_changes.append(
            {
                "criterion": change.get("criterion", ""),
                "from_status": from_status,
                "to_status": to_status,
                "confidence": change.get("confidence", "low"),
                "applies_to": applies_to,
                "reason": change.get("reason", ""),
            }
        )

    rescored = compute_score(updated_analysis)
    baseline_value = float(score_result.get("score_out_of_10", 0.0))
    projected_value = float(rescored.get("score_out_of_10", baseline_value))
    baseline_investor = 0.0
    projected_investor = baseline_investor
    baseline_scores = score_result.get("scores_by_investor", [])
    if isinstance(baseline_scores, list):
        for entry in baseline_scores:
            if str(entry.get("investor_source_pdf", "")) == weakest_investor_source:
                baseline_investor = float(entry.get("score_out_of_10", 0.0))
                projected_investor = baseline_investor
                break
    projected_scores = rescored.get("scores_by_investor", [])
    if isinstance(projected_scores, list):
        for entry in projected_scores:
            if str(entry.get("investor_source_pdf", "")) == weakest_investor_source:
                projected_investor = float(entry.get("score_out_of_10", baseline_investor))
                break
    penalty_change = float(projection.get("projected_failure_penalty_change", 0.0) or 0.0)
    projected_value = max(0.0, min(10.0, round(projected_value - penalty_change, 2)))

    return {
        "baseline_score": baseline_value,
        "projected_score": projected_value,
        "delta": round(projected_value - baseline_value, 2),
        "baseline_investor_score": round(baseline_investor, 2),
        "projected_investor_score": round(projected_investor, 2),
        "investor_delta": round(projected_investor - baseline_investor, 2),
        "applied_changes": applied_changes,
    }


def _render_saved_rewrite_panel(bundle: Dict[str, Any]) -> None:
    """Render persona rewrite UI from session_state (survives Streamlit reruns)."""
    answer = bundle["answer"]
    selector_result = bundle["selector_result"]
    selected_item = bundle["selected_item"]
    persona = bundle["persona"]
    weakest_investor_name = bundle["weakest_investor_name"]
    weakest_investor_score = bundle["weakest_investor_score"]
    rewrite_change_summary = bundle["rewrite_change_summary"]
    projection = bundle["projection"]
    projection_result = bundle["projection_result"]
    increase_summary = bundle["increase_summary"]
    tts_key_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(selected_item.get("name", "issue")))[:80]

    selection_rationale = selector_result.get("selection_rationale", [])
    if isinstance(selection_rationale, list) and selection_rationale:
        with st.expander("Reviewer's Note: Source Evidence Analysis"):
            for item in selection_rationale:
                st.write(f"- {item}")

    st.markdown(f"**Why this is weak:** {answer.get('diagnosis', '')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original content**")
        st.info(
            answer.get("original_excerpt")
            or selected_item.get("evidence")
            or "(this component is missing from the pdf)"
        )
    with col2:
        st.markdown("**Recommended rewrite**")
        st.success(answer.get("rewrite_v1", ""))

        st.markdown("---")
        st.markdown("### 🎙️ Voice of the Investor")

        if st.button("Generate Audio Critique", key=f"tts_{persona}_{tts_key_safe}"):
            tts_key = os.environ.get("ELEVENLAB_API_KEY")
            if not tts_key:
                st.error("ELEVENLAB_API_KEY not found in environment.")
            else:
                with st.spinner("Generating audio..."):
                    try:
                        import requests

                        voice_id = (os.environ.get("VOICE_ID", "") or "").strip()
                        if not voice_id:
                            st.error(
                                "VOICE_ID is not set. Add a voice from your ElevenLabs account to .env."
                            )
                            st.info(
                                "Free plans cannot use library voices via API. "
                                "Use your own account voice ID."
                            )
                            return
                        # Free tier no longer supports v1 TTS models.
                        # Default to a currently supported low-latency model, with env override.
                        tts_model_id = os.environ.get("ELEVENLAB_MODEL_ID", "eleven_flash_v2_5")
                        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

                        headers = {
                            "Accept": "audio/mpeg",
                            "Content-Type": "application/json",
                            "xi-api-key": tts_key,
                        }

                        overall_pct_change = max(
                            0, int(rewrite_change_summary.get("overall_percent_change", 0) or 0)
                        )
                        total_delta = float(projection_result.get("delta", 0.0) or 0.0)
                        investor_delta = float(
                            projection_result.get("investor_delta", 0.0) or 0.0
                        )
                        criterion_name = str(selected_item.get("name", "selected criterion")).strip()
                        tts_text = (
                            "You have the original and new version, plus percentage change showing how much "
                            "was changed from the original sentence. "
                            f"Total content change is {overall_pct_change} percent. "
                            f"Potential increase on total score is {total_delta:+.2f}. "
                            f"Potential increase on the targeted investor score is {investor_delta:+.2f}, "
                            f"for your weakest investor profile: {weakest_investor_name}. "
                            "From four investor scorecards, this weakest one is used for focus. "
                            f"The comment dropdown points to unmet criteria, currently including: {criterion_name}."
                        )
                        data = {
                            "text": tts_text,
                            "model_id": tts_model_id,
                            "voice_settings": {
                                "stability": 0.5,
                                "similarity_boost": 0.5,
                            },
                        }

                        response = requests.post(url, json=data, headers=headers)

                        if response.status_code == 200:
                            st.audio(response.content, format="audio/mpeg")
                            st.success("Audio generated successfully!")
                        else:
                            st.error(f"TTS failed: {response.status_code} - {response.text}")
                            if response.status_code == 402:
                                st.info(
                                    "Your current VOICE_ID appears to be a paid/library voice. "
                                    "On free tier, set VOICE_ID to a voice in your own account."
                                )
                            else:
                                st.info(
                                    "Tip: verify VOICE_ID, ELEVENLAB_MODEL_ID, and ElevenLabs quota."
                                )
                    except Exception as e:
                        st.error(f"Error generating audio: {e}")

    if answer.get("rewrite_v2"):
        with st.expander("View alternative rewrite"):
            st.write(answer.get("rewrite_v2", ""))

    sentence_cards = answer.get("sentence_cards", [])
    if isinstance(sentence_cards, list) and sentence_cards:
        st.markdown("---")
        st.markdown("### Rewrite Analysis")

        m1, m2 = st.columns(2)
        overall_pct_change = max(
            0, int(rewrite_change_summary.get("overall_percent_change", 0) or 0)
        )
        m1.metric(
            "Total Content Change",
            f"{overall_pct_change}%",
            help="Measures how much the original text was modified to improve the score.",
        )

        with st.expander("Sentence-by-sentence comparison"):
            percent_by_sentence: Dict[int, int] = {}
            for row in rewrite_change_summary.get("sentence_changes", []):
                if isinstance(row, dict):
                    percent_by_sentence[int(row.get("sentence_index", 0) or 0)] = int(
                        row.get("percent_change", 0) or 0
                    )

            for card in sentence_cards:
                if not isinstance(card, dict):
                    continue
                sidx = int(card.get("sentence_index", 0) or 0)
                orig_sent = str(card.get("original_sentence", "")).strip()
                new_sent = str(card.get("rewritten_sentence", "")).strip()
                if not new_sent:
                    continue
                pct = percent_by_sentence.get(sidx, 0)

                st.markdown(f"**Sentence {sidx}** ({pct}% change)")
                c1, c2 = st.columns(2)
                c1.caption("Original")
                c1.write(orig_sent or "(Empty)")
                c2.caption("Rewritten")
                c2.write(new_sent)
                st.markdown("---")

    uplift_hypothesis = answer.get("score_uplift_hypothesis", [])
    if isinstance(uplift_hypothesis, list) and uplift_hypothesis:
        st.markdown("**Score uplift hypothesis**")
        for item in uplift_hypothesis:
            st.write(f"- {item}")

    st.markdown("**Evidence still needed**")
    missing_items = answer.get("missing_proof_points", [])
    if isinstance(missing_items, list) and missing_items:
        for item in missing_items:
            st.write(f"- {item}")
    else:
        st.write("- None provided.")

    st.markdown("**Suggested next actions**")
    todo_items = answer.get("founder_todo_next_7_days", [])
    if isinstance(todo_items, list) and todo_items:
        for item in todo_items:
            st.write(f"- {item}")
    else:
        st.write("- None provided.")

    st.download_button(
        "Download persona rewrite JSON",
        data=json.dumps(answer, indent=2).encode("utf-8"),
        file_name=f"persona_rewrite_{persona}.json",
        mime="application/json",
        key=f"dl_rewrite_json_{persona}_{tts_key_safe}",
    )

    st.divider()
    st.markdown("### Projected Impact")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(
            "Projected Total Score",
            f"{projection_result['projected_score']}/10",
            delta=f"{projection_result['delta']:+.2f}",
        )
    with m_col2:
        st.metric(
            f"Projected {weakest_investor_name} Score",
            f"{projection_result.get('projected_investor_score', weakest_investor_score)}/10",
            delta=f"{projection_result.get('investor_delta', 0.0):+.2f}",
        )
    st.caption("Weakest investor among the 4 investor scorecards.")

    st.caption(
        f"Baseline: Total {projection_result['baseline_score']}/10 | "
        f"{weakest_investor_name} {projection_result.get('baseline_investor_score', weakest_investor_score)}/10"
    )

    st.markdown(f"**How this improves the {weakest_investor_name} score:**")
    st.write(increase_summary.get("target_investor_increase_reason", "Analysis pending..."))

    with st.expander("Methodology & Confidence"):
        top_drivers = increase_summary.get("top_3_drivers", [])
        if isinstance(top_drivers, list) and top_drivers:
            st.markdown("**Top 3 drivers for this uplift**")
            for driver in top_drivers:
                st.write(f"- {driver}")

        confidence_note = str(increase_summary.get("confidence_note", "")).strip()
        if confidence_note:
            st.caption(f"Note: {confidence_note}")

    assumptions = projection.get("assumptions", [])
    uncertainties = projection.get("uncertainty_notes", [])
    if (isinstance(assumptions, list) and assumptions) or (
        isinstance(uncertainties, list) and uncertainties
    ):
        with st.expander("Method notes (optional)", expanded=False):
            if isinstance(assumptions, list) and assumptions:
                st.markdown("**Assumptions**")
                for item in assumptions:
                    st.write(f"- {item}")
            if isinstance(uncertainties, list) and uncertainties:
                st.markdown("**Uncertainty notes**")
                for item in uncertainties:
                    st.write(f"- {item}")


def main() -> None:
    load_dotenv(override=True)
    st.set_page_config(page_title="InvestorLens | Startup Proposal Analyst", layout="wide")
    st.title("🔍 InvestorLens")
    st.caption("Upload your startup proposal for annotation and feedback of area of improvement to reach investor's interest for funding")

    missing_env = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing_env.append("OPENAI_API_KEY")
    if not os.environ.get("GEMINI_API_KEY"):
        missing_env.append("GEMINI_API_KEY")
    if missing_env:
        st.warning(f"Missing environment variables: {', '.join(missing_env)}")

    st.markdown("### Proposal source")
    source_mode = st.radio(
        "Choose input mode",
        options=["Upload PDF", "Use local PDF path"],
        horizontal=True,
    )

    uploaded = None
    local_path = ""
    file_bytes: bytes | None = None
    file_label = "proposal.pdf"

    if source_mode == "Upload PDF":
        uploaded = st.file_uploader("Upload startup proposal (PDF)", type=["pdf"])
        if not uploaded:
            st.info("Upload a PDF to start analysis.")
            return
        file_bytes = uploaded.getvalue()
        file_label = uploaded.name
    else:
        local_path = st.text_input(
            "Local proposal PDF path",
            value=str(BASE_DIR / "investor_proposal_sample1.pdf"),
            help="Use this mode when temporary disk space is limited.",
        ).strip()
        if not local_path:
            st.info("Enter a local PDF path to start analysis.")
            return
        if not Path(local_path).exists():
            st.error(f"File not found: {local_path}")
            return
        try:
            file_bytes = Path(local_path).read_bytes()
            file_label = Path(local_path).name
        except OSError as exc:
            st.error(f"Failed to read local PDF: {exc}")
            return

    with st.spinner("Tailoring feedback and generating investor scores..."):
        try:
            if source_mode == "Upload PDF":
                assert uploaded is not None and file_bytes is not None
                cache_key = _analysis_cache_key(file_bytes, uploaded.name)
                analysis = run_analysis_cached(
                    cache_key=cache_key,
                    file_bytes=file_bytes,
                    file_name=uploaded.name,
                    criteria_pdf="",
                    startup_pdf=DEFAULT_STARTUP_PDF,
                    criteria_pdfs=DEFAULT_CRITERIA_PDFS,
                )
            else:
                cache_key = f"path:{local_path}"
                analysis = run_analysis_from_path_cached(
                    cache_key=cache_key,
                    proposal_pdf_path=local_path,
                    criteria_pdf="",
                    startup_pdf=DEFAULT_STARTUP_PDF,
                    criteria_pdfs=DEFAULT_CRITERIA_PDFS,
                )
            score_result = compute_score(analysis)
            refinement_plan = build_refinement_plan(analysis, score_result)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

    with st.spinner("Building PDF highlights..."):
        assert file_bytes is not None
        annotated_pdf_bytes, unmatched, note_markers = annotate_pdf(file_bytes, analysis)

    left_col, right_col = st.columns([2, 1], gap="large")

    with left_col:
        st.subheader("Annotated Proposal")
        _render_pdf(annotated_pdf_bytes, height=760, note_markers=note_markers)
        highlight_notes = _collect_highlight_targets(analysis)

        if highlight_notes:
            st.markdown("**Highlight notes (click to view details)**")
            for idx, note in enumerate(highlight_notes, start=1):
                title = str(note.get("title", "Review note")).strip() or "Review note"
                with st.expander(f"{idx}. {title}", expanded=False):
                    detail = str(note.get("note", "")).strip()
                    if detail:
                        st.write(detail)
                    else:
                        st.write("No additional note details available.")
                    excerpt = str(note.get("excerpt", "")).strip()
                    if excerpt:
                        st.caption(f"Matched excerpt: {excerpt[:260]}")

        if unmatched:
            with st.expander("Debug: Unmatched highlights", expanded=False):
                for item in unmatched:
                    st.write(f"- [{item['type']}] {item['title']}: `{item['excerpt'][:140]}`")

        st.download_button(
            "Download annotated PDF",
            data=annotated_pdf_bytes,
            file_name=f"annotated_{file_label}",
            mime="application/pdf",
        )
        st.download_button(
            "Download analysis JSON",
            data=json.dumps(analysis, indent=2).encode("utf-8"),
            file_name="proposal_analysis.json",
            mime="application/json",
        )

    with right_col:
        st.subheader("Scoring")
        investor_scores = score_result.get("scores_by_investor", [])

        st.metric("Total Score", f"{score_result['score_out_of_10']}/10")
        st.progress(min(max(score_result["score_out_of_10"] / 10.0, 0.0), 1.0))
        st.info(_verdict_text(score_result["score_out_of_10"]))

        if investor_scores:
            st.markdown("**Investor-specific scores**")
            for entry in investor_scores:
                investor_name = Path(str(entry.get("investor_source_pdf", "Investor"))).stem
                score_value = float(entry.get("score_out_of_10", 0.0))
                verdict = "Ready" if score_value >= 7.0 else "Needs work"
                st.metric(investor_name, f"{score_value}/10", delta=verdict)

        if refinement_plan:
            st.markdown("**Needs refinement**")
            for item in refinement_plan:
                st.write(f"- {item['issue']} -> {item['action']}")
        else:
            st.success("No immediate critical refinements detected.")

        st.divider()
        st.subheader("Suggested Rewrite")
        flagged_items = _build_flagged_items(analysis)
        if not flagged_items:
            st.write("No major issue selected for rewrite guidance.")
        else:
            selected_idx = st.selectbox(
                "Choose a criterion gap for rewrite (focus: weakest investor score)",
                options=list(range(len(flagged_items))),
                format_func=lambda i: str(flagged_items[i].get("name", "")).strip() or "Issue to improve",
            )
            selected_item = flagged_items[int(selected_idx)]
            st.caption(f"Evidence: {selected_item['evidence'] or '(this component is missing from the pdf)'}")
            lowest_investor = _get_lowest_investor_score(score_result)
            if lowest_investor is None:
                st.warning("No investor-specific score found. Falling back to VC persona.")
                weakest_investor_name = "Investor"
                weakest_investor_source = ""
                weakest_investor_score = float(score_result.get("score_out_of_10", 0.0))
                persona = "vc"
            else:
                weakest_investor_source = str(lowest_investor.get("investor_source_pdf", ""))
                weakest_investor_name = Path(weakest_investor_source).stem or "Investor"
                weakest_investor_score = float(lowest_investor.get("score_out_of_10", 0.0))
                persona = _infer_persona_from_investor_name(weakest_investor_name)
            persona_guidance = PERSONA_GUIDANCE.get(persona, PERSONA_GUIDANCE["vc"])
            st.caption(
                "Weakest investor focus: "
                f"{weakest_investor_name} ({weakest_investor_score}/10) -> {PERSONA_LABELS.get(persona, persona)}"
            )
            assert file_bytes is not None
            rewrite_run_key = (
                f"{hashlib.sha256(file_bytes).hexdigest()}:"
                f"{int(selected_idx)}:{persona}:{weakest_investor_name}"
            )

            fast_rewrite = st.checkbox(
                "Faster rewrite (skip sentence % breakdown + projected score / uplift; fewer Gemini calls)",
                value=os.getenv("FAST_REWRITE", "").strip().lower() in ("1", "true", "yes"),
                help="Core grounded rewrite still runs. Uncheck for full InvestorLens score projection.",
                key="investorlens_fast_rewrite",
            )

            if st.button("Rewrite"):
                with st.spinner("Generating response..."):
                    try:
                        evidence_cache_key = hashlib.sha256(file_bytes).hexdigest()
                        evidence_queries = generate_query_plan_with_gemini(
                            persona=persona,
                            persona_guidance=persona_guidance,
                            criterion_name=str(selected_item.get("name", "")),
                            reason=str(selected_item.get("reason", "")),
                            improvement=str(selected_item.get("fix", "")),
                            excerpt=str(selected_item.get("evidence", "")),
                        )
                        with st.spinner("Retrieving proposal evidence for grounding..."):
                            evidence_candidates = _retrieve_evidence_chunks_with_ids(
                                proposal_pdf_bytes=file_bytes,
                                cache_key=evidence_cache_key,
                                queries=evidence_queries,
                                k_per_query=3 if fast_rewrite else 5,
                                max_chunks=14 if fast_rewrite else 24,
                            )
                            candidates_for_prompt = _format_evidence_chunks_with_ids_for_prompt(
                                evidence_candidates
                            )
                            selector_result = select_evidence_chunks_with_gemini(
                                persona=persona,
                                criterion_name=str(selected_item.get("name", "")),
                                reason=str(selected_item.get("reason", "")),
                                improvement=str(selected_item.get("fix", "")),
                                evidence_chunks_with_ids=candidates_for_prompt,
                            )
                            selected_ids = selector_result.get("selected_chunk_ids", [])
                            evidence_chunks = _select_chunks_by_ids(
                                evidence_candidates,
                                selected_ids if isinstance(selected_ids, list) else [],
                            )
                            evidence_chunks_for_prompt = _format_evidence_chunks_with_ids_for_prompt(
                                evidence_chunks
                            )
                        answer, _ = generate_persona_rewrite_with_gemini(
                            selected_item,
                            persona,
                            persona_guidance,
                            float(score_result.get("score_out_of_10", 0.0)),
                            weakest_investor_score,
                            weakest_investor_name,
                            evidence_chunks_for_prompt,
                        )
                        answer["original_excerpt"] = str(selected_item.get("evidence", ""))
                        answer = _normalize_rewrite_schema(answer)
                        local_verifier = _verify_rewrite_claims_locally(answer, evidence_chunks)
                        grounding_passed = str(local_verifier.get("verdict", "")) == "pass"
                        repair_used = False
                        if not grounding_passed:
                            try:
                                answer_repaired = repair_rewrite_with_gemini(
                                    original_rewrite_json=answer,
                                    verifier_json=local_verifier,
                                    persona_name=persona,
                                    weakest_investor_name=weakest_investor_name,
                                    evidence_chunks_for_prompt=evidence_chunks_for_prompt,
                                )
                                local_verifier = _verify_rewrite_claims_locally(
                                    answer_repaired,
                                    evidence_chunks,
                                )
                                repair_used = True
                                grounding_passed = str(local_verifier.get("verdict", "")) == "pass"
                                answer = answer_repaired
                            except Exception:
                                grounding_passed = False
                                repair_used = False
                                # Hide rewrites if we can't guarantee grounding.
                                try:
                                    answer["rewrite_v1"] = "(Grounded rewrite not available; grounding verification failed.)"
                                    answer["rewrite_v2"] = ""
                                except Exception:
                                    pass
                                local_verifier = {
                                    "verdict": "fail",
                                    "unsupported_claims_count": local_verifier.get("unsupported_claims_count", 1),
                                    "claim_checks": [],
                                }
                        sentence_cards = answer.get("sentence_cards", [])
                        _baseline_projection = {
                            "baseline_score": float(score_result.get("score_out_of_10", 0.0)),
                            "projected_score": float(score_result.get("score_out_of_10", 0.0)),
                            "delta": 0.0,
                            "baseline_investor_score": round(weakest_investor_score, 2),
                            "projected_investor_score": round(weakest_investor_score, 2),
                            "investor_delta": 0.0,
                            "applied_changes": [],
                        }
                        _empty_projection = {"assumptions": [], "uncertainty_notes": []}

                        if fast_rewrite:
                            rewrite_change_summary = {
                                "overall_percent_change": -1,
                                "sentence_changes": [],
                            }
                            projection = dict(_empty_projection)
                            projection_result = dict(_baseline_projection)
                            increase_summary = {
                                "overall_increase_reason": "Skipped for speed (fast rewrite).",
                                "target_investor_increase_reason": "Skipped for speed (fast rewrite).",
                                "top_3_drivers": [],
                                "confidence_note": "",
                            }
                        elif not grounding_passed:
                            rewrite_change_summary = {
                                "overall_percent_change": -1,
                                "sentence_changes": [],
                            }
                            projection = dict(_empty_projection)
                            projection_result = dict(_baseline_projection)
                            increase_summary = {
                                "overall_increase_reason": (
                                    "Projected score is kept unchanged because the rewrite does not materially "
                                    "change the meaning of the original sentence."
                                ),
                                "target_investor_increase_reason": (
                                    "Projected investor score is kept unchanged because the rewrite does not "
                                    "materially change the meaning of the original sentence."
                                ),
                                "top_3_drivers": [],
                                "confidence_note": "",
                            }
                        else:

                            def _run_rewrite_change_summary() -> Dict[str, Any]:
                                if isinstance(sentence_cards, list) and sentence_cards:
                                    try:
                                        return generate_rewrite_change_summary_with_gemini(
                                            sentence_cards
                                        )
                                    except Exception:
                                        return {
                                            "overall_percent_change": -1,
                                            "sentence_changes": [],
                                        }
                                return {
                                    "overall_percent_change": -1,
                                    "sentence_changes": [],
                                }

                            def _run_projection_llm() -> Tuple[Dict[str, Any], str]:
                                criteria_subset, investor_subset = _criteria_subsets_for_projection(
                                    analysis,
                                    selected_item,
                                    weakest_investor_source,
                                )
                                return generate_projection_with_gemini(
                                    persona=persona,
                                    weakest_investor_name=weakest_investor_name,
                                    baseline_overall=float(score_result.get("score_out_of_10", 0.0)),
                                    baseline_investor=weakest_investor_score,
                                    selected_finding=selected_item,
                                    persona_rewrite=answer,
                                    criteria_eval_subset=criteria_subset,
                                    weakest_investor_criteria_subset=investor_subset,
                                    failure_check=analysis.get("failure_mistakes_check", []),
                                )

                            with ThreadPoolExecutor(max_workers=2) as pool:
                                fut_summary = pool.submit(_run_rewrite_change_summary)
                                fut_projection = pool.submit(_run_projection_llm)
                                rewrite_change_summary = fut_summary.result()
                                projection, _ = fut_projection.result()

                            force_no_increase = (
                                isinstance(sentence_cards, list)
                                and bool(sentence_cards)
                                and int(rewrite_change_summary.get("overall_percent_change", -1) or -1)
                                == 0
                            )
                            if force_no_increase:
                                projection = dict(_empty_projection)
                                projection_result = dict(_baseline_projection)
                                increase_summary = {
                                    "overall_increase_reason": (
                                        "Projected score is kept unchanged because % change after rewrite is 0%."
                                    ),
                                    "target_investor_increase_reason": (
                                        "Projected investor score is kept unchanged because % change after "
                                        "rewrite is 0%."
                                    ),
                                    "top_3_drivers": [],
                                    "confidence_note": "",
                                }
                            else:
                                projection_result = _project_score_from_projection(
                                    analysis=analysis,
                                    score_result=score_result,
                                    projection=projection,
                                    weakest_investor_source=weakest_investor_source,
                                )
                                increase_summary, _ = generate_increase_summary_with_gemini(
                                    score_delta_json=projection_result,
                                    criterion_changes_json=projection_result.get(
                                        "applied_changes", []
                                    ),
                                    target_investor=weakest_investor_name,
                                )
                        st.session_state["rewrite_panel_bundle"] = {
                            "answer": answer,
                            "selector_result": selector_result,
                            "selected_item": dict(selected_item),
                            "persona": persona,
                            "weakest_investor_name": weakest_investor_name,
                            "weakest_investor_score": weakest_investor_score,
                            "rewrite_change_summary": rewrite_change_summary,
                            "projection": projection,
                            "projection_result": projection_result,
                            "increase_summary": increase_summary,
                        }
                        st.session_state["rewrite_panel_run_key"] = rewrite_run_key
                    except Exception as exc:  # pragma: no cover - UI runtime guard
                        st.session_state.pop("rewrite_panel_bundle", None)
                        st.session_state.pop("rewrite_panel_run_key", None)
                        st.error(f"Failed to generate LLM answer: {exc}")

            if (
                st.session_state.get("rewrite_panel_bundle") is not None
                and st.session_state.get("rewrite_panel_run_key") == rewrite_run_key
            ):
                _render_saved_rewrite_panel(st.session_state["rewrite_panel_bundle"])


if __name__ == "__main__":
    main()
