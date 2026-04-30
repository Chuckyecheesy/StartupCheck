"""
Analyze any uploaded startup proposal PDF and evaluate:
1) Which required criteria are filled/partial/missing.
2) Which common startup failure mistakes are present.

Usage:
    python extract_proposal.py --proposal-pdf /path/to/uploaded.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from criteria_rag import query_rag, setup_rag as setup_criteria_rag
from startup_rag import setup_rag as setup_startup_rag

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CRITERIA_PDFS = [
    str(BASE_DIR / "MarketingCriteria.pdf"),
    str(BASE_DIR / "Angel_Investor_Evaluation.pdf"),
    str(BASE_DIR / "technical_evaluation_criteria.pdf"),
    str(BASE_DIR / "VCJudgingCriteria.pdf"),
]


def _extract_json(text: str) -> Any:
    """
    Parse JSON from model output, including fenced code blocks.
    """
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def _validate_pdf(path: str) -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Proposal PDF not found: {path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path}")


def _validate_pdf_list(paths: List[str]) -> List[str]:
    if not paths:
        raise ValueError("At least one criteria PDF is required.")
    for path in paths:
        _validate_pdf(path)
    return paths


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        normalized = item.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return output


def get_criteria(criteria_chain) -> List[str]:
    """
    Get criteria list from criteria PDF knowledge base.
    """
    prompt = """
Extract the startup judging criteria from this document.
Return ONLY JSON in this exact structure:
{
  "criteria": ["criterion 1", "criterion 2"]
}
Rules:
- Keep criterion names short.
- Remove duplicates.
- Keep 5-20 criteria.
"""
    output = query_rag(criteria_chain, prompt)
    parsed = _extract_json(output)

    criteria = parsed.get("criteria", [])
    if not isinstance(criteria, list):
        raise ValueError("Invalid criteria format returned by criteria_rag.")

    cleaned = [str(item).strip() for item in criteria if str(item).strip()]
    if not cleaned:
        raise ValueError("No criteria were extracted from criteria PDF.")

    return cleaned


def evaluate_criteria(proposal_chain, criteria: List[str]) -> List[Dict[str, str]]:
    """
    Evaluate proposal completeness against required criteria.
    """
    prompt = f"""
Evaluate the startup proposal against these criteria:
{json.dumps(criteria, indent=2)}

For each criterion return:
- criterion
- status (filled | partial | missing)
- evidence (short quote or paraphrase from proposal)
- highlighted_excerpt (exact phrase/sentence from proposal in quotes when possible)
- reason (short explanation)
- improvement (one concrete suggestion)

Return ONLY JSON:
{{
  "criteria_evaluation": [
    {{
      "criterion": "...",
      "status": "filled",
      "evidence": "...",
      "highlighted_excerpt": "\"...\"",
      "reason": "...",
      "improvement": "..."
    }}
  ]
}}
"""
    output = query_rag(proposal_chain, prompt)
    parsed = _extract_json(output)
    result = parsed.get("criteria_evaluation", [])
    if not isinstance(result, list):
        raise ValueError("Invalid criteria evaluation format from proposal analysis.")
    return result


def evaluate_failure_mistakes(startup_chain, proposal_chain) -> List[Dict[str, str]]:
    """
    Detect if proposal includes common failure mistakes from startup_rag.
    """
    failure_patterns_prompt = """
List the most common startup failure mistakes from this document.
Return ONLY JSON:
{
  "mistakes": [
    "mistake 1",
    "mistake 2"
  ]
}
Rules:
- Keep each mistake as a short phrase.
- Return 5 to 15 mistakes.
"""
    mistakes_raw = query_rag(startup_chain, failure_patterns_prompt)
    mistakes_parsed = _extract_json(mistakes_raw)
    mistakes = mistakes_parsed.get("mistakes", [])

    if not isinstance(mistakes, list) or not mistakes:
        raise ValueError("Could not extract common failure mistakes from startup_rag.")

    proposal_check_prompt = f"""
Given this list of common startup failure mistakes:
{json.dumps(mistakes, indent=2)}

Check whether each mistake appears in the startup proposal.

For each mistake return:
- mistake
- present (yes | no | unclear)
- evidence (short quote/paraphrase if present)
- highlighted_excerpt (exact phrase/sentence from proposal in quotes when possible)
- risk (why this matters, <= 25 words)
- fix (one concrete recommendation)

Return ONLY JSON:
{{
  "failure_check": [
    {{
      "mistake": "...",
      "present": "yes",
      "evidence": "...",
      "highlighted_excerpt": "\"...\"",
      "risk": "...",
      "fix": "..."
    }}
  ]
}}
"""
    output = query_rag(proposal_chain, proposal_check_prompt)
    parsed = _extract_json(output)
    result = parsed.get("failure_check", [])
    if not isinstance(result, list):
        raise ValueError("Invalid failure-check format from proposal analysis.")
    return result


def analyze_uploaded_pdf(
    proposal_pdf: str,
    criteria_pdf: str = "",
    startup_pdf: str = "",
    criteria_pdfs: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Main reusable function for uploaded user PDFs.
    """
    _validate_pdf(proposal_pdf)

    if criteria_pdfs is None:
        criteria_pdfs = [criteria_pdf] if criteria_pdf else DEFAULT_CRITERIA_PDFS
    criteria_pdfs = _validate_pdf_list(criteria_pdfs)

    proposal_chain = setup_criteria_rag(proposal_pdf)

    per_rag_results: List[Dict[str, Any]] = []
    merged_criteria: List[str] = []
    merged_evaluation: List[Dict[str, str]] = []
    for source_pdf in criteria_pdfs:
        criteria_chain = setup_criteria_rag(source_pdf)
        source_criteria = get_criteria(criteria_chain)
        source_eval = evaluate_criteria(proposal_chain, source_criteria)
        per_rag_results.append(
            {
                "criteria_source_pdf": source_pdf,
                "criteria_used": source_criteria,
                "criteria_evaluation": source_eval,
            }
        )
        merged_criteria.extend(source_criteria)
        merged_evaluation.extend(source_eval)

    merged_criteria = _dedupe_keep_order(merged_criteria)
    failure_check: List[Dict[str, str]] = []
    if startup_pdf:
        _validate_pdf(startup_pdf)
        startup_chain = setup_startup_rag(startup_pdf)
        failure_check = evaluate_failure_mistakes(startup_chain, proposal_chain)

    return {
        "proposal_pdf": proposal_pdf,
        "criteria_sources": criteria_pdfs,
        "criteria_by_rag": per_rag_results,
        "criteria_used": merged_criteria,
        "criteria_evaluation": merged_evaluation,
        "failure_mistakes_check": failure_check,
    }


def print_report(analysis: Dict[str, Any]) -> None:
    criteria_eval = analysis.get("criteria_evaluation", [])
    failures = analysis.get("failure_mistakes_check", [])

    filled = [x for x in criteria_eval if str(x.get("status", "")).lower() == "filled"]
    partial = [x for x in criteria_eval if str(x.get("status", "")).lower() == "partial"]
    missing = [x for x in criteria_eval if str(x.get("status", "")).lower() == "missing"]

    present_failures = [x for x in failures if str(x.get("present", "")).lower() == "yes"]

    print("\n=== Startup Proposal Analysis ===")
    print(f"Proposal: {analysis.get('proposal_pdf', '')}")
    print(f"Criteria filled: {len(filled)}")
    print(f"Criteria partial: {len(partial)}")
    print(f"Criteria missing: {len(missing)}")
    print(f"Common failure mistakes detected: {len(present_failures)}")

    print("\n--- Missing Criteria ---")
    if not missing:
        print("None")
    else:
        for item in missing:
            print(f"- {item.get('criterion', '')}: {item.get('improvement', '')}")
            print(f"  Evidence: {item.get('evidence', '')}")
            excerpt = item.get("highlighted_excerpt", "") or "(No direct excerpt found)"
            print(f"  Highlighted excerpt: {excerpt}")

    print("\n--- Partial Criteria ---")
    if not partial:
        print("None")
    else:
        for item in partial:
            print(f"- {item.get('criterion', '')}: {item.get('improvement', '')}")
            print(f"  Evidence: {item.get('evidence', '')}")
            excerpt = item.get("highlighted_excerpt", "") or "(No direct excerpt found)"
            print(f"  Highlighted excerpt: {excerpt}")

    print("\n--- Failure Mistakes Found ---")
    if not present_failures:
        print("None")
    else:
        for item in present_failures:
            print(f"- {item.get('mistake', '')}")
            print(f"  Evidence: {item.get('evidence', '')}")
            excerpt = item.get("highlighted_excerpt", "") or "(No direct excerpt found)"
            print(f"  Highlighted excerpt: {excerpt}")
            print(f"  Risk: {item.get('risk', '')}")
            print(f"  Fix: {item.get('fix', '')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze uploaded startup proposal PDF for missing/filled criteria and "
            "common startup failure mistakes."
        )
    )
    parser.add_argument(
        "--proposal-pdf",
        required=True,
        help="Path to user-uploaded proposal PDF.",
    )
    parser.add_argument(
        "--criteria-pdf",
        default="",
        help="Legacy single criteria PDF source (optional).",
    )
    parser.add_argument(
        "--criteria-pdfs",
        nargs="+",
        default=None,
        help="One or more criteria PDFs for per-document RAG analysis.",
    )
    parser.add_argument(
        "--startup-pdf",
        default="",
        help="Optional startup failure reasons PDF source.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to save full analysis JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set in .env or system env."
        )

    analysis = analyze_uploaded_pdf(
        proposal_pdf=args.proposal_pdf,
        criteria_pdf=args.criteria_pdf,
        criteria_pdfs=args.criteria_pdfs,
        startup_pdf=args.startup_pdf,
    )
    print_report(analysis)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as out:
            json.dump(analysis, out, indent=2)
        print(f"\nSaved JSON report to: {args.save_json}")


if __name__ == "__main__":
    main()
