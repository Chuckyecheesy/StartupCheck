"""
Score startup proposals out of 10 using extract_proposal.py analysis.

Decision rule:
- score >= 7.0 -> likely to reach investor
- score < 7.0  -> refine with prioritized actions until above threshold

Usage examples:
    python scoring.py --proposal-pdf /path/to/uploaded.pdf
    python scoring.py --analysis-json /path/to/proposal_analysis.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


EXTRACT_PROPOSAL_PATH = (
    Path(__file__).resolve().parent
    / "RAG-layer-of-startup-judging"
    / "extract_proposal.py"
)


def _load_extract_module():
    if not EXTRACT_PROPOSAL_PATH.exists():
        raise FileNotFoundError(f"extract_proposal.py not found at {EXTRACT_PROPOSAL_PATH}")

    spec = importlib.util.spec_from_file_location("extract_proposal_module", EXTRACT_PROPOSAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load extract_proposal module specification.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_analysis_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Analysis JSON must be a dictionary object.")
    return data


def _normalize_status(status: str) -> str:
    value = str(status).strip().lower()
    if value in {"filled", "partial", "missing"}:
        return value
    return "missing"


def _normalize_present(value: str) -> str:
    val = str(value).strip().lower()
    if val in {"yes", "no", "unclear"}:
        return val
    return "unclear"


def _score_criteria_and_failures(
    criteria_eval: List[Dict[str, Any]],
    failure_check: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(criteria_eval, list):
        criteria_eval = []
    if not isinstance(failure_check, list):
        failure_check = []

    total_criteria = len(criteria_eval) or 1
    filled_count = 0
    partial_count = 0
    missing_count = 0

    for item in criteria_eval:
        status = _normalize_status(item.get("status", "missing"))
        if status == "filled":
            filled_count += 1
        elif status == "partial":
            partial_count += 1
        else:
            missing_count += 1

    completeness_ratio = (filled_count + 0.5 * partial_count) / total_criteria
    completeness_score = 10.0 * completeness_ratio

    total_failures = len(failure_check) or 1
    yes_count = 0
    unclear_count = 0
    for item in failure_check:
        present = _normalize_present(item.get("present", "unclear"))
        if present == "yes":
            yes_count += 1
        elif present == "unclear":
            unclear_count += 1

    penalty_ratio = (yes_count + 0.5 * unclear_count) / total_failures
    failure_penalty = min(3.0, 3.0 * penalty_ratio)

    raw_score = completeness_score - failure_penalty
    final_score = round(max(0.0, min(10.0, raw_score)), 2)

    return {
        "score_out_of_10": final_score,
        "breakdown": {
            "criteria": {
                "filled": filled_count,
                "partial": partial_count,
                "missing": missing_count,
                "total": len(criteria_eval),
                "completeness_score": round(completeness_score, 2),
            },
            "failure_mistakes": {
                "present_yes": yes_count,
                "present_unclear": unclear_count,
                "total": len(failure_check),
                "penalty": round(failure_penalty, 2),
            },
        },
    }


def compute_score(analysis: Dict[str, Any]) -> Dict[str, Any]:
    criteria_eval = analysis.get("criteria_evaluation", [])
    failure_check = analysis.get("failure_mistakes_check", [])
    criteria_by_rag = analysis.get("criteria_by_rag", [])

    overall = _score_criteria_and_failures(criteria_eval, failure_check)
    final_score = overall["score_out_of_10"]

    likely_to_reach_investor = final_score >= 7.0
    scores_by_investor: List[Dict[str, Any]] = []
    if isinstance(criteria_by_rag, list):
        for source in criteria_by_rag:
            source_pdf = str(source.get("criteria_source_pdf", "unknown_source"))
            source_eval = source.get("criteria_evaluation", [])
            source_score = _score_criteria_and_failures(source_eval, failure_check)
            source_final_score = source_score["score_out_of_10"]
            scores_by_investor.append(
                {
                    "investor_source_pdf": source_pdf,
                    "score_out_of_10": source_final_score,
                    "likely_to_reach_investor": source_final_score >= 7.0,
                    "breakdown": source_score["breakdown"],
                }
            )

    return {
        "score_out_of_10": final_score,
        "threshold": 7.0,
        "likely_to_reach_investor": likely_to_reach_investor,
        "breakdown": overall["breakdown"],
        "scores_by_investor": scores_by_investor,
    }


def build_refinement_plan(analysis: Dict[str, Any], score_result: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Build prioritized actions to move proposals toward investor readiness.
    """
    if score_result.get("likely_to_reach_investor", False):
        return []

    criteria_eval = analysis.get("criteria_evaluation", [])
    failure_check = analysis.get("failure_mistakes_check", [])

    actions: List[Tuple[int, Dict[str, str]]] = []

    for item in criteria_eval:
        status = _normalize_status(item.get("status", "missing"))
        if status not in {"missing", "partial"}:
            continue
        priority = 1 if status == "missing" else 2
        criterion = str(item.get("criterion", "Unnamed criterion")).strip()
        improvement = str(item.get("improvement", "Add clear supporting details.")).strip()
        actions.append(
            (
                priority,
                {
                    "type": "criteria_gap",
                    "priority": "high" if priority == 1 else "medium",
                    "issue": f"{criterion} is {status}.",
                    "action": improvement,
                    "expected_impact": "Increase completeness and investor confidence.",
                },
            )
        )

    for item in failure_check:
        present = _normalize_present(item.get("present", "unclear"))
        if present != "yes":
            continue
        mistake = str(item.get("mistake", "Unknown failure pattern")).strip()
        fix = str(item.get("fix", "Mitigate this risk with a concrete plan and metric.")).strip()
        actions.append(
            (
                1,
                {
                    "type": "failure_risk",
                    "priority": "high",
                    "issue": f"Failure risk detected: {mistake}.",
                    "action": fix,
                    "expected_impact": "Reduce investor concern about execution risk.",
                },
            )
        )

    actions.sort(key=lambda x: x[0])
    return [entry for _, entry in actions[:8]]


def print_scoring_report(
    analysis: Dict[str, Any],
    score_result: Dict[str, Any],
    refinement_plan: List[Dict[str, str]],
) -> None:
    score = score_result["score_out_of_10"]
    threshold = score_result["threshold"]
    verdict = (
        "LIKELY to reach investor"
        if score_result["likely_to_reach_investor"]
        else "NOT YET likely to reach investor"
    )

    print("\n=== Investor Readiness Score ===")
    print(f"Proposal: {analysis.get('proposal_pdf', '')}")
    print(f"Score: {score}/10")
    print(f"Threshold: {threshold}")
    print(f"Verdict: {verdict}")

    breakdown = score_result["breakdown"]
    c = breakdown["criteria"]
    f = breakdown["failure_mistakes"]
    print("\n--- Breakdown ---")
    print(
        "Criteria -> "
        f"filled={c['filled']}, partial={c['partial']}, missing={c['missing']}, "
        f"completeness_score={c['completeness_score']}"
    )
    print(
        "Failure mistakes -> "
        f"yes={f['present_yes']}, unclear={f['present_unclear']}, penalty={f['penalty']}"
    )

    if refinement_plan:
        print("\n--- Refine Until Investor-Ready ---")
        for idx, item in enumerate(refinement_plan, start=1):
            print(f"{idx}. [{item.get('priority', 'medium').upper()}] {item.get('issue', '')}")
            print(f"   Action: {item.get('action', '')}")
            print(f"   Impact: {item.get('expected_impact', '')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score startup proposal out of 10 using extract_proposal analysis."
    )
    parser.add_argument(
        "--analysis-json",
        default="",
        help="Path to saved JSON output from extract_proposal.py",
    )
    parser.add_argument(
        "--proposal-pdf",
        default="",
        help="Path to proposal PDF (runs extract_proposal.py logic directly).",
    )
    parser.add_argument(
        "--criteria-pdf",
        default="RAG-layer-of-startup-judging/criteria_evaluate.pdf",
        help="Path to criteria PDF source.",
    )
    parser.add_argument(
        "--startup-pdf",
        default="RAG-layer-of-startup-judging/startup_failure_reason.pdf",
        help="Path to startup failure reasons PDF source.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to save score + refinement output JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.analysis_json:
        analysis = _load_analysis_json(args.analysis_json)
    elif args.proposal_pdf:
        module = _load_extract_module()
        analysis = module.analyze_uploaded_pdf(
            proposal_pdf=args.proposal_pdf,
            criteria_pdf=args.criteria_pdf,
            startup_pdf=args.startup_pdf,
        )
    else:
        raise ValueError("Provide either --analysis-json or --proposal-pdf.")

    score_result = compute_score(analysis)
    refinement_plan = build_refinement_plan(analysis, score_result)
    print_scoring_report(analysis, score_result, refinement_plan)

    if args.save_json:
        output = {
            "analysis": analysis,
            "score": score_result,
            "refinement_plan": refinement_plan,
        }
        with open(args.save_json, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=2)
        print(f"\nSaved scoring output to: {args.save_json}")


if __name__ == "__main__":
    main()
