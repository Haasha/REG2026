from __future__ import annotations

import argparse
import json
import math
import random
import re
import csv
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

"""
import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("⏳ Waiting for debugger to attach...")
debugpy.wait_for_client()
print("✅ Debugger attached!")
debugpy.breakpoint()
"""

# ============================================================
# Default config
# ============================================================

DEFAULT_JUDGE_MODEL_PATH = "/mnt/nfs02-R6/go67sab/project_agent/LLM/Qwen3-14B"

DEFAULT_DEVICE = "cuda"


DEFAULT_WORKFLOW_SEMANTIC_BACKEND = "lexical" #llm

DEFAULT_EMBEDDING_MODEL = None


DEFAULT_MERGE_PREDICTIONS = False

DEFAULT_STRICT_MISSING_PREDICTIONS = False


DEFAULT_W1 = 0.3
DEFAULT_W2 = 0.3
DEFAULT_W3 = 0.4

DEFAULT_VOTING = 1
DEFAULT_JUDGE_MAX_NEW_TOKENS = 32768
DEFAULT_SKIP_MISSING_ROI = False


# ============================================================
# Shared helpers
# ============================================================

def safe_divide(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_trailing_punctuation(text: str) -> str:
    return re.sub(r"[\s\.\,\;\:\!\?]+$", "", text).strip()


KNOWN_ALIASES = {
    "pridominant": "predominant",
    "dianoses": "diagnoses",
    "diagnosises": "diagnoses",
    "includes": "include",
}


def canonicalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.lower()
    text = normalize_whitespace(text)
    text = strip_trailing_punctuation(text)

    for src, dst in KNOWN_ALIASES.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)

    return text


TERMINAL_TOKENS = {
    "",
    "end",
    "stop",
    "finish",
    "finished",
    "none",
    "null",
    "no next question",
    "no further question",
}


def canonicalize_next_question(text: Optional[str]) -> str:
    text = canonicalize_text(text)

    if text in TERMINAL_TOKENS:
        return "__END__"

    return text


def parse_judge_label(raw_text: str, allowed: List[str]) -> str:
    if raw_text is None:
        return allowed[-1]

    allowed_pattern = "|".join(re.escape(x) for x in allowed)

    match = re.search(
        rf"<answer>\s*({allowed_pattern})\s*</answer>",
        raw_text,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).upper()
    else: 
        return -1

def collect_json_files(paths: List[Path]) -> List[Path]:
    """
    Accept JSON files or directories.

    If a directory is given, all *.json files inside it will be collected.
    This is non-recursive by default.
    """
    json_files: List[Path] = []

    for p in paths:
        p = Path(p)

        if p.is_file():
            json_files.append(p)

        elif p.is_dir():
            json_files.extend(sorted(p.glob("*.json")))

        else:
            raise FileNotFoundError(f"Path does not exist: {p}")

    json_files = sorted(list(dict.fromkeys(json_files)))

    if not json_files:
        raise ValueError(f"No JSON files found from paths: {paths}")

    return json_files


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, output_path: Optional[Path]) -> None:
    if output_path is None:
        return

    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {output_path}")


def load_case_list_from_file(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "cases" in data and isinstance(data["cases"], list):
        return data["cases"]

    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]

    raise ValueError(
        f"Invalid JSON format in {path}. Expected a list of cases, "
        f"or a dict with key 'cases' / 'data'."
    )


def load_case_list_from_files(files: List[Path]) -> List[Dict[str, Any]]:
    all_cases: List[Dict[str, Any]] = []

    for f in files:
        cases = load_case_list_from_file(f)
        all_cases.extend(cases)

    return all_cases


def mean_dict_values(items: List[Dict[str, Any]], field: str) -> float:
    if not items:
        return 0.0

    return float(sum(float(x.get(field, 0.0)) for x in items) / len(items))


# ============================================================
# Part 1: Workflow reasoning metric
# ============================================================

@dataclass(frozen=True)
class Edge:
    src: str
    dst: str

    def key(self) -> Tuple[str, str]:
        return self.src, self.dst


@dataclass
class EdgeRecord:
    edge: Edge
    raw_question: str
    raw_answer: str
    raw_next_question: str
    step_index: int


@dataclass
class CaseScore:
    case_id: str

    binary_path_validity: float
    ordered_path_validity: float

    edge_precision: float
    edge_recall: float
    edge_f1: float

    mess: float
    ranking_score: float

    gt_edge_count: int
    pred_edge_count: int

    edge_tp: int
    edge_fp: int
    edge_fn: int


def index_cases(cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}

    for case in cases:
        case_id = case.get("id")

        if not case_id:
            raise ValueError("Each case must contain an 'id'.")

        if case_id in indexed:
            raise ValueError(f"Duplicate case id found: {case_id}")

        indexed[case_id] = case

    return indexed


def get_workflow_steps(case_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidate_keys = [
        "chain-of-thought",
        "chain_of_thought",
        "workflow",
        "workflow_steps",
        "reasoning_steps",
        "steps",
    ]

    for key in candidate_keys:
        if key in case_obj:
            steps = case_obj[key]

            if not isinstance(steps, list):
                raise ValueError(
                    f"Case {case_obj.get('id')} has invalid '{key}'. Expected list."
                )

            return steps

    raise ValueError(
        f"Case {case_obj.get('id')} does not contain workflow steps. "
        f"Expected one of: {candidate_keys}"
    )


def build_edge_records(case_obj: Dict[str, Any]) -> List[EdgeRecord]:
    steps = get_workflow_steps(case_obj)

    records: List[EdgeRecord] = []

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(
                f"Case {case_obj.get('id')} has invalid step at index {i}. "
                f"Each step must be a dict."
            )

        question = step.get("question", "") or ""
        answer = step.get("answer", "") or ""
        next_question = step.get("next_question", "") or ""

        src = canonicalize_text(question)
        dst = canonicalize_next_question(next_question)

        edge = Edge(src=src, dst=dst)

        records.append(
            EdgeRecord(
                edge=edge,
                raw_question=str(question),
                raw_answer=str(answer),
                raw_next_question=str(next_question),
                step_index=i,
            )
        )

    return records


def unique_edge_set(records: List[EdgeRecord]) -> set[Tuple[str, str]]:
    return {rec.edge.key() for rec in records}


def ordered_edge_list(records: List[EdgeRecord]) -> List[Tuple[str, str]]:
    return [rec.edge.key() for rec in records]


def edge_answer_map(records: List[EdgeRecord]) -> Dict[Tuple[str, str], List[str]]:
    mapping: Dict[Tuple[str, str], List[str]] = {}

    for rec in records:
        mapping.setdefault(rec.edge.key(), []).append(rec.raw_answer)

    return mapping


def edge_question_map(records: List[EdgeRecord]) -> Dict[Tuple[str, str], str]:
    mapping: Dict[Tuple[str, str], str] = {}

    for rec in records:
        mapping.setdefault(rec.edge.key(), rec.raw_question)

    return mapping


def compute_binary_path_validity(
    gt_edges: set[Tuple[str, str]],
    pred_edges: set[Tuple[str, str]],
) -> float:
    return 1.0 if gt_edges == pred_edges else 0.0


def compute_ordered_path_validity(
    gt_ordered_edges: List[Tuple[str, str]],
    pred_ordered_edges: List[Tuple[str, str]],
) -> float:
    return 1.0 if gt_ordered_edges == pred_ordered_edges else 0.0


def compute_edge_metrics(
    gt_edges: set[Tuple[str, str]],
    pred_edges: set[Tuple[str, str]],
) -> Tuple[float, float, float, int, int, int]:
    tp = len(gt_edges & pred_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)

    precision = safe_divide(tp, len(pred_edges))
    recall = safe_divide(tp, len(gt_edges))

    if precision + recall > 0:
        f1 = safe_divide(2 * precision * recall, precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1, tp, fp, fn


class LocalQwenJudgeLLM:
    """
    Local LLM judge.

    Used by:
      1. workflow MESS when DEFAULT_WORKFLOW_SEMANTIC_BACKEND == "llm"
      2. visual grounding metric
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 16,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

        if use_cuda:
            dev = torch.device(device)
            major, _ = torch.cuda.get_device_capability(dev)
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            dev = torch.device("cpu")
            dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
        )

        if not use_cuda:
            self.model = self.model.to(dev)

        self.model.eval()

        self.input_device = next(self.model.parameters()).device
        print(f"[Judge LLM] loaded on {self.input_device}")

    def __call__(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.input_device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return content, thinking_content


class SemanticScorer:
    """
    Semantic scorer for MESS.

    backend:
      lexical:
        token-overlap similarity

      embedding:
        sentence-transformers cosine similarity

      llm:
        local LLM judge, SAME / DIFFERENT
    """

    def __init__(
        self,
        backend: str = "lexical",
        embedding_model_name: Optional[str] = None,
        llm_judge: Optional[LocalQwenJudgeLLM] = None,
        voting: int = 1,
    ):
        self.backend = backend
        self.embedding_model_name = embedding_model_name
        self.llm_judge = llm_judge
        self.voting = max(1, int(voting))

        self.embedding_model = None

        if self.backend == "embedding":
            if embedding_model_name is None:
                raise ValueError(
                    "semantic backend is 'embedding', but embedding_model_name is None."
                )

            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(embedding_model_name)
            print(f"[Embedding scorer] loaded: {embedding_model_name}")

        if self.backend == "llm" and self.llm_judge is None:
            raise ValueError("semantic backend is 'llm', but no judge LLM is provided.")

    @staticmethod
    def lexical_similarity(a: str, b: str) -> float:
        a_tokens = canonicalize_text(a).split()
        b_tokens = canonicalize_text(b).split()

        if not a_tokens and not b_tokens:
            return 1.0

        if not a_tokens or not b_tokens:
            return 0.0

        a_count = Counter(a_tokens)
        b_count = Counter(b_tokens)

        common = sum((a_count & b_count).values())
        total = sum((a_count | b_count).values())

        return clamp01(safe_divide(common, total))

    def embedding_similarity(self, a: str, b: str) -> float:
        embeddings = self.embedding_model.encode(
            [a, b],
            normalize_embeddings=True,
        )

        sim = float((embeddings[0] * embeddings[1]).sum())

        return clamp01(sim)

    def llm_similarity(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
    ) -> float:
        def one_vote() -> float:
            prompt = f"""
You are a pathology expert.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Task:
Determine whether the two answers are semantically equivalent in a clinical sense.

Rules:
- If both answers express the same diagnosis, grade, morphology, or clinical meaning, output SAME.
- If they differ in any clinically meaningful way, output DIFFERENT.
- Be strict for clinically meaningful differences.
- Ignore only minor wording differences.

Output:
Return only one word:
SAME or DIFFERENT
"""
            result = self.llm_judge(prompt)
            label = parse_judge_label(result, allowed=["SAME", "DIFFERENT"])

            if label == "SAME":
                return 1.0

            return 0.0

        votes = [one_vote() for _ in range(self.voting)]

        return max(set(votes), key=votes.count)

    def similarity(
        self,
        answer_a: str,
        answer_b: str,
        question: str = "",
    ) -> float:
        if self.backend == "lexical":
            return self.lexical_similarity(answer_a, answer_b)

        if self.backend == "embedding":
            return self.embedding_similarity(answer_a, answer_b)

        if self.backend == "llm":
            return self.llm_similarity(question, answer_a, answer_b)

        raise ValueError(f"Unknown semantic backend: {self.backend}")


def compute_mess(
    gt_records: List[EdgeRecord],
    pred_records: List[EdgeRecord],
    scorer: SemanticScorer,
) -> float:
    """
    MESS is averaged over ground-truth edges.

    If a GT edge is missing in prediction, semantic score for this edge is 0.
    """
    if not gt_records:
        return 0.0

    gt_answer_map = edge_answer_map(gt_records)
    pred_answer_map = edge_answer_map(pred_records)
    gt_question_map = edge_question_map(gt_records)

    sims: List[float] = []

    for edge_key, gt_answers in gt_answer_map.items():
        gt_answer = gt_answers[0] if gt_answers else ""
        pred_answers = pred_answer_map.get(edge_key)

        if not pred_answers:
            sims.append(0.0)
            continue

        pred_answer = pred_answers[0] if pred_answers else ""
        question = gt_question_map.get(edge_key, "")

        sim = scorer.similarity(
            answer_a=gt_answer,
            answer_b=pred_answer,
            question=question,
        )

        sims.append(clamp01(sim))

    return float(sum(sims) / len(sims))


def evaluate_workflow_case(
    case_id: str,
    gt_case: Dict[str, Any],
    pred_case: Optional[Dict[str, Any]],
    scorer: SemanticScorer,
) -> CaseScore:
    gt_records = build_edge_records(gt_case)
    pred_records = build_edge_records(pred_case) if pred_case is not None else []

    gt_edges = unique_edge_set(gt_records)
    pred_edges = unique_edge_set(pred_records)

    gt_ordered_edges = ordered_edge_list(gt_records)
    pred_ordered_edges = ordered_edge_list(pred_records)

    bpv = compute_binary_path_validity(gt_edges, pred_edges)
    ordered_bpv = compute_ordered_path_validity(gt_ordered_edges, pred_ordered_edges)

    precision, recall, edge_f1, tp, fp, fn = compute_edge_metrics(
        gt_edges=gt_edges,
        pred_edges=pred_edges,
    )

    mess = compute_mess(
        gt_records=gt_records,
        pred_records=pred_records,
        scorer=scorer,
    )

    ranking_score = 0.35 * bpv + 0.35 * edge_f1 + 0.30 * mess

    return CaseScore(
        case_id=case_id,
        binary_path_validity=bpv,
        ordered_path_validity=ordered_bpv,
        edge_precision=precision,
        edge_recall=recall,
        edge_f1=edge_f1,
        mess=mess,
        ranking_score=ranking_score,
        gt_edge_count=len(gt_edges),
        pred_edge_count=len(pred_edges),
        edge_tp=tp,
        edge_fp=fp,
        edge_fn=fn,
    )


def evaluate_workflow_dataset(
    gt_cases: List[Dict[str, Any]],
    pred_cases: List[Dict[str, Any]],
    scorer: SemanticScorer,
    strict_missing_predictions: bool = False,
) -> Dict[str, Any]:
    gt_index = index_cases(gt_cases)
    pred_index = index_cases(pred_cases)

    case_scores: List[CaseScore] = []

    missing_case_ids: List[str] = []
    extra_case_ids = sorted(set(pred_index.keys()) - set(gt_index.keys()))

    for case_id, gt_case in gt_index.items():
        pred_case = pred_index.get(case_id)

        if pred_case is None:
            missing_case_ids.append(case_id)

            if strict_missing_predictions:
                raise ValueError(f"Missing prediction for case id: {case_id}")

        case_score = evaluate_workflow_case(
            case_id=case_id,
            gt_case=gt_case,
            pred_case=pred_case,
            scorer=scorer,
        )

        case_scores.append(case_score)

    def avg(field: str) -> float:
        if not case_scores:
            return 0.0

        return float(sum(getattr(cs, field) for cs in case_scores) / len(case_scores))

    summary = {
        "num_ground_truth_cases": len(gt_index),
        "num_prediction_cases": len(pred_index),
        "num_missing_prediction_cases": len(missing_case_ids),
        "num_extra_prediction_cases": len(extra_case_ids),

        "missing_prediction_case_ids": missing_case_ids,
        "extra_prediction_case_ids": extra_case_ids,

        "average_binary_path_validity": avg("binary_path_validity"),
        "average_ordered_path_validity": avg("ordered_path_validity"),
        "average_edge_precision": avg("edge_precision"),
        "average_edge_recall": avg("edge_recall"),
        "average_edge_f1": avg("edge_f1"),
        "average_mess": avg("mess"),
        "final_ranking_score": avg("ranking_score"),

        "per_case": [asdict(cs) for cs in case_scores],
    }

    return summary


def workflow_global_average_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "workflow_final_score": float(summary.get("final_ranking_score", 0.0)),
        "average_binary_path_validity": float(summary.get("average_binary_path_validity", 0.0)),
        "average_ordered_path_validity": float(summary.get("average_ordered_path_validity", 0.0)),
        "average_edge_precision": float(summary.get("average_edge_precision", 0.0)),
        "average_edge_recall": float(summary.get("average_edge_recall", 0.0)),
        "average_edge_f1": float(summary.get("average_edge_f1", 0.0)),
        "average_mess": float(summary.get("average_mess", 0.0)),
    }


def run_workflow_batch(
    ground_truth_paths: List[Path],
    prediction_paths: List[Path],
    semantic_backend: str,
    embedding_model: Optional[str],
    judge_llm: Optional[LocalQwenJudgeLLM],
    voting: int,
    strict_missing_predictions: bool,
    merge_predictions: bool,
) -> Dict[str, Any]:
    gt_files = collect_json_files(ground_truth_paths)
    pred_files = collect_json_files(prediction_paths)

    gt_cases = load_case_list_from_files(gt_files)

    scorer = SemanticScorer(
        backend=semantic_backend,
        embedding_model_name=embedding_model,
        llm_judge=judge_llm,
        voting=voting,
    )

    if merge_predictions:
        pred_cases = load_case_list_from_files(pred_files)

        summary = evaluate_workflow_dataset(
            gt_cases=gt_cases,
            pred_cases=pred_cases,
            scorer=scorer,
            strict_missing_predictions=strict_missing_predictions,
        )

        global_average = workflow_global_average_from_summary(summary)
        global_average["num_prediction_files"] = len(pred_files)
        global_average["best_prediction_name"] = "merged_predictions"
        global_average["best_prediction_file"] = None
        global_average["best_workflow_final_score"] = global_average["workflow_final_score"]

        return {
            "mode": "workflow",
            "merge_predictions": True,
            "ground_truth_files": [str(x) for x in gt_files],
            "prediction_files": [str(x) for x in pred_files],
            "global_average": global_average,
            "result": summary,
        }

    evaluations: Dict[str, Any] = {}
    leaderboard: List[Dict[str, Any]] = []

    for pred_file in pred_files:
        print(f"[Workflow] Evaluating prediction file: {pred_file}")

        pred_cases = load_case_list_from_file(pred_file)

        summary = evaluate_workflow_dataset(
            gt_cases=gt_cases,
            pred_cases=pred_cases,
            scorer=scorer,
            strict_missing_predictions=strict_missing_predictions,
        )

        name = pred_file.stem

        evaluations[name] = {
            "prediction_file": str(pred_file),
            "result": summary,
        }

        leaderboard.append(
            {
                "prediction_name": name,
                "prediction_file": str(pred_file),
                "final_ranking_score": summary["final_ranking_score"],
                "average_binary_path_validity": summary["average_binary_path_validity"],
                "average_ordered_path_validity": summary["average_ordered_path_validity"],
                "average_edge_precision": summary["average_edge_precision"],
                "average_edge_recall": summary["average_edge_recall"],
                "average_edge_f1": summary["average_edge_f1"],
                "average_mess": summary["average_mess"],
            }
        )

    leaderboard = sorted(
        leaderboard,
        key=lambda x: x["final_ranking_score"],
        reverse=True,
    )

    best_item = leaderboard[0] if leaderboard else None

    global_average = {
        "num_prediction_files": len(pred_files),

        # Mean across all prediction JSON files.
        # If there is only one baseline prediction JSON, this is exactly its total workflow average.
        "workflow_final_score": mean_dict_values(leaderboard, "final_ranking_score"),
        "average_binary_path_validity": mean_dict_values(leaderboard, "average_binary_path_validity"),
        "average_ordered_path_validity": mean_dict_values(leaderboard, "average_ordered_path_validity"),
        "average_edge_precision": mean_dict_values(leaderboard, "average_edge_precision"),
        "average_edge_recall": mean_dict_values(leaderboard, "average_edge_recall"),
        "average_edge_f1": mean_dict_values(leaderboard, "average_edge_f1"),
        "average_mess": mean_dict_values(leaderboard, "average_mess"),

        # Best prediction file among the submitted prediction JSON files.
        "best_prediction_name": best_item["prediction_name"] if best_item else None,
        "best_prediction_file": best_item["prediction_file"] if best_item else None,
        "best_workflow_final_score": float(best_item["final_ranking_score"]) if best_item else 0.0,
    }

    return {
        "mode": "workflow",
        "merge_predictions": False,
        "ground_truth_files": [str(x) for x in gt_files],
        "prediction_files": [str(x) for x in pred_files],
        "global_average": global_average,
        "leaderboard": leaderboard,
        "evaluations": evaluations,
    }


# # ============================================================
# # Part 2: Visual grounding metric
# # ============================================================
# question_dict_roi_local = {
#     "tissue_presence": [
#         "Does this ROI contain analyzable histological tissue? Answer briefly.",
#     ],

#     "content": [
#         "What is the dominant content in this ROI? Answer briefly.",
#     ],

#     "quality": [
#         "Is this ROI informative for histological image analysis? Answer briefly.",
#     ],
# }



class VisualGroundingEvaluator:
    def __init__(
        self,
        judge_model: Callable[[str], str],
    ):
        self.judge_model = judge_model
        # self.answer_cache: Dict[Tuple[str, str], str] = {}

    # def _get_question_dict(
    #     self,
    #     question_dict: Optional[Dict[str, List[str]]],
    # ) -> Dict[str, List[str]]:
    #     if question_dict is not None:
    #         return question_dict

    #     if self.default_question_dict is not None:
    #         return self.default_question_dict

    #     raise ValueError("question_dict must be provided")

    # def _model_answer(self, roi: Any, question: str) -> str:
    #     """
    #     Cache answers for file-path ROIs.

    #     This avoids repeatedly asking the VLM the same question on the same ROI
    #     across B1/B2/B3.
    #     """
    #     if isinstance(roi, (str, Path)):
    #         key = (str(roi), question)

    #         if key in self.answer_cache:
    #             return self.answer_cache[key]

    #         ans = self.model(roi, question)
    #         self.answer_cache[key] = ans

    #         return ans

    #     return self.model(roi, question)

    def judge_background(self, question: str, answer: str) -> float:
        prompt = f"""
        You are a pathology expert.

        The following image region is from a whole slide image, but it contains NO tissue, background only.

        Question:
        {question}

        Model Answer:
        {answer}

        Task:
        Determine whether the model answer is appropriate for a background-only image region.

        Judgment rules:
        - If the answer claims any diagnostic, morphological, grading, tumor-related, cell-related, tissue-specific, or pathology-specific information, the judgment is INCORRECT.
        - If the answer states that no reliable information can be determined because the region contains no tissue or only background, the judgment is CORRECT.
        - If the answer says background, no tissue, not assessable, not evaluable, or uncertain due to absence of tissue, the judgment is CORRECT.
        - If the answer gives a diagnosis or describes tissue morphology despite the region being background only, the judgment is INCORRECT.

        You may reason internally if needed.

        Final output requirement:
        At the end, output exactly one final judgment using this format:

        <answer>CORRECT</answer>

        or

        <answer>INCORRECT</answer>

        Do not output anything more after your answer.
        """
        result, reasoning = self.judge_model(prompt)
        label = parse_judge_label(result, allowed=["CORRECT", "INCORRECT"])
        print(f"[Judge Background] question: {question}, answer: {answer}, label: {label}, reasoning: {reasoning}")
        if label == "CORRECT":
            return 1.0
        elif label == "INCORRECT":
            return 0.0
        else:
            return -1

    def judge_similarity(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
    ) -> float:
        prompt = f"""
You are a pathology expert.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Task:
Determine whether the two answers are semantically equivalent in a clinical sense.

Judgment rules:
- If both answers express the same diagnosis, grade, morphology, visual content, or clinical meaning, the judgment is SAME.
- If they differ in any clinically meaningful way, the judgment is DIFFERENT.
- Be strict about clinically meaningful differences, including diagnosis, tumor type, benign/malignant status, grade, cell type, morphology, tissue structure, staining pattern, uncertainty level, and clinical implication.
- Ignore only minor wording differences that do not change the clinical meaning.
- If one answer is more specific than the other and the added specificity changes or narrows the clinical meaning, the judgment is DIFFERENT.
- If one answer makes a confident claim while the other says the finding is uncertain, not assessable, or cannot be determined, the judgment is DIFFERENT.

You may reason internally if needed.

Final output requirement:
At the end, output exactly one final judgment using this format:

<answer>SAME</answer>

or

<answer>DIFFERENT</answer>

Do not repeat the answer.
Do not output anything after the closing </answer> tag.
"""
        result, reasoning = self.judge_model(prompt)
        label = parse_judge_label(result, allowed=["SAME", "DIFFERENT"])

        if label == "SAME":
            return 1.0
        elif label == "DIFFERENT":
            return 0.0
        else:
            return -1

    def vote(self, func: Callable[[], float], voting: int) -> float:
        voting = max(1, int(voting))
        results = [func() for _ in range(voting)]

        return max(set(results), key=results.count)

    def background_rejection(
        self,
        question_dict: Dict[str, List[str]],
        background_rois: List[Any],
        voting: int = 1,
    ) -> float:
        scores = []

        for roi in background_rois:
            roi_scores = []

            for _, questions in question_dict.items():
                for q in questions:
                    ans = self._model_answer(roi, q)

                    score = self.vote(
                        lambda q=q, ans=ans: self.judge_background(q, ans),
                        voting,
                    )
                    if score == -1:
                        roi_scores.append(0.0)
                    else:
                        roi_scores.append(score)

            if roi_scores:
                scores.append(float(np.mean(roi_scores)))

        return float(np.mean(scores)) if scores else 0.0

    def input_sensitivity(
        self,
        question_dict: Dict[str, List[str]],
        tissue_rois: List[Any],
        perturb_fn: Callable[[Any], Any],
        voting: int = 1,
    ) -> float:
        if perturb_fn is None:
            raise ValueError("perturb_fn must be provided for input_sensitivity")

        diffs = []

        for roi in tissue_rois:
            for _, questions in question_dict.items():
                for q in questions:
                    original = self._model_answer(roi, q)

                    perturbed_roi = perturb_fn(roi)
                    perturbed = self.model(perturbed_roi, q)

                    sim = self.vote(
                        lambda q=q, original=original, perturbed=perturbed:
                            self.judge_similarity(q, original, perturbed),
                        voting,
                    )
                    if sim == -1:
                        diffs.append(0.0)
                    else:
                        diffs.append(sim)

        return float(np.mean(diffs)) if diffs else 0.0

    def cross_region_consistency(
        self,
        question_dict: Dict[str, List[str]],
        tissue_rois: List[Any],
        background_rois: List[Any],
        voting: int = 1,
    ) -> float:
        scores = []

        for _, questions in question_dict.items():
            for q in questions:
                tissue_answers = [self._model_answer(roi, q) for roi in tissue_rois]
                background_answers = [self._model_answer(roi, q) for roi in background_rois]

                for ta in tissue_answers:
                    for ba in background_answers:
                        sim = self.vote(
                            lambda q=q, ta=ta, ba=ba:
                                self.judge_similarity(q, ta, ba),
                            voting,
                        )
                        if sim == -1:
                            scores.append(0.0)
                        else:
                            scores.append(1.0 - sim)

        return float(np.mean(scores)) if scores else 0.0

    def compute_score(
        self,
        question_dict: Optional[Dict[str, List[str]]] = None,
        tissue_rois: Optional[List[Any]] = None,
        background_rois: Optional[List[Any]] = None,
        perturb_fn: Optional[Callable[[Any], Any]] = None,
        w1: float = 0.3,
        w2: float = 0.3,
        w3: float = 0.4,
        voting: int = 1,
    ) -> Dict[str, float]:
        question_dict = self._get_question_dict(question_dict)

        tissue_rois = tissue_rois or []
        background_rois = background_rois or []

        b1 = self.background_rejection(
            question_dict=question_dict,
            background_rois=background_rois,
            voting=voting,
        )

        b2 = self.input_sensitivity(
            question_dict=question_dict,
            tissue_rois=tissue_rois,
            perturb_fn=perturb_fn,
            voting=voting,
        )

        b3 = self.cross_region_consistency(
            question_dict=question_dict,
            tissue_rois=tissue_rois,
            background_rois=background_rois,
            voting=voting,
        )

        final_score = w1 * b1 + w2 * b2 + w3 * b3

        return {
            "B1_background": float(b1),
            "B2_sensitivity": float(b2),
            "B3_cross_region": float(b3),
            "B_final": float(final_score),
        }

def load_visual_answer_json(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load participant visual answer JSON.

    Expected format:
    [
      {
        "id": "roi_000000",
        "image": "...",
        "question": "...",
        "answer": "..."
      },
      ...
    ]
    """
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError(f"Visual answer JSON must be a list: {path}")

    answers_by_id: Dict[str, Dict[str, Any]] = {}

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Visual answer item {idx} is not a dict.")

        for key in ["id", "question", "answer"]:
            if key not in item:
                raise ValueError(f"Visual answer item {idx} missing key: {key}")

        roi_id = str(item["id"])

        if roi_id in answers_by_id:
            raise ValueError(f"Duplicate visual ROI id found: {roi_id}")

        answers_by_id[roi_id] = item

    return answers_by_id

def load_anonymous_mapping_txt(path: Path) -> List[Dict[str, str]]:
    """
    Load anonymous mapping txt generated by create_json_anoy.py.

    Required columns:
    anonymous_id
    variant
    paired_anonymous_id
    b3_paired_anonymous_id
    label
    question
    """
    rows: List[Dict[str, str]] = []

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        if reader.fieldnames is None:
            raise ValueError(f"Mapping txt has no header: {path}")

        required_cols = {
            "anonymous_id",
            "variant",
            "paired_anonymous_id",
            "b3_paired_anonymous_id",
            "label",
            "question",
        }

        missing_cols = required_cols - set(reader.fieldnames)

        if missing_cols:
            raise ValueError(
                f"Mapping txt missing columns: {sorted(missing_cols)}"
            )

        for row in reader:
            clean_row = {
                k: (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
            }
            rows.append(clean_row)

    return rows

def load_anonymous_mapping_txt_as_dict(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load anonymous mapping txt generated by create_json_anoy.py.

    Required columns:
    anonymous_id
    variant
    paired_anonymous_id
    b3_paired_anonymous_id
    label
    question
    """
    rows: Dict[str, Dict[str, str]] = {}

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        if reader.fieldnames is None:
            raise ValueError(f"Mapping txt has no header: {path}")

        required_cols = {
            "anonymous_id",
            "variant",
            "paired_anonymous_id",
            "b3_paired_anonymous_id",
            "label",
            "question",
        }

        missing_cols = required_cols - set(reader.fieldnames)

        if missing_cols:
            raise ValueError(
                f"Mapping txt missing columns: {sorted(missing_cols)}"
            )

        for row in reader:
            clean_row = {
                k: (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
            }
            rows[clean_row['anonymous_id']] = clean_row

    return rows


def get_answer_text(item: Optional[Dict[str, Any]]) -> str:
    if item is None:
        return ""
    return str(item.get("answer", "") or "").strip()


def get_question_text(
    item: Optional[Dict[str, Any]],
    fallback: str = "",
) -> str:
    if item is None:
        return str(fallback or "").strip()

    return str(item.get("question", fallback) or fallback or "").strip()


def score_or_zero(score: float) -> float:
    """
    The judge returns -1 when parsing fails.
    Treat parsing failure as 0.0 score.
    """
    if score == -1:
        return 0.0
    return float(score)

def score_or_zero_b3(score: float) -> float:
    """
    The judge returns -1 when parsing fails.
    Treat parsing failure as 0.0 score.
    """
    if score == -1:
        return 0.0
    b3_score = 1.0 - float(score)
    return b3_score


def compute_b1_background_from_answers(
    evaluator: VisualGroundingEvaluator,
    answers_by_id: Dict[str, Dict[str, Any]],
    mapping_rows: List[Dict[str, str]],
    voting: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    """
    B1:
    Use background original ROIs only.
    """
    scores = []
    used_ids = []
    missing_ids = []

    for row in mapping_rows:
        if row.get("label") != "background":
            continue

        if row.get("variant") != "original":
            continue

        id = row["anonymous_id"]
        id_image = row["image"]
        id_question = row["question"]
        item = answers_by_id.get(id)
        if item:
            image_pred = item["image"]
            question_pred = item["question"]

        if item is None or (id_image != image_pred) or (id_question != question_pred):
            missing_ids.append(id)
            scores.append(0.0)
            continue           

        #question = get_question_text(item, fallback=row.get("question", ""))
        question = str(id_question).strip()
        #answer = get_answer_text(item)
        answer = str(item.get("answer", "") or "").strip()

        score = evaluator.vote(
            lambda question=question, answer=answer:
                evaluator.judge_background(question, answer),
            voting,
        )

        scores.append(score_or_zero(score))
        used_ids.append(id)

    return (
        float(np.mean(scores)) if scores else 0.0,
        {
            "num_used_background_original": len(used_ids),
            "missing_background_original_ids": missing_ids,
        },
    )


def compute_b2_sensitivity_from_answers(
    evaluator: VisualGroundingEvaluator,
    answers_by_id: Dict[str, Dict[str, Any]],
    mapping_rows: List[Dict[str, str]],
    mapping_rows_dict: Dict[str, Dict[str, str]],
    voting: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    """
    B2:
    Use tissue original ROIs only.

    Compare:
        tissue original answer
        vs
        its paired tissue perturbed answer

    Pairing is from:
        paired_anonymous_id
    """
    scores = []
    used_pairs = []
    missing_pairs = []

    for row in mapping_rows:
        if row.get("label") != "tissue":
            continue

        if row.get("variant") != "original":
            continue

        if row.get("paired_anonymous_id") == "none":
            continue

        original_id = row["anonymous_id"]
        original_image = row["image"]
        original_question = row["question"]
        perturbed_id = row["paired_anonymous_id"]
        perturbed_image = mapping_rows_dict[perturbed_id]['image']
        perturbed_question = mapping_rows_dict[perturbed_id]['question']

        #original_id = row["anonymous_id"]
        #perturbed_id = row.get("paired_anonymous_id", "none")

        #if perturbed_id == "none":
        #    missing_pairs.append({
        #        "original_id": original_id,
        #        "perturbed_id": perturbed_id,
        #        "reason": "paired_anonymous_id is none",
        #    })
        #    continue

        original_item = answers_by_id.get(original_id)
        perturbed_item = answers_by_id.get(perturbed_id)

        if original_item:
            original_image_pred = original_item["image"]
            original_question_pred = original_item["question"]

        if perturbed_item:
            perturbed_image_pred = perturbed_item["image"]
            perturbed_question_pred = perturbed_item["question"]

        if original_item is None or perturbed_item is None or (original_image != original_image_pred) or (perturbed_image != perturbed_image_pred) or (original_question != original_question_pred) or (perturbed_question != perturbed_question_pred):
            missing_pairs.append({
                "original_id": original_id,
                "perturbed_id": perturbed_id,
                "missing_original_answer": original_item is None,
                "missing_perturbed_answer": perturbed_item is None,
            })
            scores.append(0.0)
            continue

        #question = get_question_text(original_item, fallback=row.get("question", ""))
        #original_answer = get_answer_text(original_item)
        #perturbed_answer = get_answer_text(perturbed_item)
        question = str(original_question).strip()
        original_answer = str(original_item.get("answer", "") or "").strip()
        perturbed_answer = str(perturbed_item.get("answer", "") or "").strip()

        sim = evaluator.vote(
            lambda question=question,
                   original_answer=original_answer,
                   perturbed_answer=perturbed_answer:
                evaluator.judge_similarity(
                    question,
                    original_answer,
                    perturbed_answer,
                ),
            voting,
        )

        scores.append(score_or_zero(sim))

        used_pairs.append({
            "original_id": original_id,
            "perturbed_id": perturbed_id,
            "pair_id": row.get("pair_id", ""),
        })

    return (
        float(np.mean(scores)) if scores else 0.0,
        {
            "num_used_tissue_original_perturbed_pairs": len(used_pairs),
            "used_pairs": used_pairs,
            "missing_pairs": missing_pairs,
        },
    )


def compute_b3_cross_region_from_answers(
    evaluator: VisualGroundingEvaluator,
    answers_by_id: Dict[str, Dict[str, Any]],
    mapping_rows: List[Dict[str, str]],
    mapping_rows_dict: Dict[str, Dict[str, str]],
    voting: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    """
    B3:
    Use tissue original ROIs only.

    Compare each tissue original answer with exactly one paired background original answer.

    Pairing is from:
        b3_paired_anonymous_id

    This avoids full tissue-background pairwise comparison.
    """
    scores = []
    used_pairs = []
    missing_pairs = []
    question_mismatch_pairs = []

    row_by_id = {
        row["anonymous_id"]: row
        for row in mapping_rows
        if row.get("anonymous_id")
    }

    for row in mapping_rows:
        if row.get("label") != "tissue":
            continue

        if row.get("variant") != "original":
            continue

        if row.get("b3_paired_anonymous_id") == "none":
            continue

        tissue_id = row["anonymous_id"]
        tissue_image = row["image"]
        tissue_question = row["question"]
        background_id = row["b3_paired_anonymous_id"]
        background_image = mapping_rows_dict[background_id]['image']
        background_question = mapping_rows_dict[background_id]['question']

        #tissue_id = row["anonymous_id"]
        #background_id = row.get("b3_paired_anonymous_id", "none")

        #if background_id == "none":
        #    missing_pairs.append({
        #        "tissue_id": tissue_id,
        #        "background_id": background_id,
        #        "reason": "b3_paired_anonymous_id is none",
        #    })
        #    continue

        tissue_item = answers_by_id.get(tissue_id)
        background_item = answers_by_id.get(background_id)
        background_row = row_by_id.get(background_id)

        if tissue_item:
            tissue_image_pred = tissue_item["image"]
            tissue_question_pred = tissue_item["question"]

        if background_item:
            background_image_pred = background_item["image"]
            background_question_pred = background_item["question"]

        if tissue_item is None or background_item is None or background_row is None or (tissue_image != tissue_image_pred) or (background_image != background_image_pred) or (tissue_question != tissue_question_pred) or (background_question != background_question_pred):
            missing_pairs.append({
                "tissue_id": tissue_id,
                "background_id": background_id,
                "missing_tissue_answer": tissue_item is None,
                "missing_background_answer": background_item is None,
                "missing_background_mapping": background_row is None,
            })
            scores.append(0.0)
            continue


        #tissue_question = get_question_text(tissue_item, fallback=row.get("question", ""))
        tissue_question = str(tissue_question).strip()
        background_question = str(background_question).strip()

        if canonicalize_text(tissue_question) != canonicalize_text(background_question):
            question_mismatch_pairs.append({
                "tissue_id": tissue_id,
                "background_id": background_id,
                "tissue_question": tissue_question,
                "background_question": background_question,
            })

        #tissue_answer = get_answer_text(tissue_item)
        tissue_answer = str(tissue_item.get("answer", "") or "").strip()
        #background_answer = get_answer_text(background_item)
        background_answer = str(background_item.get("answer", "") or "").strip()

        sim = evaluator.vote(
            lambda tissue_question=tissue_question,
                   tissue_answer=tissue_answer,
                   background_answer=background_answer:
                evaluator.judge_similarity(
                    tissue_question,
                    tissue_answer,
                    background_answer,
                ),
            voting,
        )

        scores.append(score_or_zero_b3(sim))                    

        used_pairs.append({
            "tissue_id": tissue_id,
            "background_id": background_id,
        })

    return (
        float(np.mean(scores)) if scores else 0.0,
        {
            "num_used_b3_pairs": len(used_pairs),
            "used_pairs": used_pairs,
            "missing_pairs": missing_pairs,
            "question_mismatch_pairs": question_mismatch_pairs,
        },
    )


def run_visual_dataset_from_answer_json(
    visual_answer_json_path: Path,
    mapping_txt_path: Path,
    judge_llm: LocalQwenJudgeLLM,
    w1: float,
    w2: float,
    w3: float,
    voting: int,
) -> Dict[str, Any]:
    """
    New Metric B entry.

    No VLM.
    No image reading.
    No perturb_fn.

    Inputs:
    - participant visual answer JSON
    - internal anonymous mapping txt
    """
    answers_by_id = load_visual_answer_json(visual_answer_json_path)
    mapping_rows = load_anonymous_mapping_txt(mapping_txt_path)
    mapping_rows_dict = load_anonymous_mapping_txt_as_dict(mapping_txt_path)

    evaluator = VisualGroundingEvaluator(
        judge_model=judge_llm,
    )

    b1, b1_detail = compute_b1_background_from_answers(
        evaluator=evaluator,
        answers_by_id=answers_by_id,
        mapping_rows=mapping_rows,
        voting=voting,
    )

    b2, b2_detail = compute_b2_sensitivity_from_answers(
        evaluator=evaluator,
        answers_by_id=answers_by_id,
        mapping_rows=mapping_rows,
        mapping_rows_dict=mapping_rows_dict,
        voting=voting,
    )

    b3, b3_detail = compute_b3_cross_region_from_answers(
        evaluator=evaluator,
        answers_by_id=answers_by_id,
        mapping_rows=mapping_rows,
        mapping_rows_dict=mapping_rows_dict,
        voting=voting,
    )

    final_score = w1 * b1 + w2 * b2 + w3 * b3

    global_average = {
        "visual_final_score": float(final_score),
        "average_B1_background": float(b1),
        "average_B2_sensitivity": float(b2),
        "average_B3_cross_region": float(b3),
        "num_visual_answers": len(answers_by_id),
        "num_mapping_rows": len(mapping_rows),
    }

    summary = {
        "mode": "visual_answer_json",
        "visual_answer_json": str(visual_answer_json_path),
        "mapping_txt": str(mapping_txt_path),

        "average_B1_background": float(b1),
        "average_B2_sensitivity": float(b2),
        "average_B3_cross_region": float(b3),
        "final_visual_score": float(final_score),

        "global_average": global_average,

        "details": {
            "B1_background": b1_detail,
            "B2_sensitivity": b2_detail,
            "B3_cross_region": b3_detail,
        },
    }

    return summary

# ============================================================
# Final summary helpers
# ============================================================

def summarize_workflow_results(workflow_results: Dict[str, Any]) -> Dict[str, Any]:
    global_avg = workflow_results.get("global_average", {})

    return {
        "workflow_final_score": float(global_avg.get("workflow_final_score", 0.0)),
        "average_binary_path_validity": float(global_avg.get("average_binary_path_validity", 0.0)),
        "average_ordered_path_validity": float(global_avg.get("average_ordered_path_validity", 0.0)),
        "average_edge_precision": float(global_avg.get("average_edge_precision", 0.0)),
        "average_edge_recall": float(global_avg.get("average_edge_recall", 0.0)),
        "average_edge_f1": float(global_avg.get("average_edge_f1", 0.0)),
        "average_mess": float(global_avg.get("average_mess", 0.0)),
        "num_prediction_files": int(global_avg.get("num_prediction_files", 0)),
        "best_prediction_name": global_avg.get("best_prediction_name"),
        "best_prediction_file": global_avg.get("best_prediction_file"),
        "best_workflow_final_score": float(global_avg.get("best_workflow_final_score", 0.0)),
    }


def summarize_visual_results(visual_results: Dict[str, Any]) -> Dict[str, Any]:
    global_avg = visual_results.get("global_average", {})

    return {
        "visual_final_score": float(global_avg.get("visual_final_score", 0.0)),
        "average_B1_background": float(global_avg.get("average_B1_background", 0.0)),
        "average_B2_sensitivity": float(global_avg.get("average_B2_sensitivity", 0.0)),
        "average_B3_cross_region": float(global_avg.get("average_B3_cross_region", 0.0)),
        "num_visual_cases": int(global_avg.get("num_visual_cases", 0)),
    }


# ============================================================
# ALL-only CLI
# ============================================================
def extract_workflow_final_ranking_score(workflow_results: Dict[str, Any]) -> float:
    """
    Extract the official workflow score.

    For challenge use, each submission should ideally correspond to one prediction JSON.
    If DEFAULT_MERGE_PREDICTIONS=True, use the merged result.
    If DEFAULT_MERGE_PREDICTIONS=False and multiple prediction files are evaluated,
    use the mean over prediction files.
    """

    # Case 1: merged prediction files
    if workflow_results.get("merge_predictions", False):
        return float(
            workflow_results
            .get("result", {})
            .get("final_ranking_score", 0.0)
        )

    # Case 2: non-merged prediction files, use leaderboard mean
    leaderboard = workflow_results.get("leaderboard", [])

    if not leaderboard:
        return 0.0

    return float(
        sum(float(x.get("final_ranking_score", 0.0)) for x in leaderboard)
        / len(leaderboard)
    )


def extract_visual_final_score(visual_results: Dict[str, Any]) -> float:
    """
    Extract the official visual grounding score.

    This is:
        final_score = w1 * B1 + w2 * B2 + w3 * B3

    In run_visual_dataset, this is stored as final_visual_score.
    """
    return float(visual_results.get("final_visual_score", 0.0))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ALL-only evaluator: always run workflow metric and visual grounding metric."
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        nargs="+",
        required=True,
        help="Ground-truth workflow JSON file(s) or directory.",
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="Prediction workflow JSON file(s) or directory.",
    )

    parser.add_argument(
        "--visual-json",
        type=Path,
        required=True,
        help="Participant visual answer JSON. Each item contains id, image, question, answer.",
    )

    parser.add_argument(
        "--visual-mapping-txt",
        type=Path,
        required=True,
        help="Internal anonymous ROI mapping txt generated by create_json_anoy.py.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("all_scores.json"),
        help="Output JSON path. Default: all_scores.json",
    )


    parser.add_argument(
        "--judge-model-path",
        type=str,
        default=DEFAULT_JUDGE_MODEL_PATH,
        help=f"Path to local judge LLM. Default: {DEFAULT_JUDGE_MODEL_PATH}",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device. Default: {DEFAULT_DEVICE}",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 100)
    print("[Evaluator] Running ALL metrics")
    print("=" * 100)

    print(f"[GT]          {args.ground_truth}")
    print(f"[Prediction]  {args.predictions}")
    print(f"[Visual JSON] {args.visual_json}")
    print(f"[Visual Mapping TXT] {args.visual_mapping_txt}")
    print(f"[Output]      {args.output}")
    print(f"[Judge LLM]   {args.judge_model_path}")
    print(f"[Device]      {args.device}")

    print("-" * 100)
    print("[Config]")
    print(f"workflow_semantic_backend = {DEFAULT_WORKFLOW_SEMANTIC_BACKEND}")
    print(f"embedding_model            = {DEFAULT_EMBEDDING_MODEL}")
    print(f"merge_predictions          = {DEFAULT_MERGE_PREDICTIONS}")
    print(f"strict_missing_predictions = {DEFAULT_STRICT_MISSING_PREDICTIONS}")
    print(f"visual_weights             = w1={DEFAULT_W1}, w2={DEFAULT_W2}, w3={DEFAULT_W3}")
    print(f"voting                     = {DEFAULT_VOTING}")
    print(f"skip_missing_roi           = {DEFAULT_SKIP_MISSING_ROI}")
    print("=" * 100)

    # ------------------------------------------------------------
    # Load judge LLM
    # ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("[0/2] Loading judge LLM")
    print("=" * 100)

    judge_llm = LocalQwenJudgeLLM(
        model_path=args.judge_model_path,
        device=args.device,
        max_new_tokens=DEFAULT_JUDGE_MAX_NEW_TOKENS,
    )

    # ------------------------------------------------------------
    # Run workflow metric
    # ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("[1/2] Running workflow reasoning metric")
    print("=" * 100)

    workflow_results = run_workflow_batch(
        ground_truth_paths=args.ground_truth,
        prediction_paths=args.predictions,
        semantic_backend=DEFAULT_WORKFLOW_SEMANTIC_BACKEND,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        judge_llm=judge_llm if DEFAULT_WORKFLOW_SEMANTIC_BACKEND == "llm" else None,
        voting=DEFAULT_VOTING,
        strict_missing_predictions=DEFAULT_STRICT_MISSING_PREDICTIONS,
        merge_predictions=DEFAULT_MERGE_PREDICTIONS,
    )

    # ------------------------------------------------------------
    # Run visual grounding metric
    # ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("[2/2] Running visual grounding metric")
    print("=" * 100)

    visual_results = run_visual_dataset_from_answer_json(
        visual_answer_json_path=args.visual_json,
        mapping_txt_path=args.visual_mapping_txt,
        judge_llm=judge_llm,
        w1=DEFAULT_W1,
        w2=DEFAULT_W2,
        w3=DEFAULT_W3,
        voting=DEFAULT_VOTING,
    )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------
    workflow_summary = summarize_workflow_results(workflow_results)
    visual_summary = summarize_visual_results(visual_results)

    overall_two_metric_mean = 0.5 * (
        workflow_summary["workflow_final_score"]
        + visual_summary["visual_final_score"]
    )

    workflow_final_ranking_score = extract_workflow_final_ranking_score(workflow_results)
    visual_final_score = extract_visual_final_score(visual_results)

    # submission_id:
    # If one prediction JSON is provided, use its filename stem.
    # If a prediction directory or multiple files are provided, use "submission".
    pred_files = collect_json_files(args.predictions)
    if len(pred_files) == 1:
        submission_id = pred_files[0].stem
    else:
        submission_id = "submission"

    results = {
        "submission_id": submission_id,

        # These are the two official challenge metrics.
        "official_scores": {
            "workflow_final_ranking_score": float(workflow_final_ranking_score),
            "visual_final_score": float(visual_final_score),
        },

        "score_definitions": {
            "workflow_final_ranking_score": (
                "0.35 * Binary Path Validity + "
                "0.35 * Edge-F1 + "
                "0.30 * MESS"
            ),
            "visual_final_score": (
                f"{DEFAULT_W1} * B1_background + "
                f"{DEFAULT_W2} * B2_sensitivity + "
                f"{DEFAULT_W3} * B3_cross_region"
            ),
        },

        "config": {
                "workflow_semantic_backend": DEFAULT_WORKFLOW_SEMANTIC_BACKEND,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
                "merge_predictions": DEFAULT_MERGE_PREDICTIONS,
                "strict_missing_predictions": DEFAULT_STRICT_MISSING_PREDICTIONS,
                "visual_weights": {
                    "w1": DEFAULT_W1,
                    "w2": DEFAULT_W2,
                    "w3": DEFAULT_W3,
                },
                "voting": DEFAULT_VOTING,
                "judge_model_path": args.judge_model_path,
                "device": args.device,
            },

        # Keep full details for debugging, not for official leaderboard display.
        "details": {
            "workflow": workflow_results,
            "visual": visual_results,
        },
    }

    print("\n" + "=" * 100)
    print("[Done] Official challenge scores")
    print("=" * 100)
    print(json.dumps(results["official_scores"], indent=2, ensure_ascii=False))

    save_json(results, args.output)

if __name__ == "__main__":
    main()