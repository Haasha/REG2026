import csv
import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from helpers import run_prediction_processing, setup_logger, tree

logger = logging.getLogger('evaluate')

INPUT_DIRECTORY = Path('/input')
OUTPUT_DIRECTORY = Path('/output')
GROUND_TRUTH_DIRECTORY = Path('/opt/ml/input/data/ground_truth')

WORKFLOW_WEIGHT_BPV = 0.35
WORKFLOW_WEIGHT_EDGE_F1 = 0.35
WORKFLOW_WEIGHT_MESS = 0.30

VISUAL_W1 = 0.3
VISUAL_W2 = 0.3
VISUAL_W3 = 0.4


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str


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


def safe_divide(num: float, den: float) -> float:
    return num / den if den else 0.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def canonicalize_text(text: Optional[str]) -> str:
    if text is None:
        return ''
    text = str(text).lower()
    text = normalize_whitespace(text)
    text = re.sub(r'[\s\.,;:!?]+$', '', text)
    return text


TERMINAL_TOKENS = {
    '',
    'end',
    'stop',
    'finish',
    'finished',
    'none',
    'null',
    'no next question',
    'no further question',
}


def canonicalize_next_question(text: Optional[str]) -> str:
    token = canonicalize_text(text)
    return '__END__' if token in TERMINAL_TOKENS else token


def tokenize(text: str) -> List[str]:
    return re.findall(r'[a-z0-9]+', canonicalize_text(text))


def lexical_similarity(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)

    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0

    set_a = set(ta)
    set_b = set(tb)

    return clamp01(safe_divide(len(set_a & set_b), len(set_a | set_b)))


def load_json_file(location: Path) -> Any:
    with location.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(location: Path, content: Any) -> None:
    with location.open('w', encoding='utf-8') as f:
        json.dump(content, f, indent=2, ensure_ascii=False)


def get_interface_key(job: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(sorted(v['socket']['slug'] for v in job.get('inputs', [])))


def get_socket_value(values: List[Dict[str, Any]], slug: str) -> Dict[str, Any]:
    for value in values:
        if value['socket']['slug'] == slug:
            return value
    raise RuntimeError(f'Socket {slug} not found.')


def read_job_output_json(job: Dict[str, Any], slug: str) -> Any:
    socket_value = get_socket_value(job.get('outputs', []), slug)

    if socket_value.get('value') is not None:
        return socket_value['value']

    relative_path = socket_value['socket']['relative_path']
    location = INPUT_DIRECTORY / job['pk'] / 'output' / relative_path
    return load_json_file(location)


def extract_image_basename(file_url: Optional[str], fallback_relative_path: str) -> str:
    if file_url:
        parsed = urlparse(file_url)
        candidate = Path(parsed.path).name
        if candidate:
            return candidate
    return Path(fallback_relative_path).name


def process(job: Dict[str, Any]) -> Dict[str, Any]:
    interface_key = get_interface_key(job)

    if interface_key == (
        'histopathology-region-of-interest-thumbnail',
        'visual-context-question',
    ):
        return process_interf0(job)

    if interface_key == ('whole-slide-image',):
        return process_interf1(job)

    return {
        'pk': job.get('pk'),
        'interface': list(interface_key),
        'status': 'skipped_unknown_interface',
    }


def process_interf0(job: Dict[str, Any]) -> Dict[str, Any]:
    visual_response = read_job_output_json(job, 'visual-context-response')

    question_value = get_socket_value(job.get('inputs', []), 'visual-context-question')
    question = str(question_value.get('value') or '').strip()

    image_value = get_socket_value(job.get('inputs', []), 'histopathology-region-of-interest-thumbnail')
    image_name = extract_image_basename(
        image_value.get('file'),
        image_value['socket']['relative_path'],
    )

    answer = visual_response if isinstance(visual_response, str) else json.dumps(visual_response, ensure_ascii=False)

    return {
        'pk': job.get('pk'),
        'interface': 'interf0',
        'status': 'ok',
        'visual_item': {
            'id': None,
            'image': image_name,
            'question': question,
            'answer': str(answer),
        },
    }


def process_interf1(job: Dict[str, Any]) -> Dict[str, Any]:
    chain_of_thought = read_job_output_json(job, 'chain-of-thought')

    image_value = get_socket_value(job.get('inputs', []), 'whole-slide-image')
    case_id = str((image_value.get('image') or {}).get('name') or job.get('pk'))

    if not isinstance(chain_of_thought, list):
        chain_of_thought = []

    return {
        'pk': job.get('pk'),
        'interface': 'interf1',
        'status': 'ok',
        'workflow_case': {
            'id': case_id,
            'chain-of-thought': chain_of_thought,
        },
    }


def build_edge_records(case_obj: Dict[str, Any]) -> List[Edge]:
    steps = get_workflow_steps(case_obj)

    records: List[Edge] = []

    for step in steps:
        question = str(step.get('question', '') or '')
        next_question = str(step.get('next_question', '') or '')

        src = canonicalize_text(question)
        dst = canonicalize_next_question(next_question)

        records.append(Edge(src=src, dst=dst))

    return records


def get_workflow_steps(case_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ['chain-of-thought', 'chain_of_thought', 'workflow', 'workflow_steps', 'steps']:
        if key in case_obj and isinstance(case_obj[key], list):
            return case_obj[key]
    return []


def edge_answer_map(case_obj: Dict[str, Any]) -> Dict[Tuple[str, str], str]:
    mapping: Dict[Tuple[str, str], str] = {}
    for step in get_workflow_steps(case_obj):
        src = canonicalize_text(step.get('question', ''))
        dst = canonicalize_next_question(step.get('next_question', ''))
        mapping[(src, dst)] = str(step.get('answer', '') or '')
    return mapping


def compute_edge_metrics(gt_edges: set[Tuple[str, str]], pred_edges: set[Tuple[str, str]]) -> Tuple[float, float, float]:
    tp = len(gt_edges & pred_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall) if precision + recall else 0.0

    return precision, recall, f1


def compute_mess(gt_case: Dict[str, Any], pred_case: Optional[Dict[str, Any]]) -> float:
    gt_map = edge_answer_map(gt_case)
    pred_map = edge_answer_map(pred_case or {})

    if not gt_map:
        return 0.0

    scores = []
    for edge_key, gt_answer in gt_map.items():
        pred_answer = pred_map.get(edge_key, '')
        scores.append(lexical_similarity(gt_answer, pred_answer))

    return float(mean(scores)) if scores else 0.0


def evaluate_workflow_case(case_id: str, gt_case: Dict[str, Any], pred_case: Optional[Dict[str, Any]]) -> CaseScore:
    gt_records = build_edge_records(gt_case)
    pred_records = build_edge_records(pred_case or {})

    gt_edges = {(edge.src, edge.dst) for edge in gt_records}
    pred_edges = {(edge.src, edge.dst) for edge in pred_records}

    gt_ordered = [(edge.src, edge.dst) for edge in gt_records]
    pred_ordered = [(edge.src, edge.dst) for edge in pred_records]

    bpv = 1.0 if gt_edges == pred_edges else 0.0
    ordered_bpv = 1.0 if gt_ordered == pred_ordered else 0.0

    precision, recall, edge_f1 = compute_edge_metrics(gt_edges, pred_edges)
    mess = compute_mess(gt_case, pred_case)

    ranking_score = (
        WORKFLOW_WEIGHT_BPV * bpv
        + WORKFLOW_WEIGHT_EDGE_F1 * edge_f1
        + WORKFLOW_WEIGHT_MESS * mess
    )

    return CaseScore(
        case_id=case_id,
        binary_path_validity=bpv,
        ordered_path_validity=ordered_bpv,
        edge_precision=precision,
        edge_recall=recall,
        edge_f1=edge_f1,
        mess=mess,
        ranking_score=ranking_score,
    )


def index_by_case_id(cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        cid = str(case.get('id', '') or '').strip()
        if cid:
            out[cid] = case
    return out


def id_stem(case_id: str) -> str:
    return Path(case_id).stem.lower().strip()


def evaluate_workflow_dataset(gt_cases: List[Dict[str, Any]], pred_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    gt_index = index_by_case_id(gt_cases)
    pred_index = index_by_case_id(pred_cases)

    pred_by_stem = {id_stem(k): v for k, v in pred_index.items()}

    case_scores: List[CaseScore] = []
    missing_prediction_case_ids: List[str] = []

    for case_id, gt_case in gt_index.items():
        pred_case = pred_index.get(case_id)
        if pred_case is None:
            pred_case = pred_by_stem.get(id_stem(case_id))

        if pred_case is None:
            missing_prediction_case_ids.append(case_id)

        case_scores.append(evaluate_workflow_case(case_id, gt_case, pred_case))

    global_average = {
        'workflow_final_score': float(mean([x.ranking_score for x in case_scores])) if case_scores else 0.0,
        'average_binary_path_validity': float(mean([x.binary_path_validity for x in case_scores])) if case_scores else 0.0,
        'average_ordered_path_validity': float(mean([x.ordered_path_validity for x in case_scores])) if case_scores else 0.0,
        'average_edge_precision': float(mean([x.edge_precision for x in case_scores])) if case_scores else 0.0,
        'average_edge_recall': float(mean([x.edge_recall for x in case_scores])) if case_scores else 0.0,
        'average_edge_f1': float(mean([x.edge_f1 for x in case_scores])) if case_scores else 0.0,
        'average_mess': float(mean([x.mess for x in case_scores])) if case_scores else 0.0,
    }

    return {
        'num_ground_truth_cases': len(gt_index),
        'num_prediction_cases': len(pred_index),
        'num_missing_prediction_cases': len(missing_prediction_case_ids),
        'missing_prediction_case_ids': missing_prediction_case_ids,
        'per_case': [asdict(x) for x in case_scores],
        'global_average': global_average,
    }


def parse_mapping_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        required = {
            'anonymous_id',
            'image',
            'question',
            'variant',
            'label',
            'paired_anonymous_id',
            'b3_paired_anonymous_id',
        }

        if not reader.fieldnames:
            raise ValueError('Mapping file has no header.')

        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f'Mapping file missing columns: {sorted(missing)}')

        for row in reader:
            rows.append({k: (v.strip() if isinstance(v, str) else '') for k, v in row.items()})

    return rows


def find_ground_truth_assets() -> Tuple[Path, Path]:
    if not GROUND_TRUTH_DIRECTORY.exists():
        raise FileNotFoundError(f'Ground truth directory missing: {GROUND_TRUTH_DIRECTORY}')

    mapping_candidates = sorted(GROUND_TRUTH_DIRECTORY.rglob('*.txt'))
    mapping_path = None
    for candidate in mapping_candidates:
        try:
            parse_mapping_rows(candidate)
            mapping_path = candidate
            break
        except Exception:
            continue

    if mapping_path is None:
        raise FileNotFoundError('Could not locate a valid visual mapping txt in ground truth.')

    workflow_candidates = sorted(GROUND_TRUTH_DIRECTORY.rglob('*.json'))
    workflow_path = None
    largest_case_count = -1

    for candidate in workflow_candidates:
        try:
            data = load_json_file(candidate)
            cases = data if isinstance(data, list) else data.get('cases', []) if isinstance(data, dict) else []
            if not isinstance(cases, list) or not cases:
                continue
            valid_count = sum(1 for c in cases if isinstance(c, dict) and c.get('id'))
            if valid_count > largest_case_count:
                workflow_path = candidate
                largest_case_count = valid_count
        except Exception:
            continue

    if workflow_path is None:
        raise FileNotFoundError('Could not locate a valid workflow ground truth json in ground truth.')

    return workflow_path, mapping_path


def canonical_pair_key(image: str, question: str) -> Tuple[str, str]:
    return (canonicalize_text(Path(image).name), canonicalize_text(question))


def assign_visual_ids(visual_items: List[Dict[str, Any]], mapping_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[Tuple[str, str], List[str]] = {}
    for row in mapping_rows:
        key = canonical_pair_key(row.get('image', ''), row.get('question', ''))
        lookup.setdefault(key, []).append(row['anonymous_id'])

    assigned: Dict[str, Dict[str, Any]] = {}

    for item in visual_items:
        key = canonical_pair_key(str(item.get('image', '')), str(item.get('question', '')))
        ids = lookup.get(key, [])

        anonymous_id = None
        while ids:
            candidate = ids.pop(0)
            if candidate not in assigned:
                anonymous_id = candidate
                break

        if anonymous_id is None:
            anonymous_id = f'unmatched_{len(assigned)}'

        assigned[anonymous_id] = {
            'id': anonymous_id,
            'image': str(item.get('image', '')),
            'question': str(item.get('question', '')),
            'answer': str(item.get('answer', '')),
        }

    return assigned


def judge_background_answer(answer: str) -> float:
    text = canonicalize_text(answer)

    negative_markers = [
        'no tissue',
        'background',
        'not evaluable',
        'not assessable',
        'cannot assess',
        'insufficient',
        'artifact only',
        'acellular',
        'blank',
        'empty',
    ]
    positive_markers = [
        'tumor',
        'carcinoma',
        'adenocarcinoma',
        'dysplasia',
        'metaplasia',
        'inflammation',
        'necrosis',
        'gland',
        'epitheli',
        'stroma',
        'histolog',
        'tissue',
        'cell',
        'diagnos',
    ]

    has_negative = any(m in text for m in negative_markers)
    has_positive = any(m in text for m in positive_markers)

    if has_negative and not has_positive:
        return 1.0
    if has_positive and not has_negative:
        return 0.0
    if has_negative and has_positive:
        return 0.0

    if re.search(r'\b(no|none|cannot|unable)\b', text):
        return 1.0

    return 0.0


def compute_visual_scores(visual_items: List[Dict[str, Any]], mapping_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    answers_by_id = assign_visual_ids(visual_items, mapping_rows)
    row_by_id = {row['anonymous_id']: row for row in mapping_rows}

    b1_scores: List[float] = []
    b2_scores: List[float] = []
    b3_scores: List[float] = []

    for row in mapping_rows:
        anonymous_id = row['anonymous_id']
        item = answers_by_id.get(anonymous_id)
        answer = '' if item is None else str(item.get('answer', ''))

        if row.get('label') == 'background' and row.get('variant') == 'original':
            b1_scores.append(judge_background_answer(answer))

        if row.get('label') == 'tissue' and row.get('variant') == 'original':
            paired = row.get('paired_anonymous_id', 'none')
            if paired and paired != 'none':
                paired_item = answers_by_id.get(paired)
                paired_answer = '' if paired_item is None else str(paired_item.get('answer', ''))
                b2_scores.append(lexical_similarity(answer, paired_answer))

            b3_pair = row.get('b3_paired_anonymous_id', 'none')
            if b3_pair and b3_pair != 'none' and b3_pair in row_by_id:
                background_item = answers_by_id.get(b3_pair)
                background_answer = '' if background_item is None else str(background_item.get('answer', ''))
                b3_scores.append(1.0 - lexical_similarity(answer, background_answer))

    b1 = float(mean(b1_scores)) if b1_scores else 0.0
    b2 = float(mean(b2_scores)) if b2_scores else 0.0
    b3 = float(mean(b3_scores)) if b3_scores else 0.0
    final_visual = VISUAL_W1 * b1 + VISUAL_W2 * b2 + VISUAL_W3 * b3

    return {
        'global_average': {
            'visual_final_score': float(final_visual),
            'average_B1_background': b1,
            'average_B2_sensitivity': b2,
            'average_B3_cross_region': b3,
            'num_visual_answers': len(visual_items),
        }
    }


def log_inputs() -> None:
    logger.info('Input Files:')
    for line in tree(INPUT_DIRECTORY):
        logger.info(line)


def read_predictions() -> List[Dict[str, Any]]:
    return load_json_file(INPUT_DIRECTORY / 'predictions.json')


def main() -> int:
    setup_logger(level=logging.INFO)
    log_inputs()

    predictions = read_predictions()
    processed = run_prediction_processing(fn=process, predictions=predictions)

    workflow_cases = [x['workflow_case'] for x in processed if x.get('workflow_case')]
    visual_items = [x['visual_item'] for x in processed if x.get('visual_item')]

    workflow_gt_path, visual_mapping_path = find_ground_truth_assets()
    workflow_gt_data = load_json_file(workflow_gt_path)
    workflow_gt_cases = workflow_gt_data if isinstance(workflow_gt_data, list) else workflow_gt_data.get('cases', [])

    if not isinstance(workflow_gt_cases, list):
        workflow_gt_cases = []

    mapping_rows = parse_mapping_rows(visual_mapping_path)

    workflow_results = evaluate_workflow_dataset(workflow_gt_cases, workflow_cases)
    visual_results = compute_visual_scores(visual_items, mapping_rows)

    workflow_final = float(workflow_results['global_average']['workflow_final_score'])
    visual_final = float(visual_results['global_average']['visual_final_score'])

    metrics = {
        'results': processed,
        'aggregates': {
            'workflow_final_ranking_score': workflow_final,
            'visual_final_score': visual_final,
            'overall_two_metric_mean': 0.5 * (workflow_final + visual_final),
        },
        'official_scores': {
            'workflow_final_ranking_score': workflow_final,
            'visual_final_score': visual_final,
        },
        'details': {
            'workflow': workflow_results,
            'visual': visual_results,
            'ground_truth_files': {
                'workflow_json': str(workflow_gt_path),
                'visual_mapping_txt': str(visual_mapping_path),
            },
        },
    }

    write_json_file(OUTPUT_DIRECTORY / 'metrics.json', metrics)
    logger.info('Wrote metrics to /output/metrics.json')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
