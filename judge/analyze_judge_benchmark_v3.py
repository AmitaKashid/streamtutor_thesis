from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from judge.run_judge_benchmark_v3 import (
    DEFAULT_DATASET_DIR,
    _read_json,
    _read_jsonl,
    _write_json_atomic,
    build_evaluation_tasks,
    summarize_predictions,
    verify_dataset,
)
from judge.scalar_evaluator_v3 import (
    ANSWER_DIMENSIONS,
    ALL_DIMENSIONS,
    RETRIEVAL_DIMENSION,
)


ANALYZER_VERSION = "3.0"
PROVISIONAL_REFERENCE = "provisional_ai"
HUMAN_REFERENCE = "human"
SCORE_MINIMUM = 1
SCORE_MAXIMUM = 5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_score(value: Any, *, allow_na: bool = False) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if allow_na and isinstance(value, str) and value.strip().lower() in {
        "n/a",
        "na",
        "not_applicable",
    }:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a valid score: {value!r}")
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid score: {value!r}") from exc
    if str(value).strip() not in {str(number) for number in range(1, 6)}:
        raise ValueError(f"Score must be an integer from 1 to 5: {value!r}")
    if not SCORE_MINIMUM <= score <= SCORE_MAXIMUM:
        raise ValueError(f"Score out of range: {score}")
    return score


def quadratic_weighted_kappa(
    reference: Sequence[int],
    prediction: Sequence[int],
) -> float | None:
    if len(reference) != len(prediction):
        raise ValueError("Reference and prediction lengths differ")
    if not reference:
        return None
    for score in (*reference, *prediction):
        if not SCORE_MINIMUM <= score <= SCORE_MAXIMUM:
            raise ValueError(f"Score out of range: {score}")

    size = SCORE_MAXIMUM - SCORE_MINIMUM + 1
    observed = [[0.0 for _ in range(size)] for _ in range(size)]
    reference_histogram = [0.0 for _ in range(size)]
    prediction_histogram = [0.0 for _ in range(size)]
    for actual, predicted in zip(reference, prediction):
        row = actual - SCORE_MINIMUM
        column = predicted - SCORE_MINIMUM
        observed[row][column] += 1.0
        reference_histogram[row] += 1.0
        prediction_histogram[column] += 1.0

    count = float(len(reference))
    numerator = 0.0
    denominator = 0.0
    normalizer = float((size - 1) ** 2)
    for row in range(size):
        for column in range(size):
            weight = ((row - column) ** 2) / normalizer
            expected = (
                reference_histogram[row]
                * prediction_histogram[column]
                / count
            )
            numerator += weight * observed[row][column]
            denominator += weight * expected

    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        # Kappa is undefined when the expected weighted disagreement is zero,
        # even if a constant-label sample agrees exactly. Exact agreement is
        # reported separately and must not be relabeled as chance-corrected
        # agreement.
        return None
    return 1.0 - (numerator / denominator)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


@dataclass(frozen=True)
class ReferenceScore:
    score: int
    source: str
    rater_id: str | None


def _index_dataset_responses(
    responses: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    by_response: dict[str, Mapping[str, Any]] = {}
    by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in responses:
        response_id = row.get("response_id")
        question_id = row.get("question_id")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("Dataset response has no response_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"Response {response_id} has no question_id")
        if response_id in by_response:
            raise ValueError(f"Duplicate response_id: {response_id}")
        by_response[response_id] = row
        by_question[question_id].append(row)
    return by_response, dict(by_question)


def _provisional_answer_score(
    response: Mapping[str, Any],
    dimension: str,
) -> int | None:
    annotation = response.get("provisional_annotation")
    if not isinstance(annotation, Mapping):
        raise ValueError("Response lacks provisional_annotation")
    dimensions = annotation.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("Provisional annotation lacks dimensions")
    value = dimensions.get(dimension)
    if not isinstance(value, Mapping):
        raise ValueError(f"Provisional annotation lacks {dimension}")
    if value.get("applicable") is False:
        return None
    return _parse_score(value.get("score"), allow_na=True)


def _provisional_retrieval_score(
    question_id: str,
    question_responses: Sequence[Mapping[str, Any]],
) -> int:
    scores: set[int] = set()
    for response in question_responses:
        annotation = response.get("provisional_annotation")
        if not isinstance(annotation, Mapping):
            raise ValueError("Response lacks provisional_annotation")
        retrieval = annotation.get("retrieval_relevance")
        if not isinstance(retrieval, Mapping):
            raise ValueError("Annotation lacks retrieval_relevance")
        score = _parse_score(retrieval.get("score"))
        if score is not None:
            scores.add(score)
    if len(scores) != 1:
        raise ValueError(
            f"Question {question_id} has inconsistent provisional "
            f"retrieval scores: {sorted(scores)}"
        )
    return next(iter(scores))


def load_human_annotations(
    path: str | Path,
    *,
    rater_id: str,
) -> dict[str, dict[str, int | None]]:
    annotation_path = Path(path)
    selected: dict[str, dict[str, int | None]] = {}
    with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "response_id",
            "rater_id",
            "correctness",
            "groundedness_or_na",
            "completeness",
            "clarity",
            "pedagogy",
            "learner_appropriateness",
            "conversational_context",
            "retrieval_relevance",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Human annotation file is missing columns: "
                + ", ".join(sorted(missing))
            )
        for line_number, row in enumerate(reader, start=2):
            if (row.get("rater_id") or "").strip() != rater_id:
                continue
            response_id = (row.get("response_id") or "").strip()
            if not response_id:
                raise ValueError(
                    f"Human annotation line {line_number} has no response_id"
                )
            if response_id in selected:
                raise ValueError(
                    f"Duplicate human annotation for {response_id}/{rater_id}"
                )
            selected[response_id] = {
                "correctness": _parse_score(row.get("correctness")),
                "groundedness": _parse_score(
                    row.get("groundedness_or_na"), allow_na=True
                ),
                "completeness": _parse_score(row.get("completeness")),
                "clarity": _parse_score(row.get("clarity")),
                "pedagogy": _parse_score(row.get("pedagogy")),
                "learner_appropriateness": _parse_score(
                    row.get("learner_appropriateness")
                ),
                "conversational_context": _parse_score(
                    row.get("conversational_context")
                ),
                "retrieval_relevance": _parse_score(
                    row.get("retrieval_relevance")
                ),
            }
    if not selected:
        raise ValueError(f"No annotations found for rater_id={rater_id!r}")
    return selected


def _human_retrieval_score(
    *,
    question_id: str,
    question_responses: Sequence[Mapping[str, Any]],
    human: Mapping[str, Mapping[str, int | None]],
) -> int | None:
    scores: set[int] = set()
    for response in question_responses:
        response_id = str(response["response_id"])
        annotation = human.get(response_id)
        if annotation is None:
            continue
        score = annotation.get(RETRIEVAL_DIMENSION)
        if score is not None:
            scores.add(score)
    if not scores:
        return None
    if len(scores) != 1:
        raise ValueError(
            f"Question {question_id} has inconsistent human retrieval "
            f"scores for the selected rater: {sorted(scores)}"
        )
    return next(iter(scores))


def _reference_for_prediction(
    prediction: Mapping[str, Any],
    *,
    reference_source: str,
    response_by_id: Mapping[str, Mapping[str, Any]],
    responses_by_question: Mapping[str, Sequence[Mapping[str, Any]]],
    human: Mapping[str, Mapping[str, int | None]] | None,
    rater_id: str | None,
) -> ReferenceScore | None:
    dimension = prediction.get("dimension")
    question_id = prediction.get("question_id")
    response_id = prediction.get("response_id")
    if dimension not in ALL_DIMENSIONS:
        raise ValueError(f"Unknown prediction dimension: {dimension!r}")
    if not isinstance(question_id, str):
        raise ValueError("Prediction has no question_id")

    if dimension == RETRIEVAL_DIMENSION:
        question_rows = responses_by_question.get(question_id, [])
        if reference_source == PROVISIONAL_REFERENCE:
            score = _provisional_retrieval_score(question_id, question_rows)
        else:
            assert human is not None
            score = _human_retrieval_score(
                question_id=question_id,
                question_responses=question_rows,
                human=human,
            )
    else:
        if not isinstance(response_id, str):
            raise ValueError(
                f"Answer prediction has no response_id: {prediction.get('evaluation_id')}"
            )
        response = response_by_id.get(response_id)
        if response is None:
            raise ValueError(f"Unknown response_id: {response_id}")
        if reference_source == PROVISIONAL_REFERENCE:
            score = _provisional_answer_score(response, str(dimension))
        else:
            assert human is not None
            score = human.get(response_id, {}).get(str(dimension))

    if score is None:
        return None
    return ReferenceScore(
        score=score,
        source=reference_source,
        rater_id=rater_id,
    )


def _prediction_score(prediction: Mapping[str, Any]) -> int | None:
    judge = prediction.get("judge")
    if not isinstance(judge, Mapping) or judge.get("status") != "valid":
        return None
    result = judge.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("Valid prediction has no result object")
    return _parse_score(result.get("score"))


def build_scored_rows(
    *,
    predictions: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    reference_source: str,
    human_annotations: Mapping[str, Mapping[str, int | None]] | None = None,
    rater_id: str | None = None,
) -> list[dict[str, Any]]:
    response_by_id, responses_by_question = _index_dataset_responses(responses)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        evaluation_id = prediction.get("evaluation_id")
        if not isinstance(evaluation_id, str):
            raise ValueError("Prediction has no evaluation_id")
        if evaluation_id in seen:
            raise ValueError(f"Duplicate prediction: {evaluation_id}")
        seen.add(evaluation_id)
        reference = _reference_for_prediction(
            prediction,
            reference_source=reference_source,
            response_by_id=response_by_id,
            responses_by_question=responses_by_question,
            human=human_annotations,
            rater_id=rater_id,
        )
        response_id = prediction.get("response_id")
        dataset_row = (
            response_by_id.get(str(response_id))
            if response_id is not None
            else (responses_by_question.get(str(prediction.get("question_id"))) or [None])[0]
        )
        judge = prediction.get("judge")
        status = judge.get("status") if isinstance(judge, Mapping) else None
        rows.append(
            {
                "evaluation_id": evaluation_id,
                "question_id": prediction.get("question_id"),
                "response_id": response_id,
                "conversation_id": (
                    dataset_row.get("conversation_id")
                    if isinstance(dataset_row, Mapping)
                    else None
                ),
                "dimension": prediction.get("dimension"),
                "judge_status": status,
                "prediction_score": _prediction_score(prediction),
                "reference_score": reference.score if reference else None,
                "reference_source": reference.source if reference else reference_source,
                "rater_id": reference.rater_id if reference else rater_id,
                "controlled_quality_level": (
                    dataset_row.get("controlled_quality_level")
                    if isinstance(dataset_row, Mapping) and response_id is not None
                    else None
                ),
            }
        )
    return rows


def _bootstrap_qwk(
    rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations <= 0:
        return {"iterations_requested": iterations, "iterations_valid": 0,
                "lower_95": None, "upper_95": None}
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cluster = row.get("conversation_id")
        if not isinstance(cluster, str) or not cluster:
            raise ValueError("Scored row lacks conversation_id")
        clusters[cluster].append(row)
    cluster_ids = sorted(clusters)
    if not cluster_ids:
        return {"iterations_requested": iterations, "iterations_valid": 0,
                "lower_95": None, "upper_95": None}
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled_rows: list[Mapping[str, Any]] = []
        for cluster_id in rng.choices(cluster_ids, k=len(cluster_ids)):
            sampled_rows.extend(clusters[cluster_id])
        value = quadratic_weighted_kappa(
            [int(row["reference_score"]) for row in sampled_rows],
            [int(row["prediction_score"]) for row in sampled_rows],
        )
        if value is not None and math.isfinite(value):
            samples.append(value)
    return {
        "method": "conversation_cluster_bootstrap",
        "seed": seed,
        "iterations_requested": iterations,
        "iterations_valid": len(samples),
        "lower_95": _percentile(samples, 0.025),
        "upper_95": _percentile(samples, 0.975),
    }


def agreement_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    reference_available = [
        row for row in rows if row.get("reference_score") is not None
    ]
    paired = [
        row
        for row in reference_available
        if row.get("prediction_score") is not None
    ]
    reference = [int(row["reference_score"]) for row in paired]
    prediction = [int(row["prediction_score"]) for row in paired]
    differences = [pred - actual for actual, pred in zip(reference, prediction)]
    qwk = quadratic_weighted_kappa(reference, prediction)
    seed_material = str(rows[0].get("dimension", "unknown")) if rows else "empty"
    dimension_seed = bootstrap_seed + int(
        hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16
    )
    return {
        "prediction_count": len(rows),
        "reference_applicable_count": len(reference_available),
        "paired_valid_count": len(paired),
        "coverage_of_applicable_reference": (
            len(paired) / len(reference_available) if reference_available else None
        ),
        "quadratic_weighted_kappa": qwk,
        "qwk_95_ci": _bootstrap_qwk(
            paired,
            iterations=bootstrap_iterations,
            seed=dimension_seed,
        ),
        "mean_absolute_error": (
            statistics.mean(abs(value) for value in differences)
            if differences else None
        ),
        "mean_signed_error_prediction_minus_reference": (
            statistics.mean(differences) if differences else None
        ),
        "exact_agreement_rate": (
            sum(value == 0 for value in differences) / len(differences)
            if differences else None
        ),
        "within_one_rate": (
            sum(abs(value) <= 1 for value in differences) / len(differences)
            if differences else None
        ),
        "reference_distribution": dict(sorted(Counter(reference).items())),
        "prediction_distribution": dict(sorted(Counter(prediction).items())),
    }


def macro_answer_qwk(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    paired = [
        row
        for row in rows
        if row.get("dimension") in ANSWER_DIMENSIONS
        and row.get("reference_score") is not None
        and row.get("prediction_score") is not None
    ]
    by_dimension: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired:
        by_dimension[str(row["dimension"])].append(row)
    point_values: dict[str, float] = {}
    for dimension in ANSWER_DIMENSIONS:
        dimension_rows = by_dimension.get(dimension, [])
        value = quadratic_weighted_kappa(
            [int(row["reference_score"]) for row in dimension_rows],
            [int(row["prediction_score"]) for row in dimension_rows],
        )
        if value is not None:
            point_values[dimension] = value
    point = (
        statistics.mean(point_values.values()) if point_values else None
    )

    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired:
        conversation_id = row.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("Scored row lacks conversation_id")
        clusters[conversation_id].append(row)
    cluster_ids = sorted(clusters)
    rng = random.Random(bootstrap_seed)
    samples: list[float] = []
    if bootstrap_iterations > 0 and cluster_ids and point_values:
        eligible_dimensions = tuple(sorted(point_values))
        for _ in range(bootstrap_iterations):
            sampled: list[Mapping[str, Any]] = []
            for cluster_id in rng.choices(cluster_ids, k=len(cluster_ids)):
                sampled.extend(clusters[cluster_id])
            sampled_by_dimension: dict[
                str, list[Mapping[str, Any]]
            ] = defaultdict(list)
            for row in sampled:
                sampled_by_dimension[str(row["dimension"])].append(row)
            values: list[float] = []
            valid_iteration = True
            for dimension in eligible_dimensions:
                dimension_rows = sampled_by_dimension.get(dimension, [])
                value = quadratic_weighted_kappa(
                    [int(row["reference_score"]) for row in dimension_rows],
                    [int(row["prediction_score"]) for row in dimension_rows],
                )
                if value is None or not math.isfinite(value):
                    valid_iteration = False
                    break
                values.append(value)
            if valid_iteration:
                samples.append(statistics.mean(values))
    return {
        "value": point,
        "included_dimensions": sorted(point_values),
        "dimension_count": len(point_values),
        "qwk_95_ci": {
            "method": "conversation_cluster_bootstrap_macro",
            "seed": bootstrap_seed,
            "iterations_requested": bootstrap_iterations,
            "iterations_valid": len(samples),
            "lower_95": _percentile(samples, 0.025),
            "upper_95": _percentile(samples, 0.975),
        },
    }


def quality_level_discrimination(
    scored_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_response: dict[str, list[int]] = defaultdict(list)
    applicable_count: Counter[str] = Counter()
    level_by_response: dict[str, str] = {}
    question_by_response: dict[str, str] = {}
    for row in scored_rows:
        if row.get("dimension") not in ANSWER_DIMENSIONS:
            continue
        response_id = row.get("response_id")
        level = row.get("controlled_quality_level")
        score = row.get("prediction_score")
        if not isinstance(response_id, str) or not isinstance(level, str):
            continue
        applicable_count[response_id] += 1
        if score is not None:
            by_response[response_id].append(int(score))
        level_by_response[response_id] = level
        question_by_response[response_id] = str(row.get("question_id"))

    response_means = {
        response_id: statistics.mean(scores)
        for response_id, scores in by_response.items()
        if scores
        and applicable_count[response_id] > 0
        and len(scores) == applicable_count[response_id]
    }
    incomplete_response_count = sum(
        count > 0 and len(by_response.get(response_id, [])) != count
        for response_id, count in applicable_count.items()
    )
    grouped: dict[str, list[float]] = defaultdict(list)
    for response_id, mean_score in response_means.items():
        grouped[level_by_response[response_id]].append(mean_score)
    levels = {}
    for level in ("critical", "borderline", "excellent"):
        values = grouped.get(level, [])
        levels[level] = {
            "response_count": len(values),
            "mean_response_score": statistics.mean(values) if values else None,
            "median_response_score": statistics.median(values) if values else None,
        }

    triplets: dict[str, dict[str, float]] = defaultdict(dict)
    for response_id, mean_score in response_means.items():
        triplets[question_by_response[response_id]][
            level_by_response[response_id]
        ] = mean_score
    complete = [
        values
        for values in triplets.values()
        if {"critical", "borderline", "excellent"}.issubset(values)
    ]
    strict = sum(
        values["excellent"] > values["borderline"] > values["critical"]
        for values in complete
    )
    return {
        "score_definition": (
            "mean of all reference-applicable answer-dimension predictions; "
            "responses with any missing or invalid applicable prediction are excluded"
        ),
        "excluded_incomplete_response_count": incomplete_response_count,
        "levels": levels,
        "complete_question_triplets": len(complete),
        "strict_excellent_gt_borderline_gt_critical_count": strict,
        "strict_order_accuracy": strict / len(complete) if complete else None,
    }


def analyze_run(
    *,
    run_dir: str | Path,
    dataset_dir: str | Path,
    reference_source: str,
    acknowledge_provisional_not_human: bool = False,
    human_labels: str | Path | None = None,
    rater_id: str | None = None,
    human_reference_status: str | None = None,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 20261107,
    allow_incomplete: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if reference_source == PROVISIONAL_REFERENCE:
        if not acknowledge_provisional_not_human:
            raise ValueError(
                "Provisional AI annotations are not human ground truth; pass "
                "acknowledge_provisional_not_human=True for exploratory analysis"
            )
        if (
            human_labels is not None
            or rater_id is not None
            or human_reference_status is not None
        ):
            raise ValueError("Human-label arguments cannot accompany provisional mode")
    elif reference_source == HUMAN_REFERENCE:
        if human_labels is None or not rater_id:
            raise ValueError("Human mode requires human_labels and rater_id")
        if human_reference_status not in {"individual", "adjudicated"}:
            raise ValueError(
                "Human mode requires human_reference_status to be "
                "'individual' or 'adjudicated'"
            )
    else:
        raise ValueError(f"Unknown reference source: {reference_source}")
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be non-negative")

    dataset = verify_dataset(dataset_dir)
    root: Path = dataset["root"]
    run_root = Path(run_dir).resolve()
    manifest = _read_json(run_root / "manifest.json")
    predictions = _read_jsonl(run_root / "predictions.jsonl")
    questions = _read_jsonl(root / "questions.jsonl")
    responses = _read_jsonl(root / "responses.jsonl")

    manifest_dataset = manifest.get("dataset")
    if not isinstance(manifest_dataset, Mapping):
        raise ValueError("Run manifest has no dataset identity")
    expected_identity = {
        "id": dataset["dataset_id"],
        "version": dataset["dataset_version"],
        "checksums_sha256": dataset["checksums_sha256"],
        "responses_sha256": dataset["checked_files"]["responses.jsonl"],
        "questions_sha256": dataset["checked_files"]["questions.jsonl"],
        "rubric_sha256": dataset["checked_files"]["rubric_v3.json"],
    }
    for key, expected in expected_identity.items():
        if manifest_dataset.get(key) != expected:
            raise ValueError(
                f"Run/dataset provenance mismatch for {key}: "
                f"{manifest_dataset.get(key)!r} != {expected!r}"
            )
    split = manifest.get("split")
    if split not in {"calibration", "selection", "robustness"}:
        raise ValueError(f"Run manifest has invalid split: {split!r}")
    expected_tasks = build_evaluation_tasks(
        questions=questions,
        responses=responses,
        split=str(split),
    )
    expected_ids = {task.evaluation_id for task in expected_tasks}
    actual_ids = {
        str(prediction.get("evaluation_id")) for prediction in predictions
    }
    unknown_ids = sorted(actual_ids.difference(expected_ids))
    if unknown_ids:
        raise ValueError(
            "Predictions contain evaluations outside the frozen task plan: "
            + ", ".join(unknown_ids[:10])
        )
    planned = manifest.get("planned_evaluation_count")
    if planned != len(expected_ids):
        raise ValueError(
            "Manifest planned_evaluation_count does not match the frozen "
            f"task plan: {planned!r} != {len(expected_ids)}"
        )
    is_complete = (
        isinstance(planned, int)
        and planned == len(predictions)
        and actual_ids == expected_ids
        and manifest.get("status") == "completed"
    )
    if not is_complete and not allow_incomplete:
        raise ValueError(
            "Run is incomplete; pass allow_incomplete=True only for diagnostics"
        )

    human = (
        load_human_annotations(human_labels, rater_id=str(rater_id))
        if reference_source == HUMAN_REFERENCE
        else None
    )
    if human is not None:
        dataset_response_ids = {
            str(response["response_id"]) for response in responses
        }
        unknown_human_ids = sorted(set(human).difference(dataset_response_ids))
        if unknown_human_ids:
            raise ValueError(
                "Human annotation file contains unknown response IDs: "
                + ", ".join(unknown_human_ids[:10])
            )
    scored = build_scored_rows(
        predictions=predictions,
        responses=responses,
        reference_source=reference_source,
        human_annotations=human,
        rater_id=rater_id,
    )
    by_dimension: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored:
        by_dimension[str(row["dimension"])].append(row)
    dimension_metrics = {
        dimension: agreement_metrics(
            by_dimension.get(dimension, []),
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
        for dimension in ALL_DIMENSIONS
    }
    macro_qwk = macro_answer_qwk(
        scored,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    report = {
        "analyzer_version": ANALYZER_VERSION,
        "created_utc": _utc_now(),
        "run_id": manifest.get("run_id"),
        "run_status": manifest.get("status"),
        "run_complete": is_complete,
        "split": manifest.get("split"),
        "judge_configuration": manifest.get("judge_configuration"),
        "dataset": expected_identity,
        "reference": {
            "source": reference_source,
            "rater_id": rater_id,
            "human_reference_status": human_reference_status,
            "confirmatory_human_ground_truth": (
                reference_source == HUMAN_REFERENCE
                and human_reference_status == "adjudicated"
            ),
            "warning": (
                "Exploratory only: AI-generated provisional annotations are "
                "not independent human ground truth and cannot support final "
                "model-selection claims."
                if reference_source == PROVISIONAL_REFERENCE
                else (
                    "An individual human rater is not adjudicated ground truth."
                    if human_reference_status == "individual"
                    else None
                )
            ),
        },
        "reliability": summarize_predictions(predictions),
        "agreement_by_dimension": dimension_metrics,
        "macro_answer_dimension_qwk": macro_qwk,
        "quality_level_discrimination": quality_level_discrimination(scored),
        "critical_error_evaluation": {
            "status": "not_measured",
            "reason": (
                "Scalar protocol v2.1 predicts ordinal dimension scores, not "
                "critical-error codes; score thresholds are not a valid "
                "substitute for critical-error recall or precision."
            ),
        },
        "controlled_contrast_evaluation": {
            "status": "not_measured",
            "reason": (
                "The frozen contrasts require bidirectional pairwise judging; "
                "this analysis consumes scalar benchmark predictions only."
            ),
        },
    }
    return report, scored


def _write_scored_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "evaluation_id",
        "question_id",
        "response_id",
        "conversation_id",
        "dimension",
        "judge_status",
        "prediction_score",
        "reference_score",
        "reference_source",
        "rater_id",
        "controlled_quality_level",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _write_dimension_csv(path: Path, report: Mapping[str, Any]) -> None:
    fields = (
        "dimension",
        "prediction_count",
        "reference_applicable_count",
        "paired_valid_count",
        "coverage_of_applicable_reference",
        "quadratic_weighted_kappa",
        "qwk_ci_lower_95",
        "qwk_ci_upper_95",
        "mean_absolute_error",
        "mean_signed_error_prediction_minus_reference",
        "exact_agreement_rate",
        "within_one_rate",
    )
    metrics = report["agreement_by_dimension"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dimension in ALL_DIMENSIONS:
            values = metrics[dimension]
            ci = values["qwk_95_ci"]
            writer.writerow(
                {
                    "dimension": dimension,
                    **{key: values.get(key) for key in fields[1:6]},
                    "qwk_ci_lower_95": ci.get("lower_95"),
                    "qwk_ci_upper_95": ci.get("upper_95"),
                    **{key: values.get(key) for key in fields[8:]},
                }
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a StreamTutor judge benchmark v3 run",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--reference-source",
        required=True,
        choices=(PROVISIONAL_REFERENCE, HUMAN_REFERENCE),
    )
    parser.add_argument(
        "--acknowledge-provisional-not-human",
        action="store_true",
    )
    parser.add_argument("--human-labels", default=None)
    parser.add_argument("--rater-id", default=None)
    parser.add_argument(
        "--human-reference-status",
        choices=("individual", "adjudicated"),
        default=None,
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261107)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report, scored = analyze_run(
        run_dir=args.run_dir,
        dataset_dir=args.dataset_dir,
        reference_source=args.reference_source,
        acknowledge_provisional_not_human=(
            args.acknowledge_provisional_not_human
        ),
        human_labels=args.human_labels,
        rater_id=args.rater_id,
        human_reference_status=args.human_reference_status,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        allow_incomplete=args.allow_incomplete,
    )
    run_dir = Path(args.run_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / f"analysis_{args.reference_source}"
    )
    if output_dir.exists() and not args.overwrite:
        raise ValueError(
            f"Analysis output already exists; use --overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "analysis_report.json", report)
    _write_scored_csv(output_dir / "scored_predictions.csv", scored)
    _write_dimension_csv(output_dir / "agreement_by_dimension.csv", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Analysis directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
