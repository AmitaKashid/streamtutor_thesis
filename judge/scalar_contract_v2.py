from __future__ import annotations

import re
from typing import Any


SCALAR_PROTOCOL_VERSION = "2.1"
RESULT_MARKER = "[RESULT]"

_SCALAR_OUTPUT_PATTERN = re.compile(
    r"(?P<feedback>.*?)\s*"
    r"\[RESULT\]\s*(?P<score>[1-5])",
    flags=re.DOTALL,
)


def parse_scalar_output(
    text: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Parse one protocol-v2.1 scalar judge response.

    The accepted grammar is non-empty feedback followed by:

        [RESULT] <integer 1-5>

    The literal ``Feedback:`` prefix is accepted but optional because it is
    semantically redundant. Its presence is recorded as provenance. The
    result marker must be unique and must terminate the response. No Markdown
    extraction, score clamping, score inference or output repair is performed.
    """
    if not isinstance(text, str):
        return None, "output_type_error: expected string"

    stripped = text.strip()

    if not stripped:
        return None, "empty_output"

    marker_count = stripped.count(RESULT_MARKER)
    if marker_count != 1:
        return (
            None,
            "result_marker_count_error: "
            f"expected 1 marker, found {marker_count}",
        )

    if "```" in stripped:
        return None, "format_error: Markdown fences are not permitted"

    match = _SCALAR_OUTPUT_PATTERN.fullmatch(stripped)
    if match is None:
        return (
            None,
            "format_error: expected non-empty feedback followed by "
            "'[RESULT] <integer 1-5>' with no trailing text",
        )

    feedback = match.group("feedback").strip()
    format_variant = "unprefixed_feedback"
    if feedback.startswith("Feedback:"):
        feedback = feedback[len("Feedback:") :].strip()
        format_variant = "prefixed_feedback"

    if not feedback:
        return None, "empty_feedback"

    score = int(match.group("score"))

    return {
        "protocol_version": SCALAR_PROTOCOL_VERSION,
        "feedback": feedback,
        "score": score,
        "format_variant": format_variant,
    }, None


def build_scalar_retry_prompt(
    original_prompt: str,
    validation_error: str,
) -> str:
    """
    Create the single permitted retry prompt.

    The retry reports only the formatting error. It never supplies or
    suggests an expected score.
    """
    if not original_prompt.strip():
        raise ValueError("original_prompt must not be empty")

    if not validation_error.strip():
        raise ValueError("validation_error must not be empty")

    return (
        f"{original_prompt.rstrip()}\n\n"
        "RETRY_NOTICE: Your previous response did not satisfy scalar "
        "evaluation protocol v2.1. Return exactly one response in this "
        "format:\n"
        "Feedback: <concise evidence-based explanation>\n"
        "[RESULT] <integer from 1 to 5>\n"
        "Do not use Markdown. Do not include text after the result. "
        f"Validation error: {validation_error}"
    )