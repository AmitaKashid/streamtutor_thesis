from __future__ import annotations


HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"
HARMONY_ANALYSIS_MARKER = "<|channel|>analysis<|message|>"
HARMONY_FINAL_TRANSITION = "<|end|><|start|>assistant"
HARMONY_TERMINATORS = ("<|return|>", "<|end|>")
HARMONY_CONTROL_TOKENS = (
    "<|channel|>",
    "<|message|>",
    "<|start|>",
    "<|end|>",
    "<|return|>",
)


class ResponseProtocolError(ValueError):
    """Raised when a model response cannot be decoded unambiguously."""


def decode_harmony_final(raw_text: str) -> str:
    """Extract one final-channel payload from a validated Harmony envelope."""
    text = raw_text.strip()
    marker_count = text.count(HARMONY_FINAL_MARKER)
    if marker_count != 1:
        raise ResponseProtocolError(
            "Harmony response must contain exactly one final-channel marker; "
            f"found {marker_count}."
        )

    prefix, payload = text.split(HARMONY_FINAL_MARKER, 1)
    prefix = prefix.strip()
    payload = payload.strip()

    # llama-cpp-python 0.3.20 may return either the final channel alone or
    # an analysis channel followed by the official assistant-final transition.
    if prefix:
        has_analysis = HARMONY_ANALYSIS_MARKER in prefix
        has_final_transition = prefix.endswith(HARMONY_FINAL_TRANSITION)
        if not (has_analysis and has_final_transition):
            raise ResponseProtocolError(
                "Harmony response contains an unrecognized prefix before "
                "the final channel."
            )

    for terminator in HARMONY_TERMINATORS:
        if payload.endswith(terminator):
            payload = payload[: -len(terminator)].rstrip()
            break

    if not payload:
        raise ResponseProtocolError("Harmony final-channel payload is empty.")

    residual = [token for token in HARMONY_CONTROL_TOKENS if token in payload]
    if residual:
        raise ResponseProtocolError(
            "Harmony final-channel payload contains residual control tokens: "
            + ", ".join(residual)
        )

    return payload


def decode_response_protocol(raw_text: str, protocol: str | None) -> str:
    if protocol is None:
        return raw_text.strip()
    if protocol == "harmony":
        return decode_harmony_final(raw_text)
    raise ResponseProtocolError(f"Unsupported response protocol: {protocol!r}")
