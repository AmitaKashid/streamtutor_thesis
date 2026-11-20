"""Conservatively add the student's prior topic to an elliptical RAG query.

This inexpensive heuristic does not invent a rewritten question or use model
answers as evidence. Retrieval quality must still be evaluated on the corpus.
"""
import re

_TOKENS = re.compile(r"[\w]+(?:['’][\w]+)?", re.UNICODE)
_REFERENTIAL = {
    'it', 'its', 'this', 'that', 'these', 'those', 'they', 'them', 'their',
    'same', 'previous', 'earlier', 'again', 'another', 'more', 'example',
}
_GENERAL = _REFERENTIAL | {
    'a', 'an', 'the', 'and', 'or', 'as', 'at', 'by', 'for', 'from', 'in',
    'into', 'of', 'on', 'to', 'with', 'about', 'is', 'are', 'was', 'were',
    'do', 'does', 'did', 'be', 'been', 'being', 'can', 'could', 'would',
    'will', 'should', 'may', 'might', 'i', 'me', 'my', 'we', 'us', 'our',
    'you', 'your', 'what', 'which', 'who', 'how', 'why', 'when', 'where',
    'please', 'explain', 'explanation', 'describe', 'show', 'tell', 'give',
    'provide', 'walk', 'through', 'understand', 'mean', 'means', 'work',
    'works', 'working', 'use', 'using', 'used', 'one', 'some', 'any', 'very',
    'simple', 'simpler', 'step', 'steps', 'detail', 'details', 'case',
    'discuss', 'discussed', 'not', 'but', 'so', 'then', 'also', 'there',
}


def _words(text):
    return [match.group().casefold() for match in _TOKENS.finditer(text)]


def followup_retrieval_query(question, previous_turns, *, limit=320):
    """Add one recent explicit student topic only to a topic-free follow-up."""
    question = (question or '').strip()
    words = _words(question)
    if not question or not (_REFERENTIAL.intersection(words)):
        return question
    if any(word not in _GENERAL and not word.isdecimal() for word in words):
        return question  # It names its own topic; do not mix in an older one.
    for turn in reversed(previous_turns or []):
        previous = str(turn.get('user') or '').strip()
        if not previous or previous.casefold() == question.casefold():
            continue
        if any(word not in _GENERAL and not word.isdecimal() for word in _words(previous)):
            return question + '\n' + previous[:limit]
    return question
