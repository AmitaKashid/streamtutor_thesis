"""Build the current question with an honest conversation-history boundary."""


def contextual_user_query(question: str, history: str, *, earlier_omitted: bool = False) -> str:
    question = (question or '').strip()
    if not history and not earlier_omitted:
        return question
    parts = []
    if earlier_omitted:
        parts.append('The conversation history below is incomplete: earlier messages may be missing. '
                     'Any turn numbers refer only to the excerpt, not the start of the conversation. '
                     'If the question requires missing details, say you cannot see them and ask the student to repeat them. '
                     'Do not claim a detail was never provided just because it is absent from this excerpt.')
    if history:
        parts.append('Previous conversation between the user and assistant:\n' + history)
    parts.append('Current user question:\n' + question)
    parts.append('Use the previous conversation as context. Answer the current user question directly.')
    return '\n\n'.join(parts)
