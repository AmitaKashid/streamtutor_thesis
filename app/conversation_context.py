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
    parts.append('Answer the current user question directly, using earlier turns only as context. '
                 'Never claim that the user asked a different question or that a specific example was discussed '
                 'unless it appears in the visible history. If the user corrects a false memory, acknowledge the '
                 'mistake briefly and answer their request. If they ask for an example, work through a clearly '
                 'hypothetical example rather than inventing a previous one.')
    return '\n\n'.join(parts)
