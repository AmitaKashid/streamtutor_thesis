from .registry import register, ModelSpec


register(
    "generic",
    ModelSpec(
        model_id="gpt2",  # placeholder; replace with anything
        use_chat_template=False,
        trust_remote_code=False,
        system_prompt=None,
    ),
)
