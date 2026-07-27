from .registry import register, ModelSpec


register(
    "llama3",
    ModelSpec(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        use_chat_template=True,
        trust_remote_code=False,
        system_prompt="You are a helpful assistant.",
    ),
)
