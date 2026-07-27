from .registry import register, ModelSpec


register(
    "qwen2_5",
    ModelSpec(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        use_chat_template=True,
        trust_remote_code=False,
        system_prompt="You are a helpful assistant.",
    ),
)
