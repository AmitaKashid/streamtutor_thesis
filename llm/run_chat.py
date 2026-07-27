from llmcli.common.chat_loop import run_repl
from llmcli.config import parse_args
from llmcli.models.registry import get_model_spec

from llmcli.providers.llamacpp import LlamaCppProvider
from llmcli.providers.hf_transformers import HFTransformersProvider

def main() -> None:
    args = parse_args()
    spec = get_model_spec(args.model)

    if args.backend == "llamacpp" or spec.backend == "llamacpp":
        provider = LlamaCppProvider(
            model_path=spec.gguf_path,
            chat_format=spec.chat_format,
            system_prompt=spec.system_prompt,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=-1,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
    else:
        provider = HFTransformersProvider(
            model_id=spec.model_id,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=spec.trust_remote_code,
            use_chat_template=spec.use_chat_template,
            system_prompt=spec.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )

    run_repl(provider)

if __name__ == "__main__":
    main()
