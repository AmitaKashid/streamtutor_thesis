from ..providers.base import LLMProvider


def run_repl(provider: LLMProvider) -> None:
    print("Interactive LLM CLI. Type '/exit' to quit.")
    while True:
        try:
            user_text = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit"}:
            print("Bye.")
            return

        res = provider.generate(user_text)
        print("\nAnswer:\n" + res.text)
        print(
            f"\n[Latency] total={res.total_seconds:.3f}s | "
            f"prompt_tokens={res.prompt_tokens} | new_tokens={res.new_tokens} | "
            f"{res.tokens_per_second:.2f} tok/s"
        )
