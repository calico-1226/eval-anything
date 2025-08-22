from __future__ import annotations

import argparse
import os
import sys
from typing import List, Dict


# Ensure the package root (one level up from this scripts directory) is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(CURRENT_DIR)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from eval_anything.client.response_client import (  # noqa: E402
    ResponseClient,
    InferenceConfig,
)

# Optional rich console for colored output (mirrors safe_rlhf CLI style)
try:
    from rich.console import Console  # type: ignore
except Exception:  # pragma: no cover
    Console = None  # type: ignore

console = Console(soft_wrap=False, markup=False, emoji=False, highlight=False) if Console else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive CLI for ResponseClient")
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8001/v1",
        help="Base URL of the OpenAI-compatible API (e.g., http://localhost:8001/v1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key for the OpenAI-compatible API",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use (e.g., gpt-oss-20b)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help="Optional reasoning effort hint for the model",
    )
    return parser.parse_args()


HELP_MESSAGE = (
    "Commands:\n"
    "  /help   Show this help message\n"
    "  /clear  Clear the dialogue history\n"
    "  /exit   Exit the program\n"
)


def print_hint() -> None:
    hint = (
        "HINT: "
        "Type 'Ctrl + C' or 'Ctrl + D' to exit. "
        "Type '/clear' to clear dialogue history. "
        "Type '/help' to see help message."
    )
    if console:
        console.print(hint, style="bold yellow")
        console.print()
    else:
        print(hint)
        print()


def run_cli(client: ResponseClient) -> None:
    messages: List[Dict[str, str]] = []
    round_index = 0
    print_hint()
    try:
        while True:
            try:
                if console:
                    console.print(f"[{round_index + 1}] Human: ", style="bold green", end="")
                    user_input = console.input()
                else:
                    user_input = input(f"[{round_index + 1}] Human: ")
            except UnicodeDecodeError as ex:
                if console:
                    console.print("ERROR: ", style="bold red", end="")
                    console.print(f"Invalid input. UnicodeDecodeError: {ex}")
                    console.print("Please try again.")
                else:
                    print("ERROR: Invalid input. UnicodeDecodeError:", ex)
                    print("Please try again.")
                continue

            if not user_input:
                continue

            if user_input.strip() == "/help":
                if console:
                    console.print(HELP_MESSAGE)
                else:
                    print(HELP_MESSAGE)
                continue
            if user_input.strip() == "/clear":
                messages.clear()
                round_index = 0
                if console:
                    console.clear()
                else:
                    os.system("clear" if os.name != "nt" else "cls")
                print_hint()
                continue
            if user_input.strip() == "/exit":
                if console:
                    console.print("Bye!", style="bold yellow")
                else:
                    print("Bye!")
                break

            messages.append({"role": "user", "content": user_input})

            response = client.get_response(messages=messages)

            # Display
            if console:
                console.print(f"[{round_index + 1}] Assistant:", style="bold cyan")
                if response.reasoning_text:
                    console.print("- reasoning:", style="bold italic cyan")
                    console.print(response.reasoning_text, style="dim cyan")
                console.print("- output:", style="bold italic cyan")
                console.print(response.output_text, style="cyan", soft_wrap=True)
                console.print()
            else:
                print(f"[{round_index + 1}] Assistant:")
                if response.reasoning_text:
                    print("- reasoning:")
                    print(response.reasoning_text)
                print("- output:")
                print(response.output_text)
                print()

            # Append assistant message to the history and advance round
            messages.append({"role": "assistant", "content": response.output_text})
            round_index += 1

    except (KeyboardInterrupt, EOFError):
        if console:
            console.print()  # newline after ^C/^D
            console.print("Bye!", style="bold yellow")
        else:
            print()
            print("Bye!")


def main() -> None:
    args = parse_args()
    client = ResponseClient(
        base_url=args.api_base,
        api_key=args.api_key,
        inference_config=InferenceConfig(
            model_name=args.model,
            reasoning_effort=args.reasoning_effort,
        ),
    )
    run_cli(client)


if __name__ == "__main__":
    main()
