"""第1回: LLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask
    uv run python -m app.ask --chat

通常モードでは、会話履歴を使わずに1回だけ質問する。
質問を省略すると、講座に固有の質問をする。

--chatを付けると、会話履歴を保って続けて質問できる。
会話モードは exit または quit で終了する。
履歴は実行中のみ保持し、終了後には保存しない。
用語集の検索や外部ツールの利用は行わない。
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def show_usage(response) -> None:
    """入力・出力トークン数を表示する。"""
    usage = response.usage_metadata

    if usage is not None:
        print(f"入力トークン数: {usage.prompt_token_count}")
        print(f"出力トークン数: {usage.candidates_token_count}")
    else:
        print("トークン数の情報を取得できませんでした。")


def ask(question: str) -> str:
    """会話履歴を使わず、1回だけ質問する。"""
    client = genai.Client(api_key=get_api_key())

    start = time.perf_counter()

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=question,
        )
    finally:
        elapsed = time.perf_counter() - start
        print(f"API呼び出し時間: {elapsed:.2f}秒")

    show_usage(response)
    return response.text or ""


def chat_loop() -> None:
    """同じチャットを使い、会話履歴を保って質問を続ける。"""
    client = genai.Client(api_key=get_api_key())

    # 繰り返しの外で1回だけ作り、同じ会話を使い続ける
    chat = client.chats.create(model=MODEL)

    print("会話モードです。終了するには exit または quit と入力してください。")

    while True:
        question = input("\nあなた> ").strip()

        # 終了の指示があれば、繰り返しを抜ける
        if question.lower() in ("exit", "quit"):
            print("会話を終了します。")
            break

        # 空の質問は送らず、再び入力を待つ
        if not question:
            continue

        # これまでの会話履歴を使って質問する
        start = time.perf_counter()

        try:
            response = chat.send_message(question)
        finally:
            elapsed = time.perf_counter() - start
            print(f"API呼び出し時間: {elapsed:.2f}秒")

        show_usage(response)
        print(f"\nAI> {response.text or '本文が返りませんでした。'}")

def main() -> None:
    parser = argparse.ArgumentParser(description="LLMに質問する")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="会話履歴を保って続けて質問する",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")

    try:
        if args.chat:
            chat_loop()
        else:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))

    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)

    except (KeyboardInterrupt, EOFError):
        print("\n終了しました。")


if __name__ == "__main__":
    main()
