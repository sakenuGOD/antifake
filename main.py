"""CLI точка входа для Fact-Checker Pipeline."""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Fact-Checker: проверка достоверности утверждений",
    )
    parser.add_argument(
        "claim",
        nargs="?",
        help="Утверждение для проверки",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Интерактивный режим",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="adapters/fact_checker_lora",
        help="Путь к LoRA адаптерам (по умолчанию: adapters/fact_checker_lora)",
    )
    args = parser.parse_args()

    if not args.claim and not args.interactive:
        parser.print_help()
        sys.exit(1)

    # Проверка API-ключа
    if not os.environ.get("SERPAPI_API_KEY"):
        print("Ошибка: переменная окружения SERPAPI_API_KEY не установлена.")
        print("Получите ключ на https://serpapi.com/ и установите:")
        print("  export SERPAPI_API_KEY='ваш_ключ'")
        sys.exit(1)

    # Импортируем здесь, чтобы быстро показать ошибки CLI без загрузки модели
    from pipeline import FactCheckPipeline
    from config import SearchConfig

    search_config = SearchConfig(api_key=os.environ["SERPAPI_API_KEY"])

    adapter_path = args.adapter_path if os.path.exists(args.adapter_path) else None
    pipeline = FactCheckPipeline(
        adapter_path=adapter_path,
        search_config=search_config,
    )

    if args.interactive:
        print("\n=== Fact-Checker: Интерактивный режим ===")
        print("Введите утверждение для проверки (или 'выход' для завершения)\n")
        while True:
            try:
                claim = input("Утверждение: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nЗавершение работы.")
                break

            if not claim:
                continue
            if claim.lower() in ("выход", "exit", "quit", "q"):
                print("Завершение работы.")
                break

            result = pipeline.check(claim)
            print(f"\n{result['verdict']}\n")
    else:
        result = pipeline.check(args.claim)
        print(f"\n{result['verdict']}")


if __name__ == "__main__":
    main()
