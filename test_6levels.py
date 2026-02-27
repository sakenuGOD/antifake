"""6 тестов от простого к сложному — проверка пайплайна после рефакторинга."""
import time, sys, os

# 6 утверждений по возрастающей сложности:
# 1. Простой факт (общеизвестный, легко проверяемый)
# 2. Простой фейк (очевидная подмена)
# 3. Средний: числовой факт (нужна точная проверка чисел)
# 4. Средний: манипуляция (полуправда — частично верно)
# 5. Сложный: тонкая подмена (одна деталь изменена)
# 6. Очень сложный: составное утверждение (часть правда, часть ложь)

TESTS = [
    {
        "claim": "Земля вращается вокруг Солнца",
        "expected": "ПРАВДА",
        "difficulty": "Простой факт",
        "comment": "Базовый научный факт, должен подтверждаться всеми источниками",
    },
    {
        "claim": "Эйфелева башня находится в Берлине",
        "expected": "ЛОЖЬ",
        "difficulty": "Простой фейк",
        "comment": "Очевидная подмена города — Париж, а не Берлин",
    },
    {
        "claim": "Население Японии составляет примерно 125 миллионов человек",
        "expected": "ПРАВДА",
        "difficulty": "Средний: числовой",
        "comment": "Нужна проверка числа — ~125 млн (2023-2024)",
    },
    {
        "claim": "Илон Маск основал компанию Tesla",
        "expected": "МАНИПУЛЯЦИЯ",
        "difficulty": "Средний: манипуляция",
        "comment": "Tesla основали Тарпеннинг и Эберхард, Маск — ранний инвестор и позже CEO",
    },
    {
        "claim": "Олимпийские игры 2024 года прошли в Токио",
        "expected": "ЛОЖЬ",
        "difficulty": "Сложный: тонкая подмена",
        "comment": "ОИ-2024 в Париже, а Токио — это ОИ-2020/2021. Тонкая подмена года/города",
    },
    {
        "claim": "Юрий Гагарин — первый человек в космосе, он совершил полёт на корабле Восток-2 12 апреля 1961 года",
        "expected": "ЧАСТИЧНО",
        "difficulty": "Очень сложный: составное",
        "comment": "Гагарин — да, 12 апреля 1961 — да, но корабль Восток-1, а не Восток-2",
    },
]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ".")

    from pipeline import FactCheckPipeline
    from model import find_best_adapter

    adapter_path = find_best_adapter()
    print(f"Адаптер: {adapter_path or 'base model (без адаптеров)'}")
    print("=" * 80)

    pipe = FactCheckPipeline(adapter_path=adapter_path)

    results = []
    total_time = 0

    for i, test in enumerate(TESTS):
        print(f"\n{'='*80}")
        print(f"ТЕСТ {i+1}/6 [{test['difficulty']}]")
        print(f"Утверждение: {test['claim']}")
        print(f"Ожидаемый вердикт: {test['expected']}")
        print(f"Комментарий: {test['comment']}")
        print("-" * 60)

        t0 = time.time()
        try:
            result = pipe.check(test["claim"])
            dt = time.time() - t0
            total_time += dt

            verdict = result.get("verdict", "N/A")
            score = result.get("credibility_score", "N/A")
            reasoning = result.get("reasoning", "")

            # Определяем совпадение (нестрогое)
            v_upper = verdict.upper()
            exp_upper = test["expected"].upper()
            if exp_upper == "ЧАСТИЧНО":
                match = any(x in v_upper for x in ("ПОЛУПРАВДА", "ЧАСТИЧНО",
                                                     "МАНИПУЛЯЦИЯ"))
            elif exp_upper == "МАНИПУЛЯЦИЯ":
                match = any(x in v_upper for x in ("МАНИПУЛЯЦИЯ", "ПОЛУПРАВДА",
                                                     "ЧАСТИЧНО"))
            else:
                match = exp_upper in v_upper or v_upper in exp_upper

            status = "PASS" if match else "FAIL"

            print(f"\nВердикт:    {verdict}")
            print(f"Score:      {score}")
            print(f"Время:      {dt:.1f}с")
            print(f"Результат:  [{status}] (ожидали '{test['expected']}', получили '{verdict}')")

            # Краткое reasoning (первые 300 символов)
            if reasoning:
                short = reasoning[:300].replace("\n", " ")
                print(f"Reasoning:  {short}...")

            results.append({
                "test": i + 1,
                "difficulty": test["difficulty"],
                "claim": test["claim"],
                "expected": test["expected"],
                "got": verdict,
                "score": score,
                "match": match,
                "time": round(dt, 1),
            })

        except Exception as e:
            dt = time.time() - t0
            total_time += dt
            print(f"\nОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "test": i + 1,
                "difficulty": test["difficulty"],
                "claim": test["claim"],
                "expected": test["expected"],
                "got": f"ERROR: {e}",
                "score": None,
                "match": False,
                "time": round(dt, 1),
            })

    # Итоговая таблица
    print(f"\n\n{'='*80}")
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 80)
    print(f"{'#':<4} {'Сложность':<28} {'Ожид.':<14} {'Получ.':<20} {'Score':<7} {'Время':<7} {'Статус'}")
    print("-" * 90)

    passed = 0
    for r in results:
        status = "PASS" if r["match"] else "FAIL"
        if r["match"]:
            passed += 1
        print(f"{r['test']:<4} {r['difficulty']:<28} {r['expected']:<14} {str(r['got']):<20} {str(r['score']):<7} {r['time']:<7} {status}")

    print("-" * 90)
    print(f"Итого: {passed}/{len(results)} тестов пройдено ({passed/len(results)*100:.0f}%)")
    print(f"Общее время: {total_time:.1f}с (среднее: {total_time/len(results):.1f}с на тест)")


if __name__ == "__main__":
    main()
