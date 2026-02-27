"""5 тестов от простого к сложному — v4 (новые утверждения).

Все промпты уникальные, не дублируют v3.
Градация: простой факт → простой фейк → средний (числа+подмена) → сложный → очень сложный.
"""
import time, sys, os

TESTS = [
    {
        "claim": "Земля вращается вокруг Солнца",
        "expected": "ПРАВДА",
        "difficulty": "1. Простой факт",
        "comment": "Базовый астрономический факт — гелиоцентрическая модель",
    },
    {
        "claim": "Эйфелева башня находится в Берлине",
        "expected": "ЛОЖЬ",
        "difficulty": "2. Простой фейк",
        "comment": "Эйфелева башня в Париже, а не в Берлине",
    },
    {
        "claim": "Население Японии составляет примерно 2 миллиарда человек",
        "expected": "ЛОЖЬ",
        "difficulty": "3. Средний: числовой фейк",
        "comment": "Население Японии ~125 млн, не 2 млрд",
    },
    {
        "claim": "Первый человек в космосе — Нил Армстронг, совершивший полёт 12 апреля 1961 года",
        "expected": "ЛОЖЬ",
        "difficulty": "4. Сложный: подмена персоны с верной датой",
        "comment": "Дата верная, но первым был Юрий Гагарин, а не Нил Армстронг",
    },
    {
        "claim": "Великая Китайская стена видна из космоса невооружённым глазом и её длина составляет более 20 000 км",
        "expected": "ЧАСТИЧНО",
        "difficulty": "5. Очень сложный: миф + факт",
        "comment": "Длина ~21 196 км — правда, но видимость из космоса — популярный миф, опровергнутый космонавтами",
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
    total_start = time.time()

    for i, test in enumerate(TESTS, 1):
        print(f"\n{'='*80}")
        print(f"ТЕСТ {i}/{len(TESTS)} [{test['difficulty']}]")
        print(f"Утверждение: {test['claim']}")
        print(f"Ожидаемый вердикт: {test['expected']}")
        print(f"Комментарий: {test['comment']}")
        print("-" * 60)

        start = time.time()
        result = pipe.check(test["claim"])
        elapsed = time.time() - start

        verdict = result.get("verdict", "???")
        score = result.get("credibility_score", -1)

        # Гибкое сравнение
        expected_upper = test["expected"].upper()
        verdict_upper = verdict.upper()
        if expected_upper == "ЧАСТИЧНО":
            passed = verdict_upper in ("ЛОЖЬ", "МАНИПУЛЯЦИЯ", "МАНИПУЛЯЦИЯ / ПОЛУПРАВДА")
        elif expected_upper == "ПРАВДА":
            passed = "ПРАВДА" in verdict_upper
        elif expected_upper == "ЛОЖЬ":
            passed = "ЛОЖЬ" in verdict_upper
        else:
            passed = expected_upper in verdict_upper

        status = "PASS" if passed else "FAIL"
        results.append({
            "test": test, "verdict": verdict, "score": score,
            "time": elapsed, "passed": passed,
        })

        reasoning = result.get("reasoning", "")
        if reasoning:
            reasoning = reasoning[:120].replace("\n", " ") + "..."

        print(f"\nВердикт:    {verdict}")
        print(f"Score:      {score}")
        print(f"Время:      {elapsed:.1f}с")
        print(f"Результат:  [{status}] (ожидали '{test['expected']}', получили '{verdict}')")
        if reasoning:
            print(f"Reasoning:  {reasoning}")

    total_time = time.time() - total_start
    passed_count = sum(1 for r in results if r["passed"])

    print(f"\n\n{'='*80}")
    print(f"ИТОГОВАЯ ТАБЛИЦА — test_5levels_v4")
    print(f"{'='*80}")
    print(f"{'#':<5}{'Сложность':<35}{'Ожид.':<15}{'Получ.':<20}{'Score':<8}{'Время':<9}{'Статус'}")
    print("-" * 95)
    for i, r in enumerate(results, 1):
        t = r["test"]
        s = "PASS" if r["passed"] else "FAIL"
        print(f"{i:<5}{t['difficulty']:<35}{t['expected']:<15}{r['verdict']:<20}"
              f"{r['score']:<8}{r['time']:<9.1f}{s}")
    print("-" * 95)
    print(f"Итого: {passed_count}/{len(TESTS)} тестов пройдено ({100*passed_count//len(TESTS)}%)")
    print(f"Общее время: {total_time:.1f}с (среднее: {total_time/len(TESTS):.1f}с на тест)")


if __name__ == "__main__":
    main()
