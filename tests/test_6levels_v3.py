"""6 новых тестов от очень простого к очень сложному — v3 (после обновлений).

Все промпты уникальные, не дублируют v1/v2.
Градация: очень простой → простой → средний (числа) → средний (подмена) → сложный → очень сложный.
"""
import _path  # noqa: F401,E402 — inject project root into sys.path
import time, sys, os

TESTS = [
    {
        "claim": "Луна является естественным спутником Земли",
        "expected": "ПРАВДА",
        "difficulty": "1. Очень простой",
        "comment": "Базовый астрономический факт, не требует поиска",
    },
    {
        "claim": "Амазонка — самая длинная река в мире, протекающая через Африку",
        "expected": "ЛОЖЬ",
        "difficulty": "2. Простой фейк",
        "comment": "Амазонка в Южной Америке, а не в Африке. Также спорно про длину (Нил vs Амазонка)",
    },
    {
        "claim": "Скорость звука в воздухе составляет примерно 343 м/с при 20°C",
        "expected": "ПРАВДА",
        "difficulty": "3. Средний: числовой факт",
        "comment": "Точное числовое значение — нужна проверка конкретной цифры 343 м/с",
    },
    {
        "claim": "Компания Microsoft была основана Стивом Джобсом в 1975 году",
        "expected": "ЛОЖЬ",
        "difficulty": "4. Средний: подмена персоны",
        "comment": "Год верный (1975), но основатели — Билл Гейтс и Пол Аллен, а не Стив Джобс",
    },
    {
        "claim": "Титаник затонул 15 апреля 1912 года после столкновения с айсбергом в Тихом океане",
        "expected": "ЛОЖЬ",
        "difficulty": "5. Сложный: тонкая подмена",
        "comment": "Дата верна, причина верна, но Титаник затонул в Атлантическом океане, не в Тихом",
    },
    {
        "claim": "Марс — четвёртая планета от Солнца с двумя спутниками Фобос и Деймос, а температура на его поверхности достигает +50°C летом",
        "expected": "ЧАСТИЧНО",
        "difficulty": "6. Очень сложный: составное",
        "comment": "Четвёртая планета — да, Фобос и Деймос — да, но макс. температура ~+20°C, не +50°C",
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
            v_upper = verdict.upper().strip()
            exp_upper = test["expected"].upper()

            if exp_upper == "ЧАСТИЧНО":
                # Составное утверждение: ожидаем НЕ ПОДТВЕРЖДЕНО или ЛОЖЬ (частично верно)
                match = any(x in v_upper for x in (
                    "ПОЛУПРАВДА", "ЧАСТИЧНО", "МАНИПУЛЯЦИЯ",
                    "НЕ ПОДТВЕРЖДЕНО", "ЛОЖЬ",
                ))
            elif exp_upper == "ЛОЖЬ":
                match = any(x in v_upper for x in ("ЛОЖЬ", "ФЕЙК", "FALSE"))
            elif exp_upper == "ПРАВДА":
                match = any(x in v_upper for x in ("ПРАВДА", "TRUE"))
            else:
                match = exp_upper in v_upper or v_upper in exp_upper

            status = "PASS" if match else "FAIL"

            print(f"\nВердикт:    {verdict}")
            print(f"Score:      {score}")
            print(f"Время:      {dt:.1f}с")
            print(f"Результат:  [{status}] (ожидали '{test['expected']}', получили '{verdict}')")

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
    print("ИТОГОВАЯ ТАБЛИЦА — test_6levels_v3")
    print("=" * 80)
    print(f"{'#':<4} {'Сложность':<32} {'Ожид.':<14} {'Получ.':<20} {'Score':<7} {'Время':<8} {'Статус'}")
    print("-" * 95)

    passed = 0
    for r in results:
        status = "PASS" if r["match"] else "FAIL"
        if r["match"]:
            passed += 1
        print(f"{r['test']:<4} {r['difficulty']:<32} {r['expected']:<14} {str(r['got']):<20} {str(r['score']):<7} {str(r['time'])+'с':<8} {status}")

    print("-" * 95)
    print(f"Итого: {passed}/{len(results)} тестов пройдено ({passed/len(results)*100:.0f}%)")
    print(f"Общее время: {total_time:.1f}с (среднее: {total_time/len(results):.1f}с на тест)")


if __name__ == "__main__":
    main()
