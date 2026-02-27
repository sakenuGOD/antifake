"""6 новых тестов от очень простого к очень сложному — v2."""
import time, sys, os

# 6 утверждений по возрастающей сложности (новые, не дублируют test_6levels.py):
# 1. Очень простой: общеизвестный научный факт
# 2. Простой фейк: распространённое заблуждение
# 3. Средний: исторический факт с датой
# 4. Средний: манипуляция (популярный миф)
# 5. Сложный: тонкая подмена одной детали
# 6. Очень сложный: составное утверждение (часть правда, часть ложь)

TESTS = [
    {
        "claim": "Вода кипит при 100 градусах Цельсия при нормальном атмосферном давлении",
        "expected": "ПРАВДА",
        "difficulty": "1. Очень простой",
        "comment": "Базовый физический факт, подтверждается любым источником",
    },
    {
        "claim": "Столица Австралии — Сидней",
        "expected": "ЛОЖЬ",
        "difficulty": "2. Простой фейк",
        "comment": "Распространённое заблуждение — столица Канберра, а не Сидней",
    },
    {
        "claim": "Берлинская стена пала 9 ноября 1989 года",
        "expected": "ПРАВДА",
        "difficulty": "3. Средний: дата",
        "comment": "Исторический факт с конкретной датой, нужна точная проверка",
    },
    {
        "claim": "Альберт Эйнштейн провалил экзамен по математике в школе",
        "expected": "ЛОЖЬ",
        "difficulty": "4. Средний: миф",
        "comment": "Популярный миф — Эйнштейн отлично успевал по математике",
    },
    {
        "claim": "Первый iPhone был представлен Стивом Джобсом в 2008 году",
        "expected": "ЛОЖЬ",
        "difficulty": "5. Сложный: подмена года",
        "comment": "iPhone представлен в январе 2007, не 2008. Тонкая подмена на 1 год",
    },
    {
        "claim": "Байкал — самое глубокое озеро в мире с максимальной глубиной 1642 метра, расположенное в Казахстане",
        "expected": "ЧАСТИЧНО",
        "difficulty": "6. Очень сложный: составное",
        "comment": "Глубина и рекорд верны, но Байкал в России (Сибирь), а не в Казахстане",
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
            elif exp_upper == "ЛОЖЬ":
                match = any(x in v_upper for x in ("ЛОЖЬ", "ФЕЙК", "FALSE"))
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
    print("ИТОГОВАЯ ТАБЛИЦА — test_6levels_v2")
    print("=" * 80)
    print(f"{'#':<4} {'Сложность':<30} {'Ожид.':<14} {'Получ.':<20} {'Score':<7} {'Время':<7} {'Статус'}")
    print("-" * 92)

    passed = 0
    for r in results:
        status = "PASS" if r["match"] else "FAIL"
        if r["match"]:
            passed += 1
        print(f"{r['test']:<4} {r['difficulty']:<30} {r['expected']:<14} {str(r['got']):<20} {str(r['score']):<7} {r['time']:<7} {status}")

    print("-" * 92)
    print(f"Итого: {passed}/{len(results)} тестов пройдено ({passed/len(results)*100:.0f}%)")
    print(f"Общее время: {total_time:.1f}с (среднее: {total_time/len(results):.1f}с на тест)")


if __name__ == "__main__":
    main()
