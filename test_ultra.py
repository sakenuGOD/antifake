"""Ultra-hard 15-claim test — scam schemes, conspiracies, adversarial cases.

Категории:
  simple_true/false  — базовые факты
  scam               — мошеннические схемы (нигерийский принц, лотерея, CVV)
  scam_composite     — составное + мошенничество (МММ)
  date_swap          — подмена даты
  person_swap        — подмена автора
  numerical_false    — ложное число
  myths              — научно-опровергнутые мифы
  composite          — два суб-утверждения (одно правда, одно ложь)
  conspiracy         — конспирология
  adversarial_true   — выглядит как фейк, но правда
"""
import time

TEST_CLAIMS = [
    # ===== EASY (d=1) — 3 утверждения =====

    {"claim": "Нью-Йорк является самым населённым городом США",
     "expected": "ПРАВДА", "diff": 1, "cat": "simple_true"},

    {"claim": "Австралия расположена в Северном полушарии",
     "expected": "ЛОЖЬ", "diff": 1, "cat": "simple_false"},

    # Классическая нигерийская схема — авансовый платёж
    {"claim": "Нигерийский принц обещает выплатить 5 миллионов долларов "
              "если вы поможете перевести наследство и заплатите небольшой авансовый взнос",
     "expected": "ЛОЖЬ", "diff": 1, "cat": "scam"},

    # ===== MEDIUM (d=2) — 4 утверждения =====

    {"claim": "Юрий Гагарин совершил первый полёт в космос в 1965 году",
     "expected": "ЛОЖЬ", "diff": 2, "cat": "date_swap"},

    {"claim": "Теорию специальной относительности разработал Никола Тесла",
     "expected": "ЛОЖЬ", "diff": 2, "cat": "person_swap"},

    # Лотерейная схема — оплати налог на выигрыш
    {"claim": "Вам сообщают что вы выиграли в лотерею миллион рублей "
              "и для получения выигрыша необходимо сначала оплатить налог в размере 15 тысяч рублей",
     "expected": "ЛОЖЬ", "diff": 2, "cat": "scam"},

    {"claim": "Расстояние от Земли до Луны составляет около 3840 километров",
     "expected": "ЛОЖЬ", "diff": 2, "cat": "numerical_false"},

    # ===== HARD (d=3) — 4 утверждения =====

    # Apple 1976 = ПРАВДА; первый iPhone в 2010 = ЛОЖЬ (2007)
    {"claim": "Стив Джобс основал компанию Apple в 1976 году "
              "и представил первый iPhone в 2010 году",
     "expected": "СОСТАВНОЕ", "diff": 3, "cat": "composite",
     "expected_subs": ["ПРАВДА", "ЛОЖЬ"]},

    # Наполеон: миф о 157 см (на самом деле ~169 см — средний рост для той эпохи)
    {"claim": "Наполеон Бонапарт отличался очень низким ростом — около 157 сантиметров",
     "expected": "ЛОЖЬ", "diff": 3, "cat": "myths"},

    # Понци-схема: гарантированная доходность = красный флаг
    {"claim": "Инвестиционная компания гарантирует доходность 50 процентов в месяц "
              "и полную защиту от потери вложенных средств",
     "expected": "ЛОЖЬ", "diff": 3, "cat": "scam"},

    # Миф о 10% мозга — хорошо опровергнут нейронаукой
    {"claim": "Учёные доказали что человек использует только 10 процентов возможностей своего мозга",
     "expected": "ЛОЖЬ", "diff": 3, "cat": "myths"},

    # ===== VERY HARD (d=4) — 4 утверждения =====

    # Вишинг (банковский фрод): настоящий банк НИКОГДА не просит CVV/SMS-код
    {"claim": "Звонок от сотрудника банка с просьбой сообщить CVV код карты "
              "и одноразовый SMS-пароль является законной процедурой защиты счёта от мошенников",
     "expected": "ЛОЖЬ", "diff": 4, "cat": "scam"},

    # МММ: была пирамидой = ПРАВДА; никогда не платила = ЛОЖЬ (ранним вкладчикам платили)
    {"claim": "МММ Сергея Мавроди была финансовой пирамидой "
              "и никогда не выплачивала денег своим вкладчикам",
     "expected": "СОСТАВНОЕ", "diff": 4, "cat": "scam_composite",
     "expected_subs": ["ПРАВДА", "ЛОЖЬ"]},

    # Конспирология: вакцины + микрочипы Билла Гейтса
    {"claim": "Вакцины против COVID-19 содержат микрочипы Билла Гейтса "
              "для слежки за людьми через сеть 5G",
     "expected": "ЛОЖЬ", "diff": 4, "cat": "conspiracy"},

    # Adversarial true: выглядит как миф, но подтверждается авиастатистикой
    {"claim": "Большинство смертельных авиакатастроф происходит во время взлёта или посадки "
              "а не в крейсерском полёте",
     "expected": "ПРАВДА", "diff": 4, "cat": "adversarial_true"},
]

N = len(TEST_CLAIMS)

VERDICT_NORMALIZE = {
    "СОСТАВНОЕ": "СОСТАВНОЕ",
    "СОСТАВНОЕ УТВЕРЖДЕНИЕ": "СОСТАВНОЕ",
    "НЕ УВЕРЕНА": "НЕ УВЕРЕНА",
    "ЛОЖЬ": "ЛОЖЬ", "FALSE": "ЛОЖЬ", "ФЕЙК": "ЛОЖЬ",
    "ПРАВДА": "ПРАВДА", "TRUE": "ПРАВДА",
    "МАНИПУЛЯЦИЯ": "СОСТАВНОЕ",
    "ПОЛУПРАВДА": "СОСТАВНОЕ",
    "ЧАСТИЧНО ПОДТВЕРЖДЕНО": "СОСТАВНОЕ",
    "НЕ ПОДТВЕРЖДЕНО": "НЕ УВЕРЕНА",
}


def norm(v):
    v = v.strip().upper()
    for key, val in VERDICT_NORMALIZE.items():
        if key in v:
            return val
    return "НЕ УВЕРЕНА"


def check_composite_subs(result, expected_subs):
    sub_verdicts = result.get("sub_verdicts", [])
    if not sub_verdicts or not expected_subs:
        return False
    actual = sorted(sv.get("status", "").upper() for sv in sub_verdicts[:len(expected_subs)])
    return actual == sorted(expected_subs)


def main():
    from pipeline import FactCheckPipeline
    from config import SearchConfig
    from model import find_best_adapter

    adapter_path = find_best_adapter()
    print(f"Adapter: {adapter_path or 'base'}")
    pipeline = FactCheckPipeline(adapter_path=adapter_path, search_config=SearchConfig())

    results = []
    total_time = 0

    for i, tc in enumerate(TEST_CLAIMS):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{N}] (d={tc['diff']}, {tc['cat']}) {tc['claim'][:80]}")
        print(f"Expected: {tc['expected']}")
        if tc.get("expected_subs"):
            print(f"Expected subs: {tc['expected_subs']}")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            result = pipeline.check(tc["claim"])
            elapsed = time.time() - t0
            total_time += elapsed

            predicted = norm(result.get("verdict", ""))
            score = result.get("credibility_score", 50)

            if tc["expected"] == "СОСТАВНОЕ" and tc.get("expected_subs"):
                correct = predicted == "СОСТАВНОЕ" and check_composite_subs(result, tc["expected_subs"])
                subs_actual = [sv.get("status", "?") for sv in result.get("sub_verdicts", [])]
                print(f"  Sub-verdicts: {subs_actual} (expected: {tc['expected_subs']})")
            else:
                correct = predicted == tc["expected"]

            status = "OK" if correct else "MISS"
            print(f"\n>>> RESULT: {predicted} (score={score}) [{status}] ({elapsed:.1f}s)")

            if predicted == "НЕ УВЕРЕНА" and result.get("_explanation"):
                print(f"    Explanation: {result['_explanation'][:120]}")

            results.append({
                "claim": tc["claim"][:52],
                "expected": tc["expected"],
                "predicted": predicted,
                "score": score,
                "correct": correct,
                "time": round(elapsed, 1),
                "diff": tc["diff"],
                "cat": tc["cat"],
            })
        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            print(f"\n>>> ERROR: {e} ({elapsed:.1f}s)")
            results.append({
                "claim": tc["claim"][:52],
                "expected": tc["expected"],
                "predicted": "ERROR",
                "score": 0,
                "correct": False,
                "time": round(elapsed, 1),
                "diff": tc["diff"],
                "cat": tc["cat"],
            })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"SUMMARY: ultra-hard {N}-claim test (scam + composite + adversarial)")
    print(f"{'='*70}")
    correct_count = sum(1 for r in results if r["correct"])
    print(f"Accuracy: {correct_count}/{N} ({correct_count/N:.0%})")
    print(f"Avg time: {total_time/N:.1f}s | Total: {total_time:.0f}s")

    print(f"\n{'Claim':<54s} {'Exp':<15s} {'Got':<15s} {'Score':>5s} {'Time':>6s} {'':>4s}")
    print("-" * 102)
    for r in results:
        ok = "OK" if r["correct"] else "MISS"
        print(f"{r['claim']:<54s} {r['expected']:<15s} {r['predicted']:<15s} "
              f"{r['score']:>5d} {r['time']:>5.1f}s {ok:>4s}")

    print(f"\nPer difficulty:")
    for d in sorted(set(r["diff"] for r in results)):
        dr = [r for r in results if r["diff"] == d]
        print(f"  d={d}: {sum(r['correct'] for r in dr)}/{len(dr)}")

    print(f"\nPer category:")
    for cat in sorted(set(r["cat"] for r in results)):
        cr = [r for r in results if r["cat"] == cat]
        print(f"  {cat}: {sum(r['correct'] for r in cr)}/{len(cr)}")

    critical = sum(1 for r in results
                   if (r["expected"] == "ПРАВДА" and r["predicted"] == "ЛОЖЬ")
                   or (r["expected"] == "ЛОЖЬ" and r["predicted"] == "ПРАВДА"))
    print(f"\nCritical errors (ПРАВДА<->ЛОЖЬ): {critical}")


if __name__ == "__main__":
    main()
