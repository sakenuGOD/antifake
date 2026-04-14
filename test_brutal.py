"""Brutal 20-claim test — heavy scam, adversarial true, negation traps, composite.

Категории:
  simple_true/false     — базовые факты с подвохом
  scam                  — мошенничество (криптопирамида, фишинг, романтический скам)
  scam_composite        — составное + мошенничество
  adversarial_true      — выглядит как фейк, но правда
  adversarial_false     — выглядит как правда, но ложь
  negation_trap         — утверждение с "никогда/ни разу" (ловушка отрицания)
  date_swap             — подмена даты
  number_swap           — подмена числа
  person_swap           — подмена автора/субъекта
  myths                 — устойчивые мифы
  composite             — два суб-утверждения (одно правда, одно ложь)
  conspiracy            — конспирология
"""
import time

TEST_CLAIMS = [
    # ===== EASY (d=1) — 3 =====

    # 1. Простая правда — но формулировка "официально"
    {"claim": "Москва является столицей Российской Федерации",
     "expected": "ПРАВДА", "diff": 1, "cat": "simple_true"},

    # 2. Простая ложь — географическая подмена
    {"claim": "Река Амазонка протекает через территорию Африки",
     "expected": "ЛОЖЬ", "diff": 1, "cat": "simple_false"},

    # 3. Фишинг-скам: Госуслуги
    {"claim": "Сотрудники портала Госуслуги звонят гражданам и просят продиктовать "
              "код из СМС для подтверждения личности и защиты аккаунта",
     "expected": "ЛОЖЬ", "diff": 1, "cat": "scam"},

    # ===== MEDIUM (d=2) — 5 =====

    # 4. Число — подмена порядка величины
    {"claim": "Население Китая составляет около 140 миллионов человек",
     "expected": "ЛОЖЬ", "diff": 2, "cat": "number_swap"},

    # 5. Автор — подмена (Достоевский vs Толстой)
    {"claim": "Фёдор Достоевский написал роман Анна Каренина",
     "expected": "ЛОЖЬ", "diff": 2, "cat": "person_swap"},

    # 6. Криптовалютный скам — гарантированная доходность
    {"claim": "Криптовалютный бот с искусственным интеллектом гарантирует ежедневный "
              "доход 3 процента от вложений без каких-либо рисков",
     "expected": "ЛОЖЬ", "diff": 2, "cat": "scam"},

    # 7. Adversarial TRUE — звучит невероятно, но факт
    {"claim": "Клеопатра жила ближе по времени к запуску iPhone чем к строительству пирамид в Гизе",
     "expected": "ПРАВДА", "diff": 2, "cat": "adversarial_true"},

    # 8. Дата — подмена (Берлинская стена 1991 vs 1989)
    {"claim": "Берлинская стена была разрушена в 1991 году",
     "expected": "ЛОЖЬ", "diff": 2, "cat": "date_swap"},

    # ===== HARD (d=3) — 6 =====

    # 9. Составное: Менделеев + таблица (правда) + водка (ложь)
    {"claim": "Менделеев создал периодическую таблицу химических элементов и изобрёл водку",
     "expected": "СОСТАВНОЕ", "diff": 3, "cat": "composite",
     "expected_subs": ["ПРАВДА", "ЛОЖЬ"]},

    # 10. Романтический скам — солдат просит деньги
    {"claim": "Американский военный познакомился с вами в интернете и просит перевести "
              "ему 50 тысяч рублей на лечение потому что армия не оплачивает медицинские расходы",
     "expected": "ЛОЖЬ", "diff": 3, "cat": "scam"},

    # 11. Adversarial TRUE — мёд не портится
    {"claim": "Мёд является продуктом который практически не портится и может храниться тысячелетиями",
     "expected": "ПРАВДА", "diff": 3, "cat": "adversarial_true"},

    # 12. Миф — Великая Китайская стена
    {"claim": "Великую Китайскую стену видно невооружённым глазом из космоса",
     "expected": "ЛОЖЬ", "diff": 3, "cat": "myths"},

    # 13. Negation trap — "ни разу не" (ложь, были полёты)
    {"claim": "СССР ни разу не отправлял женщин в космос до 1990 года",
     "expected": "ЛОЖЬ", "diff": 3, "cat": "negation_trap"},

    # 14. Составное: Титаник затонул в 1912 (правда) + в Тихом океане (ложь)
    {"claim": "Титаник затонул в 1912 году в Тихом океане",
     "expected": "СОСТАВНОЕ", "diff": 3, "cat": "composite",
     "expected_subs": ["ПРАВДА", "ЛОЖЬ"]},

    # ===== VERY HARD (d=4-5) — 6 =====

    # 15. Adversarial FALSE — правдоподобная ложь (Эверест не самая высокая)
    {"claim": "Эверест является самой высокой горой от подножия до вершины",
     "expected": "ЛОЖЬ", "diff": 4, "cat": "adversarial_false"},

    # 16. Scam-composite: реальная компания + фейковая акция
    {"claim": "Сбербанк является крупнейшим банком России и сейчас раздаёт по 10 тысяч рублей "
              "каждому кто перейдёт по ссылке и введёт данные карты",
     "expected": "СОСТАВНОЕ", "diff": 4, "cat": "scam_composite",
     "expected_subs": ["ПРАВДА", "ЛОЖЬ"]},

    # 17. Конспирология — плоская Земля
    {"claim": "Учёные NASA скрывают что Земля на самом деле плоская и окружена ледяной стеной",
     "expected": "ЛОЖЬ", "diff": 4, "cat": "conspiracy"},

    # 18. Adversarial TRUE — бананы радиоактивны (но безвредно)
    {"claim": "Бананы содержат радиоактивный изотоп калий-40 и являются слегка радиоактивными",
     "expected": "ПРАВДА", "diff": 4, "cat": "adversarial_true"},

    # 19. Скам с deepfake — президент раздаёт деньги
    {"claim": "Президент России объявил о программе выплаты 100 тысяч рублей каждому "
              "гражданину кто зарегистрируется на специальном сайте до конца месяца",
     "expected": "ЛОЖЬ", "diff": 4, "cat": "scam"},

    # 20. Negation trap TRUE — "никогда не" (правда!)
    {"claim": "Ни одна страна мира никогда не высаживала человека на Марс",
     "expected": "ПРАВДА", "diff": 5, "cat": "negation_trap"},
]

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
    actual_statuses = [sv.get("status", "").upper() for sv in sub_verdicts]
    expected_sorted = sorted(expected_subs)
    actual_sorted = sorted(actual_statuses[:len(expected_subs)])
    return expected_sorted == actual_sorted


def main():
    from pipeline import FactCheckPipeline
    from config import SearchConfig
    from model import find_best_adapter

    n = len(TEST_CLAIMS)
    adapter_path = find_best_adapter()
    print(f"Adapter: {adapter_path or 'base'}")
    pipeline = FactCheckPipeline(adapter_path=adapter_path, search_config=SearchConfig())

    results = []
    total_time = 0

    for i, tc in enumerate(TEST_CLAIMS):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{n}] (d={tc['diff']}, {tc['cat']}) {tc['claim'][:80]}")
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
                print(f"    Explanation: {result['_explanation']}")

            results.append({
                "claim": tc["claim"][:55],
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
            import traceback
            traceback.print_exc()
            print(f"\n>>> ERROR: {e} ({elapsed:.1f}s)")
            results.append({
                "claim": tc["claim"][:55],
                "expected": tc["expected"],
                "predicted": "ERROR",
                "score": 0,
                "correct": False,
                "time": round(elapsed, 1),
                "diff": tc["diff"],
                "cat": tc["cat"],
            })

    # Summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY: brutal {n}-claim test (scam + adversarial + negation traps)")
    print(f"{'='*70}")
    correct_count = sum(1 for r in results if r["correct"])
    print(f"Accuracy: {correct_count}/{n} ({correct_count/n:.0%})")
    print(f"Avg time: {total_time/n:.1f}s | Total: {total_time:.0f}s")

    print(f"\n{'Claim':<57s} {'Exp':<15s} {'Got':<15s} {'Score':>5s} {'Time':>6s}")
    print("-" * 102)
    for r in results:
        ok = "OK" if r["correct"] else "MISS"
        print(f"{r['claim']:<57s} {r['expected']:<15s} {r['predicted']:<15s} {r['score']:>5d} {r['time']:>5.1f}s {ok:>5s}")

    # Per-difficulty
    print(f"\nPer difficulty:")
    for d in sorted(set(r["diff"] for r in results)):
        d_results = [r for r in results if r["diff"] == d]
        d_correct = sum(1 for r in d_results if r["correct"])
        print(f"  d={d}: {d_correct}/{len(d_results)}")

    # Per-category
    print(f"\nPer category:")
    for cat in sorted(set(r["cat"] for r in results)):
        c_results = [r for r in results if r["cat"] == cat]
        c_correct = sum(1 for r in c_results if r["correct"])
        print(f"  {cat}: {c_correct}/{len(c_results)}")

    # Critical errors
    critical = sum(1 for r in results
                   if (r["expected"] == "ПРАВДА" and r["predicted"] == "ЛОЖЬ")
                   or (r["expected"] == "ЛОЖЬ" and r["predicted"] == "ПРАВДА"))
    print(f"\nCritical errors (ПРАВДА<->ЛОЖЬ): {critical}")

if __name__ == "__main__":
    main()
