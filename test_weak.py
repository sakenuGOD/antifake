"""Тест на 6 слабых кейсах — тонкие фейки локаций/персон."""
import time, os, sys

WEAK_CASES = [
    {"claim": "Олимпийские игры 2024 года прошли в Токио",                "label": 0},
    {"claim": "Чемпионат мира по футболу 2022 года прошёл в Японии",      "label": 0},
    {"claim": "Олимпийские игры 2024 года прошли в Париже",               "label": 1},
    {"claim": "Марк Цукерберг является генеральным директором Tesla",     "label": 0},
    {"claim": "Samsung разработала операционную систему Android",          "label": 0},
    {"claim": "Сотрудники банка звонят и просят перевести деньги на безопасный счёт", "label": 0},
]

def main():
    sys.path.insert(0, os.path.dirname(__file__))
    from pipeline import FactCheckPipeline
    from model import find_best_adapter

    adapter_path = find_best_adapter()
    pipe = FactCheckPipeline(adapter_path=adapter_path)
    pipe._meta_classifier = None

    correct = 0
    for i, s in enumerate(WEAK_CASES):
        t0 = time.time()
        result = pipe.check(s["claim"])
        dt = time.time() - t0
        v = result.get("verdict", "").upper().strip()
        pred = 1 if v in ("ПРАВДА", "TRUE") else 0
        ok = pred == s["label"]
        if ok:
            correct += 1
        status = "OK" if ok else "MISS"
        print(f"[{i+1}/6] {s['claim'][:60]}")
        print(f"  → {v} (score={result.get('credibility_score')}) pred={pred} true={s['label']} [{status}] ({dt:.0f}s)\n")

    print(f"Результат: {correct}/{len(WEAK_CASES)} = {correct/len(WEAK_CASES):.0%}")

if __name__ == "__main__":
    main()
