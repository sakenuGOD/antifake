"""Быстрый тест на 5 кейсах + сбор фичей."""
import _path  # noqa: F401,E402 — inject project root into sys.path
import json, time, os

CLAIMS = [
    {"claim": "Население Земли превысило 8 миллиардов человек", "label": 1, "type": "numerical"},
    {"claim": "Берлинская стена пала 9 ноября 1989 года", "label": 1, "type": "date"},
    {"claim": "ChatGPT был создан компанией Google", "label": 0, "type": "person"},
    {"claim": "Вторая мировая война закончилась в 1943 году", "label": 0, "type": "date"},
    {"claim": "Скорость света в вакууме составляет примерно 300 000 км/с", "label": 1, "type": "numerical"},
    {"claim": "Bitcoin стал официальной валютой Евросоюза", "label": 0, "type": "institutional"},
    {"claim": "Первый iPhone был представлен Apple в 2007 году", "label": 1, "type": "date"},
    {"claim": "Столица Австралии — Сидней", "label": 0, "type": "general"},
    {"claim": "ООН была основана в 1945 году", "label": 1, "type": "institutional"},
    {"claim": "Криштиану Роналду является бразильским футболистом", "label": 0, "type": "person"},
]

def main():
    from pipeline import FactCheckPipeline
    from model import find_best_adapter

    adapter_path = find_best_adapter()
    pipe = FactCheckPipeline(adapter_path=adapter_path)
    pipe._meta_classifier = None

    correct = 0
    records = []

    for i, s in enumerate(CLAIMS):
        print(f"\n[{i+1}/{len(CLAIMS)}] {s['claim'][:70]}...")
        t0 = time.time()
        result = pipe.check(s["claim"])
        dt = time.time() - t0

        v = result.get("verdict", "").upper()
        pred = 1 if v in ("ПРАВДА", "TRUE") else 0
        ok = pred == s["label"]
        if ok: correct += 1

        print(f"  → {v} (score={result.get('credibility_score')}) | pred={pred} true={s['label']} [{'OK' if ok else 'MISS'}] ({dt:.0f}s)")

        feat = result.get("_ensemble_features", {})
        if feat:
            records.append({"claim": s["claim"], "label": s["label"], "type": s["type"],
                           "predicted_verdict": result.get("verdict", ""), "features": feat})

    print(f"\nРезультат: {correct}/{len(CLAIMS)} = {correct/len(CLAIMS):.0%}")

    if records:
        path = "data/meta_features.jsonl"
        os.makedirs("data", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Фичи добавлены в {path} (+{len(records)})")

if __name__ == "__main__":
    main()
