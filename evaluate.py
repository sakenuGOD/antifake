"""
Скрипт тестирования системы на контрольной выборке (Раздел 3.2).

Методика:
  - 50 информационных сообщений (25 true + 25 fake)
  - Включая 5 "тонких" фейков — неверные цифры, устаревшие данные, подмена контекста
  - Метрики: Accuracy, Precision, Recall, F1-Score, Latency, per-class метрики
"""

import json
import os
import time
from typing import List, Dict

# Синтетический датасет (Synthetic Adversarial Dataset)
# 25 достоверных + 25 недостоверных = 50
# Каждый кейс помечен типом для анализа по категориям
TEST_DATASET = [
    # === Class 1 — True: ЧИСЛОВЫЕ (конкретные цифры, проценты, суммы) ===
    {"claim": "Уровень безработицы в России составил 2.3% в 2025 году по данным Росстата", "label": 1, "type": "numerical"},
    {"claim": "Население Земли превысило 8 миллиардов человек", "label": 1, "type": "numerical"},
    {"claim": "Россия является самой большой страной по площади — 17.1 млн км²", "label": 1, "type": "numerical"},
    {"claim": "Глубина озера Байкал составляет 1642 метра", "label": 1, "type": "numerical"},
    {"claim": "Население России составляет около 146 миллионов человек", "label": 1, "type": "numerical"},

    # === Class 1 — True: ДАТЫ (конкретные даты, события во времени) ===
    {"claim": "Олимпийские игры 2024 года прошли в Париже", "label": 1, "type": "date"},
    {"claim": "Великобритания вышла из Евросоюза 31 января 2020 года", "label": 1, "type": "date"},
    {"claim": "ВОЗ объявила пандемию COVID-19 11 марта 2020 года", "label": 1, "type": "date"},
    {"claim": "Чемпионат мира по футболу 2022 года прошёл в Катаре", "label": 1, "type": "date"},
    {"claim": "МКС находится на орбите с ноября 1998 года", "label": 1, "type": "date"},

    # === Class 1 — True: ПЕРСОНЫ (действия конкретных людей) ===
    {"claim": "Илон Маск является генеральным директором Tesla и SpaceX", "label": 1, "type": "person"},
    {"claim": "Apple представила новый iPhone на ежегодной презентации", "label": 1, "type": "person"},
    {"claim": "Google разработала поисковую систему и Android", "label": 1, "type": "person"},

    # === Class 1 — True: ИНСТИТУЦИОНАЛЬНЫЕ (организации, решения) ===
    {"claim": "Центральный банк России регулирует ключевую ставку", "label": 1, "type": "institutional"},
    {"claim": "ВОЗ является специализированным учреждением ООН по вопросам здравоохранения", "label": 1, "type": "institutional"},
    {"claim": "ТАСС является государственным информационным агентством России", "label": 1, "type": "institutional"},
    {"claim": "Газпром является крупнейшей газовой компанией в мире", "label": 1, "type": "institutional"},
    {"claim": "Сбербанк является крупнейшим банком России", "label": 1, "type": "institutional"},

    # === Class 1 — True: СОБЫТИЯ (запуски, открытия, факты) ===
    {"claim": "SpaceX успешно запустила ракету Falcon 9 с мыса Канаверал", "label": 1, "type": "event"},
    {"claim": "Москва является столицей Российской Федерации", "label": 1, "type": "general"},
    {"claim": "Эйфелева башня находится в Париже", "label": 1, "type": "general"},
    {"claim": "Python является одним из самых популярных языков программирования", "label": 1, "type": "general"},
    {"claim": "NVIDIA производит графические процессоры для AI", "label": 1, "type": "general"},
    {"claim": "Telegram является одним из популярных мессенджеров в России", "label": 1, "type": "general"},
    {"claim": "Арктика подвергается последствиям изменения климата", "label": 1, "type": "general"},

    # === Class 0 — False: ТОНКИЕ ЧИСЛОВЫЕ ФЕЙКИ (неверные цифры) ===
    {"claim": "ЦБ РФ экстренно поднял ключевую ставку до 50% в феврале 2025", "label": 0, "type": "numerical"},
    {"claim": "Уровень безработицы в России достиг 12% в 2025 году по данным Росстата", "label": 0, "type": "numerical"},
    {"claim": "Население России превысило 200 миллионов человек по переписи 2025 года", "label": 0, "type": "numerical"},
    {"claim": "Глубина Байкала составляет всего 400 метров", "label": 0, "type": "numerical"},
    {"claim": "Площадь России составляет 5 млн км²", "label": 0, "type": "numerical"},

    # === Class 0 — False: ТОНКИЕ ФЕЙКИ С ДАТАМИ (неверные даты/периоды) ===
    {"claim": "Олимпийские игры 2024 года прошли в Токио", "label": 0, "type": "date"},
    {"claim": "Великобритания вышла из Евросоюза в 2016 году", "label": 0, "type": "date"},
    {"claim": "SpaceX запустила первый пилотируемый полёт на Марс в январе 2025", "label": 0, "type": "date"},
    {"claim": "Чемпионат мира по футболу 2022 года прошёл в Японии", "label": 0, "type": "date"},
    {"claim": "МКС была запущена на орбиту в 2010 году", "label": 0, "type": "date"},

    # === Class 0 — False: ПОДМЕНА ПЕРСОН/КОНТЕКСТА ===
    {"claim": "Марк Цукерберг является генеральным директором Tesla", "label": 0, "type": "person"},
    {"claim": "Samsung разработала операционную систему Android", "label": 0, "type": "person"},
    {"claim": "Россия стала второй по площади страной мира после передачи территорий", "label": 0, "type": "person"},

    # === Class 0 — False: ИНСТИТУЦИОНАЛЬНЫЕ ФЕЙКИ ===
    {"claim": "Bitcoin стал официальной валютой Евросоюза", "label": 0, "type": "institutional"},
    {"claim": "Рубль заменён на криптовалюту по решению правительства", "label": 0, "type": "institutional"},
    {"claim": "ВОЗ объявила что витамин C лечит все виды рака", "label": 0, "type": "institutional"},
    {"claim": "Правительство запретило использование искусственного интеллекта", "label": 0, "type": "institutional"},

    # === Class 0 — False: АБСУРДНЫЕ СОБЫТИЯ ===
    {"claim": "NASA подтвердило обнаружение инопланетной цивилизации на Марсе", "label": 0, "type": "event"},
    {"claim": "Землетрясение магнитудой 15 баллов произошло в центре Москвы", "label": 0, "type": "event"},
    {"claim": "МКС упала на Землю и разрушила целый город", "label": 0, "type": "event"},
    {"claim": "Антарктида полностью растаяла за последний месяц", "label": 0, "type": "event"},
    {"claim": "Москву переименовали в Новоград по указу президента", "label": 0, "type": "event"},
    {"claim": "Интернет будет полностью отключён в России с 1 января", "label": 0, "type": "event"},
    {"claim": "Apple полностью прекратила производство iPhone навсегда", "label": 0, "type": "event"},
]


def compute_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Вычисление Accuracy, Precision, Recall, F1-Score + per-class метрики.

    Positive class = 0 (fake) — мы ищем фейки.
    """
    assert len(predictions) == len(labels)

    tp = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)  # True fake detected
    fp = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)  # Real marked as fake
    fn = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)  # Fake missed
    tn = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)  # True real detected

    accuracy = (tp + tn) / len(labels) if labels else 0
    # Fake detection metrics (positive = fake)
    precision_fake = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_fake = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0
    # Real detection metrics (positive = real)
    precision_real = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_real = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_real = 2 * precision_real * recall_real / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0

    return {
        "accuracy": round(accuracy, 3),
        "precision_fake": round(precision_fake, 3),
        "recall_fake": round(recall_fake, 3),
        "f1_fake": round(f1_fake, 3),
        "precision_real": round(precision_real, 3),
        "recall_real": round(recall_real, 3),
        "f1_real": round(f1_real, 3),
        "f1_macro": round((f1_fake + f1_real) / 2, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate_rag(pipeline) -> Dict:
    """Этап Б: Тестирование с RAG-конвейером (с поиском)."""
    predictions = []
    labels = []
    latencies = []

    for i, sample in enumerate(TEST_DATASET):
        claim = sample["claim"]
        true_label = sample["label"]

        print(f"[{i+1}/{len(TEST_DATASET)}] {claim[:60]}...")

        t0 = time.time()
        result = pipeline.check(claim)
        latency = time.time() - t0
        latencies.append(latency)

        # Маппинг вердикта в label
        verdict = result.get("verdict", "").upper()
        score = result.get("credibility_score", 50)

        if verdict in ("ЛОЖЬ", "FALSE") or score < 30:
            predicted = 0  # fake
        elif verdict in ("ПРАВДА", "TRUE") or score >= 70:
            predicted = 1  # real
        elif score >= 50:
            # НЕ ПОДТВЕРЖДЕНО (50-69): score выше среднего — скорее правда
            predicted = 1
        else:
            # НЕ ПОДТВЕРЖДЕНО (30-49): score ниже среднего — скорее фейк
            predicted = 0

        predictions.append(predicted)
        labels.append(true_label)

        # Трекинг НЕ ПОДТВЕРЖДЕНО отдельно
        is_unverified = "НЕ ПОДТВЕРЖДЕНО" in verdict or (30 <= score < 70 and verdict not in ("ПРАВДА", "TRUE", "ЛОЖЬ", "FALSE"))

        status = "OK" if predicted == true_label else "MISS"
        uv_tag = " [UV]" if is_unverified else ""
        print(f"  -> {verdict} (score={score}) | predicted={predicted}, true={true_label} [{status}]{uv_tag}")

    metrics = compute_metrics(predictions, labels)
    metrics["avg_latency"] = round(sum(latencies) / len(latencies), 2)

    # Per-type breakdown
    type_results = {}
    for i, sample in enumerate(TEST_DATASET):
        ctype = sample.get("type", "general")
        if ctype not in type_results:
            type_results[ctype] = {"preds": [], "labels": []}
        type_results[ctype]["preds"].append(predictions[i])
        type_results[ctype]["labels"].append(labels[i])

    metrics["per_type"] = {}
    for ctype, data in type_results.items():
        type_metrics = compute_metrics(data["preds"], data["labels"])
        metrics["per_type"][ctype] = {
            "accuracy": type_metrics["accuracy"],
            "f1_macro": type_metrics["f1_macro"],
            "count": len(data["labels"]),
        }

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Оценка эффективности системы (Раздел 3.2)")
    parser.add_argument("--output", "-o", type=str, default="data/eval_results.json",
                        help="Путь для сохранения результатов")
    args = parser.parse_args()

    if not os.environ.get("SERPAPI_API_KEY"):
        print("Ошибка: SERPAPI_API_KEY не установлен.")
        return

    from pipeline import FactCheckPipeline
    from config import SearchConfig
    from model import find_best_adapter

    search_config = SearchConfig(api_key=os.environ["SERPAPI_API_KEY"])
    adapter_path = find_best_adapter()
    if adapter_path:
        print(f"Используются адаптеры: {adapter_path}")
    else:
        print("Адаптеры не найдены — используется base модель")

    print("=" * 60)
    print("Загрузка модели для тестирования...")
    print("=" * 60)
    pipeline = FactCheckPipeline(adapter_path=adapter_path, search_config=search_config)

    print("\n" + "=" * 60)
    print("Этап Б: Тестирование RAG-системы (с поиском)")
    print(f"Контрольная выборка: {len(TEST_DATASET)} примеров")
    print("=" * 60 + "\n")

    metrics = evaluate_rag(pipeline)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print(f"Accuracy:     {metrics['accuracy']}")
    print(f"F1 (macro):   {metrics['f1_macro']}")
    print(f"Avg Latency:  {metrics['avg_latency']} сек.")
    print(f"\n--- Fake detection (positive = fake) ---")
    print(f"Precision:    {metrics['precision_fake']}")
    print(f"Recall:       {metrics['recall_fake']}")
    print(f"F1-Score:     {metrics['f1_fake']}")
    print(f"\n--- Real detection (positive = real) ---")
    print(f"Precision:    {metrics['precision_real']}")
    print(f"Recall:       {metrics['recall_real']}")
    print(f"F1-Score:     {metrics['f1_real']}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted FAKE  Predicted REAL")
    print(f"  Actual FAKE    TP={metrics['tp']:3d}          FN={metrics['fn']:3d}")
    print(f"  Actual REAL    FP={metrics['fp']:3d}          TN={metrics['tn']:3d}")

    # Per-type breakdown
    if "per_type" in metrics:
        print(f"\n--- Per-type accuracy ---")
        for ctype, tm in sorted(metrics["per_type"].items()):
            print(f"  {ctype:15s}  acc={tm['accuracy']:.3f}  f1={tm['f1_macro']:.3f}  (n={tm['count']})")
    print("=" * 60)

    # Сохранение
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {args.output}")


if __name__ == "__main__":
    main()
