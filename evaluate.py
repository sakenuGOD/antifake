"""
Скрипт тестирования системы на контрольной выборке (Раздел 3.2).

Методика:
  - 50 информационных сообщений (25 true + 25 fake)
  - Два этапа: Baseline (без поиска) vs RAG (с поиском)
  - Метрики: Accuracy, Precision, Recall, F1-Score, Latency
"""

import json
import os
import time
from typing import List, Dict

# Синтетический датасет (Synthetic Adversarial Dataset)
# 25 достоверных + 25 недостоверных сообщений
TEST_DATASET = [
    # === Class 1 — True (достоверные) ===
    {"claim": "Уровень безработицы в России составил 2.3% в 2025 году по данным Росстата", "label": 1},
    {"claim": "SpaceX успешно запустила ракету Falcon 9 с мыса Канаверал", "label": 1},
    {"claim": "Москва является столицей Российской Федерации", "label": 1},
    {"claim": "Олимпийские игры 2024 года прошли в Париже", "label": 1},
    {"claim": "Apple представила новый iPhone на ежегодной презентации", "label": 1},
    {"claim": "Население Земли превысило 8 миллиардов человек", "label": 1},
    {"claim": "Центральный банк России регулирует ключевую ставку", "label": 1},
    {"claim": "Telegram является одним из популярных мессенджеров в России", "label": 1},
    {"claim": "Международная космическая станция находится на орбите Земли", "label": 1},
    {"claim": "Python является одним из самых популярных языков программирования", "label": 1},
    {"claim": "ВОЗ является специализированным учреждением ООН по вопросам здравоохранения", "label": 1},
    {"claim": "Газпром является крупнейшей газовой компанией в мире", "label": 1},
    {"claim": "Нобелевская премия вручается ежегодно в Стокгольме", "label": 1},
    {"claim": "Байкал является самым глубоким озером в мире", "label": 1},
    {"claim": "Рубль является официальной валютой Российской Федерации", "label": 1},
    {"claim": "NVIDIA производит графические процессоры для AI", "label": 1},
    {"claim": "Курс доллара определяется на валютном рынке", "label": 1},
    {"claim": "Сбербанк является крупнейшим банком России", "label": 1},
    {"claim": "Google разработала поисковую систему и Android", "label": 1},
    {"claim": "Чемпионат мира по футболу проводится раз в 4 года", "label": 1},
    {"claim": "Россия является самой большой страной по площади", "label": 1},
    {"claim": "Эйфелева башня находится в Париже", "label": 1},
    {"claim": "Яндекс является российской технологической компанией", "label": 1},
    {"claim": "Арктика подвергается последствиям изменения климата", "label": 1},
    {"claim": "ТАСС является государственным информационным агентством России", "label": 1},

    # === Class 0 — False (недостоверные) ===
    {"claim": "ЦБ РФ экстренно поднял ключевую ставку до 50% сегодня утром", "label": 0},
    {"claim": "Россия вышла из ООН и разорвала все дипломатические отношения", "label": 0},
    {"claim": "Apple полностью прекратила производство iPhone навсегда", "label": 0},
    {"claim": "Учёные доказали что Земля на самом деле плоская", "label": 0},
    {"claim": "Москву переименовали в Новоград по указу президента", "label": 0},
    {"claim": "Bitcoin стал официальной валютой Евросоюза", "label": 0},
    {"claim": "NASA подтвердило обнаружение инопланетной цивилизации на Марсе", "label": 0},
    {"claim": "Интернет будет полностью отключён в России с 1 января", "label": 0},
    {"claim": "Все школы России перейдут на шестидневную учебную неделю навсегда", "label": 0},
    {"claim": "Telegram заблокирован и удалён из всех магазинов приложений мира", "label": 0},
    {"claim": "Бензин в России станет бесплатным по новому закону", "label": 0},
    {"claim": "Антарктида полностью растаяла за последний месяц", "label": 0},
    {"claim": "Китай присоединил Луну к своей территории", "label": 0},
    {"claim": "ВОЗ объявила что витамин C лечит все виды рака", "label": 0},
    {"claim": "Газпром бесплатно раздаёт газ всем странам Европы", "label": 0},
    {"claim": "Рубль заменён на криптовалюту по решению правительства", "label": 0},
    {"claim": "Google закрывается и прекращает работу всех сервисов", "label": 0},
    {"claim": "Землетрясение магнитудой 15 баллов произошло в центре Москвы", "label": 0},
    {"claim": "Все университеты России отменили вступительные экзамены навсегда", "label": 0},
    {"claim": "Илон Маск купил Россию за 1 триллион долларов", "label": 0},
    {"claim": "Нобелевскую премию отменили навсегда", "label": 0},
    {"claim": "Озеро Байкал полностью высохло", "label": 0},
    {"claim": "Сбербанк раздаёт по миллиону рублей каждому клиенту", "label": 0},
    {"claim": "Правительство запретило использование искусственного интеллекта", "label": 0},
    {"claim": "МКС упала на Землю и разрушила целый город", "label": 0},
]


def compute_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Вычисление Accuracy, Precision, Recall, F1-Score."""
    assert len(predictions) == len(labels)

    tp = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)  # True fake detected
    fp = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)  # Real marked as fake
    fn = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)  # Fake missed
    tn = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)  # True real detected

    accuracy = (tp + tn) / len(labels) if labels else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
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
        else:
            # НЕ ПОДТВЕРЖДЕНО (30-69): используем score для решения
            predicted = 1 if score >= 50 else 0

        predictions.append(predicted)
        labels.append(true_label)

        status = "OK" if predicted == true_label else "MISS"
        print(f"  -> {verdict} (score={score}) | predicted={predicted}, true={true_label} [{status}]")

    metrics = compute_metrics(predictions, labels)
    metrics["avg_latency"] = round(sum(latencies) / len(latencies), 2)

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

    search_config = SearchConfig(api_key=os.environ["SERPAPI_API_KEY"])
    adapter_path = "adapters/fact_checker_lora"
    adapter_path = adapter_path if os.path.exists(adapter_path) else None

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
    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"F1-Score:  {metrics['f1_score']}")
    print(f"Avg Latency: {metrics['avg_latency']} сек.")
    print(f"\nConfusion Matrix: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    print("=" * 60)

    # Сохранение
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {args.output}")


if __name__ == "__main__":
    main()
