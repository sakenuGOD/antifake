"""
Быстрая оценка на малых батчах (10 кейсов) с накоплением фичей.

Workflow:
  python quick_eval.py                    # прогоняет следующий батч из 10 кейсов
  python quick_eval.py --train            # + обучает meta-classifier после прогона
  python quick_eval.py --batch N          # прогнать конкретный батч N

Фичи НАКАПЛИВАЮТСЯ в data/meta_features.jsonl между запусками.
Каждый запуск добавляет 10 новых записей.
"""
import _path  # noqa: F401,E402 — inject project root into sys.path

import json
import os
import time
import argparse
from typing import List, Dict

# Батчи разнообразных кейсов — каждый батч уникален
CLAIM_BATCHES = [
    # === Батч 0: Основной eval (49 кейсов) — запускается через evaluate.py ===

    # === Батч 1: География и природа ===
    [
        {"claim": "Юрий Гагарин стал первым человеком в космосе 12 апреля 1961 года", "label": 1, "type": "date"},
        {"claim": "Марианская впадина является самой глубокой точкой Мирового океана — около 11 034 метра", "label": 1, "type": "numerical"},
        {"claim": "Китай является крупнейшей страной мира по численности населения", "label": 0, "type": "numerical"},
        {"claim": "Сахара является самой большой пустыней в мире по площади", "label": 1, "type": "general"},
        {"claim": "Нобелевскую премию мира 2023 года получила Грета Тунберг", "label": 0, "type": "person"},
        {"claim": "Река Амазонка является самой длинной рекой в мире", "label": 0, "type": "general"},
        {"claim": "Валюта Японии — юань", "label": 0, "type": "institutional"},
        {"claim": "Большой адронный коллайдер расположен в Швейцарии и Франции", "label": 1, "type": "institutional"},
        {"claim": "Землетрясение и цунами в Японии 2011 года привело к аварии на АЭС Фукусима", "label": 1, "type": "event"},
        {"claim": "Вулкан Везувий уничтожил Помпеи в 79 году нашей эры", "label": 1, "type": "date"},
    ],

    # === Батч 2: Технологии и наука ===
    [
        {"claim": "Первый iPhone был представлен Apple в 2007 году", "label": 1, "type": "date"},
        {"claim": "Linux был создан Биллом Гейтсом в 1991 году", "label": 0, "type": "person"},
        {"claim": "Скорость света в вакууме составляет примерно 300 000 км/с", "label": 1, "type": "numerical"},
        {"claim": "Квантовый компьютер Google решил задачу за 200 секунд в 2019 году", "label": 1, "type": "event"},
        {"claim": "ChatGPT был создан компанией Google", "label": 0, "type": "person"},
        {"claim": "Периодическая таблица содержит 118 химических элементов", "label": 1, "type": "numerical"},
        {"claim": "Wi-Fi был изобретён в Японии в 2005 году", "label": 0, "type": "date"},
        {"claim": "Tesla произвела свой первый электромобиль Roadster в 2008 году", "label": 1, "type": "date"},
        {"claim": "Операционная система Windows занимает менее 10% рынка десктопов", "label": 0, "type": "numerical"},
        {"claim": "Международная космическая станция вращается вокруг Земли примерно за 90 минут", "label": 1, "type": "numerical"},
    ],

    # === Батч 3: История и политика ===
    [
        {"claim": "Берлинская стена пала 9 ноября 1989 года", "label": 1, "type": "date"},
        {"claim": "Вторая мировая война закончилась в 1943 году", "label": 0, "type": "date"},
        {"claim": "ООН была основана в 1945 году", "label": 1, "type": "institutional"},
        {"claim": "Столица Австралии — Сидней", "label": 0, "type": "general"},
        {"claim": "Транссибирская магистраль является самой длинной железной дорогой в мире", "label": 1, "type": "general"},
        {"claim": "В составе Европейского союза 35 стран-членов", "label": 0, "type": "numerical"},
        {"claim": "Первые Олимпийские игры современности прошли в Афинах в 1896 году", "label": 1, "type": "date"},
        {"claim": "Суэцкий канал соединяет Средиземное и Красное моря", "label": 1, "type": "general"},
        {"claim": "Канада является второй по площади страной в мире", "label": 1, "type": "numerical"},
        {"claim": "Евро является валютой Великобритании", "label": 0, "type": "institutional"},
    ],

    # === Батч 4: Экономика и бизнес ===
    [
        {"claim": "Amazon была основана Джеффом Безосом в 1994 году", "label": 1, "type": "person"},
        {"claim": "Биткоин был создан Виталиком Бутериным", "label": 0, "type": "person"},
        {"claim": "ВВП США является крупнейшим в мире", "label": 1, "type": "institutional"},
        {"claim": "Компания Samsung является южнокорейской", "label": 1, "type": "institutional"},
        {"claim": "Цена нефти Brent никогда не превышала 50 долларов за баррель", "label": 0, "type": "numerical"},
        {"claim": "Alibaba была основана в Японии", "label": 0, "type": "institutional"},
        {"claim": "Минимальная заработная плата в России в 2024 году составляет 19 242 рубля", "label": 1, "type": "numerical"},
        {"claim": "Microsoft приобрела LinkedIn в 2016 году", "label": 1, "type": "event"},
        {"claim": "Фондовая биржа NYSE расположена в Лондоне", "label": 0, "type": "general"},
        {"claim": "Доллар США является мировой резервной валютой", "label": 1, "type": "general"},
    ],

    # === Батч 5: Спорт и культура ===
    [
        {"claim": "Лионель Месси выиграл чемпионат мира по футболу 2022 года с Аргентиной", "label": 1, "type": "event"},
        {"claim": "Олимпийские игры 2028 года пройдут в Лос-Анджелесе", "label": 1, "type": "date"},
        {"claim": "Криштиану Роналду является бразильским футболистом", "label": 0, "type": "person"},
        {"claim": "Шахматы были включены в программу Олимпийских игр 2024", "label": 0, "type": "event"},
        {"claim": "Эрмитаж является одним из крупнейших музеев мира", "label": 1, "type": "general"},
        {"claim": "Формула-1 Гран-при Монако проходит на уличной трассе", "label": 1, "type": "event"},
        {"claim": "Бейсбол является самым популярным видом спорта в мире", "label": 0, "type": "general"},
        {"claim": "Третьяковская галерея находится в Санкт-Петербурге", "label": 0, "type": "general"},
        {"claim": "Большой театр в Москве был основан в 1776 году", "label": 1, "type": "date"},
        {"claim": "НБА является главной баскетбольной лигой Северной Америки", "label": 1, "type": "institutional"},
    ],
]

FEATURES_PATH = os.path.join("data", "meta_features.jsonl")
STATE_PATH = os.path.join("data", "quick_eval_state.json")


def get_next_batch_id() -> int:
    """Возвращает номер следующего непройденного батча."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            state = json.load(f)
        return state.get("next_batch", 1)
    return 1


def save_batch_id(batch_id: int):
    """Сохраняет номер следующего батча."""
    state = {"next_batch": batch_id}
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def run_batch(batch_id: int):
    """Прогоняет один батч через pipeline и сохраняет фичи."""
    if batch_id < 1 or batch_id >= len(CLAIM_BATCHES):
        print(f"Батч {batch_id} не существует (доступны 1-{len(CLAIM_BATCHES)-1})")
        return None

    batch = CLAIM_BATCHES[batch_id]
    print(f"\n{'='*60}")
    print(f"БАТЧ {batch_id}: {len(batch)} кейсов")
    print(f"{'='*60}\n")

    # Загружаем pipeline
    from pipeline import FactCheckPipeline
    from model import find_best_adapter

    adapter_path = find_best_adapter()
    pipe = FactCheckPipeline(adapter_path=adapter_path)
    pipe._meta_classifier = None  # НЕ используем meta-classifier при сборе фичей

    correct = 0
    total = len(batch)
    feature_records = []

    for i, sample in enumerate(batch):
        claim = sample["claim"]
        label = sample["label"]

        print(f"[{i+1}/{total}] {claim[:70]}...")
        t0 = time.time()

        try:
            result = pipe.check(claim)
            latency = time.time() - t0

            verdict = result.get("verdict", "").upper()
            score = result.get("credibility_score", 50)

            if verdict in ("ЛОЖЬ", "FALSE"):
                predicted = 0
            elif verdict in ("ПРАВДА", "TRUE"):
                predicted = 1
            else:
                predicted = 0

            is_correct = predicted == label
            if is_correct:
                correct += 1

            status = "OK" if is_correct else "MISS"
            print(f"  -> {verdict} (score={score}) | pred={predicted}, true={label} [{status}] ({latency:.1f}s)")

            # Фичи
            features = result.get("_ensemble_features", {})
            if features:
                feature_records.append({
                    "claim": claim,
                    "label": label,
                    "type": sample.get("type", "unknown"),
                    "predicted_verdict": result.get("verdict", ""),
                    "features": features,
                    "latency": round(latency, 2),
                    "batch_id": batch_id,
                })

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"БАТЧ {batch_id} РЕЗУЛЬТАТ: {correct}/{total} = {accuracy:.1%}")
    print(f"{'='*60}")

    # Дописываем фичи (append, не перезаписываем)
    if feature_records:
        with open(FEATURES_PATH, "a", encoding="utf-8") as f:
            for record in feature_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Фичи добавлены в {FEATURES_PATH} (+{len(feature_records)} записей)")

    # Считаем общее количество
    total_records = 0
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            total_records = sum(1 for _ in f)
    print(f"Всего фичей: {total_records}")

    # Обновляем state
    save_batch_id(batch_id + 1)

    return {
        "batch_id": batch_id,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "new_features": len(feature_records),
        "total_features": total_records,
    }


def train_meta():
    """Обучает meta-classifier на накопленных фичах."""
    from train_meta import train_classifier
    train_classifier()


def main():
    parser = argparse.ArgumentParser(description="Быстрая оценка батчами по 10 кейсов")
    parser.add_argument("--batch", type=int, help="Номер батча (1-5)")
    parser.add_argument("--train", action="store_true", help="Обучить meta-classifier после прогона")
    parser.add_argument("--list", action="store_true", help="Показать доступные батчи")
    args = parser.parse_args()

    if args.list:
        for i, batch in enumerate(CLAIM_BATCHES):
            if i == 0:
                continue
            print(f"Батч {i}: {len(batch)} кейсов")
            for sample in batch[:3]:
                print(f"  - {sample['claim'][:60]}... [{sample['type']}]")
            print(f"  ...")
        return

    batch_id = args.batch if args.batch else get_next_batch_id()
    result = run_batch(batch_id)

    if result and args.train and result["total_features"] >= 20:
        print("\n--- Обучение meta-classifier ---")
        train_meta()
    elif result and args.train:
        print(f"\nНедостаточно фичей для обучения ({result['total_features']}/20)")


if __name__ == "__main__":
    main()
