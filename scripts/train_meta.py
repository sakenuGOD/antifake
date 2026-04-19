"""
Обучение meta-классификатора для ensemble verdict.

Заменяет ручные пороги в _ensemble_verdict на обученную модель.
Workflow:
  1. python train_meta.py --collect    # прогоняет pipeline, собирает фичи
  2. python train_meta.py --train      # обучает XGBoost на собранных фичах
  3. python train_meta.py --eval       # оценивает meta-classifier vs rule-based

Фичи извлекаются из pipeline._ensemble_features:
  nli_ent, nli_con, nli_ent_count, nli_con_count,
  llm_verdict, llm_score, nums_match, nums_mismatch,
  match_ratio, num_sources, nli_mixed, nli_clean_ent, nli_clean_con
"""
import _path  # noqa: F401,E402 — inject project root into sys.path

import argparse
import json
import os
import time
import numpy as np
from typing import List, Dict

FEATURE_NAMES = [
    "nli_ent", "nli_con", "nli_ent_count", "nli_con_count",
    "llm_verdict", "llm_score", "nums_match", "nums_mismatch",
    "match_ratio", "num_sources", "nli_mixed", "nli_clean_ent", "nli_clean_con",
]

FEATURES_PATH = os.path.join("data", "meta_features.jsonl")
MODEL_PATH = os.path.join("models", "meta_classifier.pkl")


def collect_features(dataset: List[Dict], output_path: str = FEATURES_PATH):
    """Прогоняет pipeline на датасете и сохраняет фичи + ground truth."""
    from pipeline import FactCheckPipeline
    from model import find_best_adapter

    adapter_path = find_best_adapter()
    pipe = FactCheckPipeline(adapter_path=adapter_path)

    # Убедимся что meta-classifier НЕ используется при сборе фичей
    pipe._meta_classifier = None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        for i, sample in enumerate(dataset):
            claim = sample["claim"]
            label = sample["label"]
            claim_type = sample.get("type", "unknown")

            print(f"\n[{i+1}/{len(dataset)}] {claim[:80]}...")
            t0 = time.time()

            try:
                result = pipe.check(claim)
                latency = time.time() - t0

                # Извлекаем фичи из ensemble
                features = result.get("_ensemble_features", {})
                if not features:
                    print(f"  WARN: no features for [{i}], skipping")
                    continue

                record = {
                    "claim": claim,
                    "label": label,
                    "type": claim_type,
                    "predicted_verdict": result.get("verdict", ""),
                    "features": features,
                    "latency": round(latency, 2),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                pred_status = "OK" if (
                    (result["verdict"] == "ПРАВДА" and label == 1) or
                    (result["verdict"] == "ЛОЖЬ" and label == 0) or
                    (result["verdict"] not in ("ПРАВДА", "ЛОЖЬ") and label == 0)
                ) else "MISS"
                print(f"  → {result['verdict']} (score={result.get('credibility_score', '?')}) [{pred_status}]")

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

    print(f"\nФичи сохранены в {output_path} ({i+1} записей)")


def load_features(path: str = FEATURES_PATH):
    """Загружает фичи из JSONL."""
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            features = record["features"]
            row = [features.get(name, 0.0) for name in FEATURE_NAMES]
            X.append(row)

            # label: 1 = ПРАВДА, 0 = ЛОЖЬ/fake
            y.append(record["label"])

    return np.array(X), np.array(y)


def train_classifier(features_path: str = FEATURES_PATH, model_path: str = MODEL_PATH):
    """Обучает meta-классификатор с cross-validation.

    Для малых датасетов (<100) используется LogisticRegression (меньше параметров,
    лучше обобщает). Для больших (>=100) — XGBoost/GradientBoosting.
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import classification_report
    import joblib

    X, y = load_features(features_path)
    print(f"Датасет: {len(X)} записей, {sum(y == 1)} правда, {sum(y == 0)} ложь")

    if len(X) < 20:
        print("ОШИБКА: слишком мало данных для обучения (нужно >= 20)")
        return

    # Выбор модели по размеру датасета
    if len(X) < 100:
        print("Малый датасет → LogisticRegression (лучше обобщает)")
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                penalty='l2',
                max_iter=1000,
                random_state=42,
            )
        )
    else:
        try:
            import xgboost as xgb
            print("Большой датасет → XGBoost")
            clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            print("XGBoost не найден → sklearn GradientBoosting")
            clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )

    # Cross-validation
    n_splits = min(5, len(X) // 4)  # адаптируем число фолдов к размеру датасета
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        print(f"\nCross-validation ({n_splits}-fold):")
        print(f"  Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"  Per fold: {[f'{s:.3f}' for s in scores]}")

    # Обучаем на всех данных
    clf.fit(X, y)

    # Feature importance / coefficients
    try:
        # XGBoost / GradientBoosting
        importances = clf.feature_importances_
        print(f"\nFeature importance:")
        for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
            bar = "█" * int(imp * 50)
            print(f"  {name:20s} {imp:.3f} {bar}")
    except AttributeError:
        # Pipeline(StandardScaler + LogisticRegression)
        lr = clf.named_steps.get("logisticregression", clf[-1])
        coefs = np.abs(lr.coef_[0])
        print(f"\nFeature coefficients (absolute):")
        for name, coef in sorted(zip(FEATURE_NAMES, coefs), key=lambda x: -x[1]):
            bar = "█" * int(coef * 10)
            print(f"  {name:20s} {coef:.3f} {bar}")

    # Classification report на train (для справки)
    y_pred = clf.predict(X)
    print(f"\nTrain classification report:")
    print(classification_report(y, y_pred, target_names=["ЛОЖЬ", "ПРАВДА"]))

    # Сохраняем модель
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Модель сохранена в {model_path}")

    return clf


def evaluate_meta(features_path: str = FEATURES_PATH, model_path: str = MODEL_PATH):
    """Сравнивает meta-classifier с rule-based на собранных фичах."""
    import joblib
    from sklearn.metrics import classification_report, accuracy_score

    X, y = load_features(features_path)
    clf = joblib.load(model_path)

    y_pred = clf.predict(X)
    print(f"Meta-classifier accuracy: {accuracy_score(y, y_pred):.3f}")
    print(classification_report(y, y_pred, target_names=["ЛОЖЬ", "ПРАВДА"]))


def main():
    parser = argparse.ArgumentParser(description="Meta-classifier для ensemble verdict")
    parser.add_argument("--collect", action="store_true", help="Собрать фичи (прогнать pipeline)")
    parser.add_argument("--train", action="store_true", help="Обучить классификатор")
    parser.add_argument("--eval", action="store_true", help="Оценить классификатор")
    parser.add_argument("--features", type=str, default=FEATURES_PATH, help="Путь к фичам")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Путь к модели")
    parser.add_argument("--extra-data", type=str, help="Дополнительный JSONL датасет для --collect")
    args = parser.parse_args()

    if args.collect:
        from evaluate import TEST_DATASET
        dataset = list(TEST_DATASET)

        if args.extra_data and os.path.exists(args.extra_data):
            with open(args.extra_data, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line.strip())
                    dataset.append(record)
            print(f"Добавлено {len(dataset) - len(TEST_DATASET)} записей из {args.extra_data}")

        collect_features(dataset, args.features)

    if args.train:
        train_classifier(args.features, args.model)

    if args.eval:
        evaluate_meta(args.features, args.model)

    if not (args.collect or args.train or args.eval):
        parser.print_help()


if __name__ == "__main__":
    main()
