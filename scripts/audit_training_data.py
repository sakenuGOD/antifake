"""
Аудит обучающих данных для фактчекера.

Находит:
1. Противоречивые примеры (одно утверждение — разные вердикты)
2. Распределение классов (ПРАВДА / ЛОЖЬ / МАНИПУЛЯЦИЯ / НЕ ПОДТВЕРЖДЕНО)
3. Дубликаты и почти-дубликаты утверждений (first-50-char key + SequenceMatcher)
4. Низкокачественные reasoning:
   - Слишком короткие (< 50 символов reasoning)
   - Отсутствие поля ОБОСНОВАНИЕ
   - Отсутствие поля ИСТОЧНИКИ
   - Шаблонные, несогласованные и прочие проблемы
5. Сохраняет проблемные примеры в data/audit_report.json

Использование:
    python audit_training_data.py
    python audit_training_data.py --russian data/train_russian.jsonl --combined data/train_combined.jsonl
"""
import _path  # noqa: F401,E402 — inject project root into sys.path

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_jsonl(path: str) -> list:
    """Загрузить JSONL файл, вернуть список примеров."""
    examples = []
    if not os.path.exists(path):
        print(f"  [ПРЕДУПРЕЖДЕНИЕ] Файл не найден: {path}")
        return examples

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["_source_file"] = os.path.basename(path)
                obj["_line_num"] = line_num
                examples.append(obj)
            except json.JSONDecodeError as e:
                print(f"  [ОШИБКА] {path}:{line_num} — невалидный JSON: {e}")

    print(f"  Загружено {len(examples)} примеров из {os.path.basename(path)}")
    return examples


def extract_claim(example: dict) -> str:
    """Извлечь текст утверждения из conversations."""
    convos = example.get("conversations", [])
    if not convos:
        return ""

    human_msg = convos[0].get("value", "")
    # Утверждение обычно после "Утверждение:"
    match = re.search(r"Утверждение:\s*(.+?)(?:\n|$)", human_msg)
    if match:
        return match.group(1).strip()
    return human_msg.strip()


def extract_verdict(example: dict) -> str:
    """Извлечь вердикт из gpt-ответа."""
    convos = example.get("conversations", [])
    if len(convos) < 2:
        return ""

    gpt_msg = convos[1].get("value", "")
    match = re.search(r"ВЕРДИКТ:\s*(.+?)(?:\n|$)", gpt_msg)
    if match:
        return match.group(1).strip().upper()
    return ""


def extract_reasoning(example: dict) -> str:
    """Извлечь reasoning из gpt-ответа."""
    convos = example.get("conversations", [])
    if len(convos) < 2:
        return ""

    gpt_msg = convos[1].get("value", "")
    match = re.search(r"<reasoning>(.*?)</reasoning>", gpt_msg, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_answer_block(example: dict) -> str:
    """Извлечь блок <answer>...</answer> из gpt-ответа."""
    convos = example.get("conversations", [])
    if len(convos) < 2:
        return ""

    gpt_msg = convos[1].get("value", "")
    match = re.search(r"<answer>(.*?)</answer>", gpt_msg, re.DOTALL)
    if match:
        return match.group(1).strip()
    return gpt_msg


def extract_score(example: dict) -> int:
    """Извлечь ДОСТОВЕРНОСТЬ из gpt-ответа."""
    convos = example.get("conversations", [])
    if len(convos) < 2:
        return -1

    gpt_msg = convos[1].get("value", "")
    match = re.search(r"ДОСТОВЕРНОСТЬ:\s*(\d+)", gpt_msg)
    if match:
        return int(match.group(1))
    return -1


def normalize_claim(claim: str) -> str:
    """Нормализовать утверждение для поиска дубликатов."""
    text = claim.lower().strip()
    # Убрать знаки препинания
    text = re.sub(r"[^\w\s]", "", text)
    # Убрать множественные пробелы
    text = re.sub(r"\s+", " ", text)
    return text


# ===== АУДИТ 1: Противоречивые примеры =====

def find_contradictions(examples: list) -> list:
    """Найти примеры с одинаковым утверждением, но разными вердиктами."""
    claim_to_verdicts = defaultdict(list)

    for ex in examples:
        claim = normalize_claim(extract_claim(ex))
        verdict = extract_verdict(ex)
        if claim and verdict:
            claim_to_verdicts[claim].append({
                "claim": extract_claim(ex),
                "verdict": verdict,
                "source_file": ex.get("_source_file", ""),
                "line_num": ex.get("_line_num", 0),
            })

    contradictions = []
    for claim, entries in claim_to_verdicts.items():
        verdicts = set(e["verdict"] for e in entries)
        if len(verdicts) > 1:
            contradictions.append({
                "claim": entries[0]["claim"],
                "verdicts": list(verdicts),
                "count": len(entries),
                "entries": entries,
            })

    return contradictions


# ===== АУДИТ 2: Распределение классов =====

def count_class_distribution(examples: list) -> dict:
    """Подсчитать распределение вердиктов."""
    verdicts = [extract_verdict(ex) for ex in examples]
    counter = Counter(verdicts)

    # Нормализуем ключи
    distribution = {}
    for verdict, count in counter.most_common():
        if not verdict:
            distribution["(пустой вердикт)"] = count
        else:
            distribution[verdict] = count

    total = len(examples)
    distribution["_total"] = total

    # Проценты
    percentages = {}
    for key, count in distribution.items():
        if key != "_total":
            percentages[key] = round(100 * count / total, 1) if total > 0 else 0

    return {"counts": distribution, "percentages": percentages}


# ===== АУДИТ 3: Дубликаты и почти-дубликаты =====

def find_duplicates(examples: list, similarity_threshold: float = 0.85) -> dict:
    """Найти точные дубликаты и near-дубликаты утверждений."""
    claims = []
    for ex in examples:
        claim = extract_claim(ex)
        if claim:
            claims.append({
                "claim": claim,
                "normalized": normalize_claim(claim),
                "source_file": ex.get("_source_file", ""),
                "line_num": ex.get("_line_num", 0),
                "verdict": extract_verdict(ex),
            })

    # Точные дубликаты (полная нормализация)
    norm_counter = Counter(c["normalized"] for c in claims)
    exact_duplicates = []
    seen_exact = set()
    for c in claims:
        if norm_counter[c["normalized"]] > 1 and c["normalized"] not in seen_exact:
            seen_exact.add(c["normalized"])
            dups = [x for x in claims if x["normalized"] == c["normalized"]]
            exact_duplicates.append({
                "claim": c["claim"],
                "count": len(dups),
                "entries": dups,
            })

    # Дубликаты по первым 50 символам (грубый ключ)
    first50_groups = defaultdict(list)
    for c in claims:
        key50 = c["normalized"][:50]
        first50_groups[key50].append(c)
    first50_duplicates = []
    for key50, group in first50_groups.items():
        if len(group) > 1:
            first50_duplicates.append({
                "key": key50,
                "count": len(group),
                "entries": group,
            })

    # Near-дубликаты (сравниваем только уникальные, O(n^2) но n невелико)
    unique_claims = list(seen_exact.symmetric_difference(
        set(c["normalized"] for c in claims)
    ))
    # Берём все уникальные нормализованные
    all_norms = list(set(c["normalized"] for c in claims))

    near_duplicates = []
    checked_pairs = set()

    # Оптимизация: сравниваем только если первые 10 символов похожи
    for i in range(len(all_norms)):
        for j in range(i + 1, len(all_norms)):
            if (i, j) in checked_pairs:
                continue
            checked_pairs.add((i, j))

            a, b = all_norms[i], all_norms[j]
            if a == b:
                continue

            # Быстрая предфильтрация: длины должны быть схожи
            len_ratio = min(len(a), len(b)) / max(len(a), len(b)) if max(len(a), len(b)) > 0 else 0
            if len_ratio < 0.5:
                continue

            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio >= similarity_threshold:
                entries_a = [x for x in claims if x["normalized"] == a]
                entries_b = [x for x in claims if x["normalized"] == b]
                near_duplicates.append({
                    "claim_a": entries_a[0]["claim"],
                    "claim_b": entries_b[0]["claim"],
                    "similarity": round(ratio, 3),
                    "verdict_a": entries_a[0]["verdict"],
                    "verdict_b": entries_b[0]["verdict"],
                })

    return {
        "exact_duplicates": exact_duplicates,
        "first50_duplicates": first50_duplicates,
        "near_duplicates": near_duplicates,
        "total_exact_duplicate_groups": len(exact_duplicates),
        "total_first50_duplicate_groups": len(first50_duplicates),
        "total_near_duplicate_pairs": len(near_duplicates),
    }


# ===== АУДИТ 4: Низкокачественные reasoning =====

def check_reasoning_quality(examples: list) -> list:
    """Проверить качество reasoning: слишком короткие, шаблонные."""
    problems = []

    template_phrases = [
        "множественные источники подтвердили",
        "стилистика нейтральна",
        "маркеры манипулятивного воздействия",
        "признаки фабрикации",
        "соответствует стандартам качественной журналистики",
        "мультиисточниковая проверка",
    ]

    # Типичные шаблонные обоснования (появляются многократно)
    reasoning_counter = Counter()

    for ex in examples:
        reasoning = extract_reasoning(ex)
        answer_block = extract_answer_block(ex)
        claim = extract_claim(ex)
        verdict = extract_verdict(ex)
        source_file = ex.get("_source_file", "")
        line_num = ex.get("_line_num", 0)

        issue_list = []

        # Проверка 1: Пустой или слишком короткий reasoning (< 50 символов)
        char_count = len(reasoning) if reasoning else 0
        word_count = len(reasoning.split()) if reasoning else 0
        if char_count == 0:
            issue_list.append("reasoning_отсутствует")
        elif char_count < 50:
            issue_list.append(f"reasoning_слишком_короткий ({char_count} символов)")
        elif word_count < 30:
            issue_list.append(f"reasoning_мало_слов ({word_count} слов)")

        # Проверка 2: Шаблонные фразы
        reasoning_lower = reasoning.lower()
        for phrase in template_phrases:
            if phrase in reasoning_lower:
                issue_list.append(f"шаблонная_фраза: '{phrase}'")

        # Проверка 3a: Отсутствие поля ОБОСНОВАНИЕ в ответе
        if "ОБОСНОВАНИЕ:" not in answer_block:
            issue_list.append("отсутствует_поле_ОБОСНОВАНИЕ")

        # Проверка 3b: Отсутствие поля ИСТОЧНИКИ в ответе
        if "ИСТОЧНИКИ:" not in answer_block:
            issue_list.append("отсутствует_поле_ИСТОЧНИКИ")

        # Проверка 3c: Обрезанные цитаты (заканчиваются на «)
        truncated_quotes = re.findall(r"«[^»]{10,}$", reasoning, re.MULTILINE)
        if truncated_quotes:
            issue_list.append("обрезанная_цитата")

        # Проверка 4: Reasoning не содержит никаких шагов
        if reasoning and not re.search(r"шаг\s*\d|идентификац|факт|проверк|самопроверк", reasoning_lower):
            issue_list.append("нет_структурированных_шагов")

        # Проверка 5: Вердикт без обоснования (нет цитат из источников)
        if reasoning and verdict in ("ПРАВДА", "ЛОЖЬ"):
            if not re.search(r"источник|сообщает|по данным|цитата|«", reasoning_lower):
                issue_list.append("нет_ссылок_на_источники")

        # Проверка 6: Несогласованность вердикта и достоверности
        score = extract_score(ex)
        if score >= 0 and verdict:
            if verdict == "ПРАВДА" and score < 50:
                issue_list.append(f"несогласованность: ПРАВДА + достоверность={score}")
            elif verdict == "ЛОЖЬ" and score > 50:
                issue_list.append(f"несогласованность: ЛОЖЬ + достоверность={score}")

        # Проверка 7: Reasoning повторяет утверждение дословно без анализа
        if reasoning and claim:
            claim_norm = normalize_claim(claim)
            reasoning_norm = normalize_claim(reasoning[:len(claim) * 2])
            if claim_norm in reasoning_norm and word_count < 50:
                issue_list.append("reasoning_просто_повторяет_утверждение")

        # Подсчёт дубликатов reasoning
        if reasoning:
            # Берём первые 100 символов как отпечаток
            fingerprint = reasoning[:100].strip()
            reasoning_counter[fingerprint] += 1

        if issue_list:
            problems.append({
                "claim": claim,
                "verdict": verdict,
                "source_file": source_file,
                "line_num": line_num,
                "issues": issue_list,
                "reasoning_length": word_count,
            })

    # Проверка 8: Шаблонные reasoning (одинаковые начала)
    template_reasonings = []
    for fingerprint, count in reasoning_counter.items():
        if count >= 5:
            template_reasonings.append({
                "fingerprint": fingerprint,
                "count": count,
                "issue": "шаблонный_reasoning_повторяется_5+_раз",
            })

    return problems, template_reasonings


# ===== ГЛАВНАЯ ФУНКЦИЯ =====

def run_audit(russian_path: str, combined_path: str, output_path: str):
    """Запустить полный аудит и вывести отчёт."""
    print("=" * 70)
    print("АУДИТ ОБУЧАЮЩИХ ДАННЫХ ДЛЯ ФАКТЧЕКЕРА")
    print("=" * 70)

    # Загрузка
    print("\n--- Загрузка данных ---")
    russian_examples = load_jsonl(russian_path)
    combined_examples = load_jsonl(combined_path)
    all_examples = russian_examples + combined_examples
    print(f"  Всего: {len(all_examples)} примеров")

    report = {
        "total_examples": len(all_examples),
        "russian_examples": len(russian_examples),
        "combined_examples": len(combined_examples),
    }

    # 1. Противоречия
    print("\n--- 1. Противоречивые примеры ---")
    contradictions = find_contradictions(all_examples)
    report["contradictions"] = contradictions
    if contradictions:
        print(f"  Найдено {len(contradictions)} групп противоречий:")
        for c in contradictions[:10]:
            print(f"    '{c['claim'][:60]}...' — вердикты: {c['verdicts']} ({c['count']} примеров)")
    else:
        print("  Противоречий не найдено.")

    # 2. Распределение классов
    print("\n--- 2. Распределение классов ---")
    print("  train_russian.jsonl:")
    dist_ru = count_class_distribution(russian_examples)
    for verdict, count in dist_ru["counts"].items():
        if verdict != "_total":
            pct = dist_ru["percentages"].get(verdict, 0)
            print(f"    {verdict}: {count} ({pct}%)")

    print("  train_combined.jsonl:")
    dist_comb = count_class_distribution(combined_examples)
    for verdict, count in dist_comb["counts"].items():
        if verdict != "_total":
            pct = dist_comb["percentages"].get(verdict, 0)
            print(f"    {verdict}: {count} ({pct}%)")

    print("  Общее:")
    dist_all = count_class_distribution(all_examples)
    for verdict, count in dist_all["counts"].items():
        if verdict != "_total":
            pct = dist_all["percentages"].get(verdict, 0)
            print(f"    {verdict}: {count} ({pct}%)")

    report["distribution"] = {
        "russian": dist_ru,
        "combined": dist_comb,
        "all": dist_all,
    }

    # 3. Дубликаты
    print("\n--- 3. Дубликаты ---")
    duplicates = find_duplicates(all_examples)
    report["duplicates"] = {
        "exact_duplicate_groups": duplicates["total_exact_duplicate_groups"],
        "first50_duplicate_groups": duplicates["total_first50_duplicate_groups"],
        "near_duplicate_pairs": duplicates["total_near_duplicate_pairs"],
        "exact_duplicates": duplicates["exact_duplicates"][:20],
        "first50_duplicates": duplicates["first50_duplicates"][:20],
        "near_duplicates": duplicates["near_duplicates"][:20],
    }
    print(f"  Точных дубликатов (групп): {duplicates['total_exact_duplicate_groups']}")
    if duplicates["exact_duplicates"]:
        for d in duplicates["exact_duplicates"][:5]:
            print(f"    '{d['claim'][:60]}...' — {d['count']} копий")

    print(f"  Дубликатов по первым 50 символам (групп): {duplicates['total_first50_duplicate_groups']}")
    if duplicates["first50_duplicates"]:
        for d in duplicates["first50_duplicates"][:5]:
            print(f"    key='{d['key'][:50]}' — {d['count']} примеров")

    print(f"  Почти-дубликатов (пар, threshold=0.85): {duplicates['total_near_duplicate_pairs']}")
    if duplicates["near_duplicates"]:
        for nd in duplicates["near_duplicates"][:5]:
            print(f"    [{nd['similarity']:.0%}] '{nd['claim_a'][:40]}...' vs '{nd['claim_b'][:40]}...'")
            if nd["verdict_a"] != nd["verdict_b"]:
                print(f"         Вердикты РАЗЛИЧАЮТСЯ: {nd['verdict_a']} vs {nd['verdict_b']}")

    # 4. Качество reasoning
    print("\n--- 4. Качество reasoning ---")
    problems, template_reasonings = check_reasoning_quality(all_examples)
    report["reasoning_problems"] = problems[:100]  # Ограничим размер отчёта
    report["template_reasonings"] = template_reasonings

    # Подсчёт проблем по типу
    issue_counter = Counter()
    for p in problems:
        for issue in p["issues"]:
            issue_type = issue.split(":")[0].split(" (")[0]
            issue_counter[issue_type] += 1

    print(f"  Примеров с проблемами: {len(problems)} / {len(all_examples)}")
    if issue_counter:
        print("  Типы проблем:")
        for issue_type, count in issue_counter.most_common():
            print(f"    {issue_type}: {count}")

    if template_reasonings:
        print(f"\n  Шаблонных reasoning (повторяются 5+ раз): {len(template_reasonings)}")
        for tr in template_reasonings[:5]:
            print(f"    [{tr['count']}x] '{tr['fingerprint'][:80]}...'")

    # 5. Сводка
    print("\n" + "=" * 70)
    print("СВОДКА АУДИТА")
    print("=" * 70)
    total_issues = 0
    total_issues += len(contradictions)
    total_issues += duplicates["total_exact_duplicate_groups"]
    total_issues += len(problems)

    print(f"  Всего примеров: {len(all_examples)}")
    print(f"  Противоречий: {len(contradictions)}")
    print(f"  Точных дубликатов (групп): {duplicates['total_exact_duplicate_groups']}")
    print(f"  Дубликатов по first-50-char (групп): {duplicates['total_first50_duplicate_groups']}")
    print(f"  Почти-дубликатов: {duplicates['total_near_duplicate_pairs']}")
    print(f"  Проблемных reasoning: {len(problems)}")
    print(f"  Шаблонных reasoning: {len(template_reasonings)}")

    # Рекомендации
    print("\n--- Рекомендации ---")
    if contradictions:
        print("  [!] Удалить или исправить противоречивые примеры")
    if duplicates["total_exact_duplicate_groups"] > 10:
        print("  [!] Дедупликация: удалить точные дубликаты")
    pravda_pct = dist_all["percentages"].get("ПРАВДА", 0)
    lozh_pct = dist_all["percentages"].get("ЛОЖЬ", 0)
    ne_podtv_pct = dist_all["percentages"].get("НЕ ПОДТВЕРЖДЕНО", 0)
    if pravda_pct > 60:
        print(f"  [!] Дисбаланс: ПРАВДА={pravda_pct}% — добавить больше ЛОЖЬ/НЕ ПОДТВЕРЖДЕНО")
    if lozh_pct < 20:
        print(f"  [!] Мало примеров ЛОЖЬ ({lozh_pct}%) — добавить")
    manip_pct = dist_all["percentages"].get("МАНИПУЛЯЦИЯ", 0)
    if manip_pct < 5:
        print(f"  [!] Мало примеров МАНИПУЛЯЦИЯ ({manip_pct}%) — добавить")
    if len(problems) > len(all_examples) * 0.1:
        print(f"  [!] Более 10% примеров с проблемами reasoning — улучшить шаблоны")

    # Сохранение отчёта
    print(f"\n  Сохранение отчёта: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("  Готово!")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Аудит обучающих данных фактчекера"
    )
    parser.add_argument(
        "--russian", type=str,
        default=os.path.join(PROJECT_ROOT, "data", "train_russian.jsonl"),
        help="Путь к train_russian.jsonl",
    )
    parser.add_argument(
        "--combined", type=str,
        default=os.path.join(PROJECT_ROOT, "data", "train_combined.jsonl"),
        help="Путь к train_combined.jsonl",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(PROJECT_ROOT, "data", "audit_report.json"),
        help="Путь для сохранения отчёта",
    )
    args = parser.parse_args()

    run_audit(args.russian, args.combined, args.output)


if __name__ == "__main__":
    main()
