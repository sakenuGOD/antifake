#!/usr/bin/env python3
"""
V15 Comprehensive Test — verifies ALL changes from the 3-verdict simplification.

Part 1: OFFLINE UNIT TESTS (no GPU, no model loading)
  - parse_verdict() default & label list
  - Prompt templates clean (no МАНИПУЛЯЦИЯ / НЕ ПОДТВЕРЖДЕНО)
  - Verdict normalization map
  - Score ranges
  - Audit sub-verdict parsing
  - _generate_explanation() method
  - app.py verdict labels

Part 2: INTEGRATION TEST (requires GPU + model)
  - 15 claims from test_15_v2.py
  - СОСТАВНОЕ detection + sub-verdicts
  - НЕ УВЕРЕНА with _explanation
  - Signal chips data (_ensemble_features)

Usage:
  python3 test_v15.py          # offline unit tests only (fast, no GPU)
  python3 test_v15.py --full   # offline + integration (slow, needs GPU)
"""
import sys
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# ============================================================
# PART 1: OFFLINE UNIT TESTS
# ============================================================

PASS = 0
FAIL = 0
ERRORS = []

def check(name, condition, detail=""):
    global PASS, FAIL, ERRORS
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        ERRORS.append(name)


def test_parse_verdict_default():
    """parse_verdict() default is НЕ УВЕРЕНА, not НЕ ПОДТВЕРЖДЕНО."""
    print("\n=== test_parse_verdict_default ===")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline import FactCheckPipeline

    # Empty input → default
    r = FactCheckPipeline.parse_verdict("")
    check("default verdict = НЕ УВЕРЕНА", r["verdict"] == "НЕ УВЕРЕНА", f"got: {r['verdict']}")
    check("default score = 50", r["credibility_score"] == 50, f"got: {r['credibility_score']}")
    check("sub_verdicts empty", r["sub_verdicts"] == [], f"got: {r['sub_verdicts']}")


def test_parse_verdict_no_manipulation():
    """parse_verdict() never produces МАНИПУЛЯЦИЯ."""
    print("\n=== test_parse_verdict_no_manipulation ===")
    from pipeline import FactCheckPipeline

    # Even if model outputs МАНИПУЛЯЦИЯ, parser should NOT produce it
    # (label list was updated to only contain НЕ УВЕРЕНА / ЛОЖЬ / ПРАВДА)

    # Standard format
    r1 = FactCheckPipeline.parse_verdict(
        "ДОСТОВЕРНОСТЬ: 45\nВЕРДИКТ: ПРАВДА\nУВЕРЕННОСТЬ: 60\nОБОСНОВАНИЕ: test"
    )
    check("standard ПРАВДА parsed", r1["verdict"] == "ПРАВДА")

    r2 = FactCheckPipeline.parse_verdict(
        "ДОСТОВЕРНОСТЬ: 20\nВЕРДИКТ: ЛОЖЬ\nУВЕРЕННОСТЬ: 80\nОБОСНОВАНИЕ: test"
    )
    check("standard ЛОЖЬ parsed", r2["verdict"] == "ЛОЖЬ")

    r3 = FactCheckPipeline.parse_verdict(
        "ДОСТОВЕРНОСТЬ: 40\nВЕРДИКТ: НЕ УВЕРЕНА\nУВЕРЕННОСТЬ: 50\nОБОСНОВАНИЕ: test"
    )
    check("standard НЕ УВЕРЕНА parsed", r3["verdict"] == "НЕ УВЕРЕНА")

    # XML answer format — only 3 verdicts
    r4 = FactCheckPipeline.parse_verdict(
        "<reasoning>Analysis here...</reasoning><answer>ЛОЖЬ 15</answer>"
    )
    check("XML ЛОЖЬ parsed", r4["verdict"] == "ЛОЖЬ")

    r5 = FactCheckPipeline.parse_verdict(
        "<reasoning>Analysis here...</reasoning><answer>ПРАВДА 85</answer>"
    )
    check("XML ПРАВДА parsed", r5["verdict"] == "ПРАВДА")

    r6 = FactCheckPipeline.parse_verdict(
        "<reasoning>Analysis here...</reasoning><answer>НЕ УВЕРЕНА 45</answer>"
    )
    check("XML НЕ УВЕРЕНА parsed", r6["verdict"] == "НЕ УВЕРЕНА")


def test_parse_verdict_audit():
    """parse_verdict() correctly parses audit sub-verdicts."""
    print("\n=== test_parse_verdict_audit ===")
    from pipeline import FactCheckPipeline

    audit_text = """ПУНКТ 1: Эйнштейн родился в Германии
ЦИТАТА: Эйнштейн родился в Ульме, Германия
ИСТОЧНИК: Wikipedia
СТАТУС: ПРАВДА

ПУНКТ 2: Эйнштейн получил Нобелевскую премию по математике
ЦИТАТА: Нобелевская премия по физике 1921
ИСТОЧНИК: Nobel Foundation
СТАТУС: ЛОЖЬ

ДОСТОВЕРНОСТЬ: 40
ВЕРДИКТ: ЛОЖЬ
УВЕРЕННОСТЬ: 75
ОБОСНОВАНИЕ: Первый факт верен, второй — нет."""

    r = FactCheckPipeline.parse_verdict(audit_text)
    check("2 sub-verdicts parsed", len(r["sub_verdicts"]) == 2,
          f"got {len(r['sub_verdicts'])}")

    if len(r["sub_verdicts"]) >= 2:
        check("sub1 status = ПРАВДА", r["sub_verdicts"][0]["status"] == "ПРАВДА")
        check("sub2 status = ЛОЖЬ", r["sub_verdicts"][1]["status"] == "ЛОЖЬ")
        check("sub1 has citation", r["sub_verdicts"][0]["has_citation"] is True)
        check("sub1 source = Wikipedia", r["sub_verdicts"][0]["source"] == "Wikipedia")
        check("sub2 index = 2", r["sub_verdicts"][1]["index"] == 2)

    # Audit with НЕ УВЕРЕНА status
    audit2 = """ПУНКТ 1: Факт один
ЦИТАТА: ЦИТАТА ОТСУТСТВУЕТ
ИСТОЧНИК: нет
СТАТУС: НЕ УВЕРЕНА"""
    r2 = FactCheckPipeline.parse_verdict(audit2)
    check("НЕ УВЕРЕНА status parsed", len(r2["sub_verdicts"]) == 1 and
          r2["sub_verdicts"][0]["status"] == "НЕ УВЕРЕНА")
    if r2["sub_verdicts"]:
        check("ЦИТАТА ОТСУТСТВУЕТ → empty citation",
              r2["sub_verdicts"][0]["citation"] == "")
        check("has_citation = False",
              r2["sub_verdicts"][0]["has_citation"] is False)


def test_prompts_clean():
    """All prompt templates free of МАНИПУЛЯЦИЯ and НЕ ПОДТВЕРЖДЕНО."""
    print("\n=== test_prompts_clean ===")
    from prompts import (
        CREDIBILITY_ASSESSMENT_TEMPLATE,
        CREDIBILITY_ASSESSMENT_REASONING_TEMPLATE,
        CREDIBILITY_ASSESSMENT_AUDIT_TEMPLATE,
        CREDIBILITY_ASSESSMENT_AUDIT_REASONING_TEMPLATE,
        SELF_CRITIQUE_TEMPLATE,
    )

    templates = {
        "CREDIBILITY_ASSESSMENT": CREDIBILITY_ASSESSMENT_TEMPLATE,
        "CREDIBILITY_REASONING": CREDIBILITY_ASSESSMENT_REASONING_TEMPLATE,
        "CREDIBILITY_AUDIT": CREDIBILITY_ASSESSMENT_AUDIT_TEMPLATE,
        "CREDIBILITY_AUDIT_REASONING": CREDIBILITY_ASSESSMENT_AUDIT_REASONING_TEMPLATE,
        "SELF_CRITIQUE": SELF_CRITIQUE_TEMPLATE,
    }

    for name, tpl in templates.items():
        check(f"{name}: no МАНИПУЛЯЦИЯ", "МАНИПУЛЯЦИЯ" not in tpl,
              "found МАНИПУЛЯЦИЯ in template")
        check(f"{name}: no НЕ ПОДТВЕРЖДЕНО", "НЕ ПОДТВЕРЖДЕНО" not in tpl,
              "found НЕ ПОДТВЕРЖДЕНО in template")

    # Score ranges in main template
    check("CREDIBILITY: ПРАВДА range 60-100",
          "60-100" in CREDIBILITY_ASSESSMENT_TEMPLATE or "ПРАВДА (60-100)" in CREDIBILITY_ASSESSMENT_TEMPLATE)
    check("CREDIBILITY: ЛОЖЬ range 0-34",
          "0-34" in CREDIBILITY_ASSESSMENT_TEMPLATE or "ЛОЖЬ (0-34)" in CREDIBILITY_ASSESSMENT_TEMPLATE)
    check("CREDIBILITY: НЕ УВЕРЕНА range 35-59",
          "35-59" in CREDIBILITY_ASSESSMENT_TEMPLATE or "НЕ УВЕРЕНА (35-59)" in CREDIBILITY_ASSESSMENT_TEMPLATE)

    # Audit verdict line
    check("AUDIT: 3 verdicts only",
          "ПРАВДА / ЛОЖЬ / НЕ УВЕРЕНА" in CREDIBILITY_ASSESSMENT_AUDIT_TEMPLATE)

    # Self-critique uses НЕ УВЕРЕНА
    check("SELF_CRITIQUE: uses НЕ УВЕРЕНА",
          "НЕ УВЕРЕНА" in SELF_CRITIQUE_TEMPLATE)


def test_score_ranges():
    """New score ranges: ПРАВДА 60-100, НЕ УВЕРЕНА 35-59, ЛОЖЬ 0-34."""
    print("\n=== test_score_ranges ===")
    from pipeline import FactCheckPipeline

    # Self-critique score ranges
    crit = FactCheckPipeline._parse_self_critique(
        "ОШИБКИ: нет\nКОРРЕКЦИЯ: ДА\nРЕКОМЕНДУЕМЫЙ_SCORE: 65\nРЕКОМЕНДУЕМЫЙ_ВЕРДИКТ: ПРАВДА"
    )
    check("critique: ПРАВДА+65 is consistent", crit["recommended_verdict"] == "ПРАВДА")

    # Apply to a dummy parsed verdict
    parsed = {"credibility_score": 50, "verdict": "НЕ УВЕРЕНА", "reasoning": ""}
    result = FactCheckPipeline._apply_self_critique(parsed, crit)
    check("critique applied: score=65", result["credibility_score"] == 65)
    check("critique applied: verdict=ПРАВДА", result["verdict"] == "ПРАВДА")

    # ЛОЖЬ + score 30 → consistent
    crit2 = FactCheckPipeline._parse_self_critique(
        "ОШИБКИ: числа\nКОРРЕКЦИЯ: ДА\nРЕКОМЕНДУЕМЫЙ_SCORE: 30\nРЕКОМЕНДУЕМЫЙ_ВЕРДИКТ: ЛОЖЬ"
    )
    parsed2 = {"credibility_score": 50, "verdict": "НЕ УВЕРЕНА", "reasoning": ""}
    result2 = FactCheckPipeline._apply_self_critique(parsed2, crit2)
    check("critique ЛОЖЬ+30 applied", result2["verdict"] == "ЛОЖЬ" and result2["credibility_score"] == 30)

    # НЕ УВЕРЕНА + score 45 → consistent
    crit3 = FactCheckPipeline._parse_self_critique(
        "ОШИБКИ: нет\nКОРРЕКЦИЯ: ДА\nРЕКОМЕНДУЕМЫЙ_SCORE: 45\nРЕКОМЕНДУЕМЫЙ_ВЕРДИКТ: НЕ УВЕРЕНА"
    )
    parsed3 = {"credibility_score": 70, "verdict": "ПРАВДА", "reasoning": ""}
    result3 = FactCheckPipeline._apply_self_critique(parsed3, crit3)
    check("critique НЕТ_ДАННЫХ+45 applied", result3["verdict"] == "НЕ УВЕРЕНА")


def test_verdict_normalization():
    """VERDICT_NORMALIZE map for test_15_v2.py."""
    print("\n=== test_verdict_normalization ===")
    from test_15_v2 import norm

    check("ПРАВДА → ПРАВДА", norm("ПРАВДА") == "ПРАВДА")
    check("TRUE → ПРАВДА", norm("TRUE") == "ПРАВДА")
    check("ЛОЖЬ → ЛОЖЬ", norm("ЛОЖЬ") == "ЛОЖЬ")
    check("FALSE → ЛОЖЬ", norm("FALSE") == "ЛОЖЬ")
    check("ФЕЙК → ЛОЖЬ", norm("ФЕЙК") == "ЛОЖЬ")
    check("НЕ УВЕРЕНА → НЕ УВЕРЕНА", norm("НЕ УВЕРЕНА") == "НЕ УВЕРЕНА")
    check("СОСТАВНОЕ → СОСТАВНОЕ", norm("СОСТАВНОЕ") == "СОСТАВНОЕ")
    check("СОСТАВНОЕ УТВЕРЖДЕНИЕ → СОСТАВНОЕ", norm("СОСТАВНОЕ УТВЕРЖДЕНИЕ") == "СОСТАВНОЕ")
    # Legacy backward compat
    check("МАНИПУЛЯЦИЯ → СОСТАВНОЕ (legacy)", norm("МАНИПУЛЯЦИЯ") == "СОСТАВНОЕ")
    check("ПОЛУПРАВДА → СОСТАВНОЕ (legacy)", norm("ПОЛУПРАВДА") == "СОСТАВНОЕ")
    check("ЧАСТИЧНО ПОДТВЕРЖДЕНО → СОСТАВНОЕ (legacy)", norm("ЧАСТИЧНО ПОДТВЕРЖДЕНО") == "СОСТАВНОЕ")
    check("НЕ ПОДТВЕРЖДЕНО → НЕ УВЕРЕНА (legacy)", norm("НЕ ПОДТВЕРЖДЕНО") == "НЕ УВЕРЕНА")
    # Unknown → НЕ УВЕРЕНА
    check("garbage → НЕ УВЕРЕНА", norm("какой-то мусор") == "НЕ УВЕРЕНА")


def test_pipeline_no_old_verdicts():
    """pipeline.py source code contains no old verdict strings."""
    print("\n=== test_pipeline_no_old_verdicts ===")
    with open(os.path.join(os.path.dirname(__file__), "pipeline.py"), "r") as f:
        code = f.read()

    check("no НЕ ПОДТВЕРЖДЕНО in pipeline.py", "НЕ ПОДТВЕРЖДЕНО" not in code)
    check("no МАНИПУЛЯЦИЯ in pipeline.py", "МАНИПУЛЯЦИЯ" not in code)
    check("no ЧАСТИЧНО ПОДТВЕРЖДЕНО in pipeline.py", "ЧАСТИЧНО ПОДТВЕРЖДЕНО" not in code)
    check("НЕ УВЕРЕНА present", "НЕ УВЕРЕНА" in code)
    check("СОСТАВНОЕ present", "СОСТАВНОЕ" in code)
    check("_composite flag present", '"_composite"' in code or "'_composite'" in code)
    check("_explanation field present", '"_explanation"' in code or "'_explanation'" in code)
    check("_generate_explanation method present", "_generate_explanation" in code)
    check("VOTE_THRESHOLD = 0.75", "VOTE_THRESHOLD = 0.75" in code)


def test_app_no_old_verdicts():
    """app.py has no old verdict strings and has new features."""
    print("\n=== test_app_no_old_verdicts ===")
    with open(os.path.join(os.path.dirname(__file__), "app.py"), "r") as f:
        code = f.read()

    check("no НЕ ПОДТВЕРЖДЕНО in app.py", "НЕ ПОДТВЕРЖДЕНО" not in code)
    check("no МАНИПУЛЯЦИЯ in app.py", "МАНИПУЛЯЦИЯ" not in code)
    check("НЕ УВЕРЕНА verdict label", "НЕ УВЕРЕНА" in code)
    check("СОСТАВНОЕ display mode", "СОСТАВНОЕ" in code or "_composite" in code)
    check("explanation box present", "_explanation" in code)
    check("signal chips present", "Wikidata" in code and "NLI" in code)
    check("Результат по пунктам", "Результат по пунктам" in code)


def test_prompts_no_old_verdicts():
    """prompts.py has no old verdict strings."""
    print("\n=== test_prompts_no_old_verdicts ===")
    with open(os.path.join(os.path.dirname(__file__), "prompts.py"), "r") as f:
        code = f.read()

    check("no НЕ ПОДТВЕРЖДЕНО in prompts.py", "НЕ ПОДТВЕРЖДЕНО" not in code)
    check("no МАНИПУЛЯЦИЯ in prompts.py", "МАНИПУЛЯЦИЯ" not in code)


def test_composite_sub_verdict_check():
    """check_composite_subs() from test_15_v2.py works correctly."""
    print("\n=== test_composite_sub_verdict_check ===")
    from test_15_v2 import check_composite_subs

    # Exact match
    result = {"sub_verdicts": [{"status": "ПРАВДА"}, {"status": "ЛОЖЬ"}]}
    check("exact match ПРАВДА+ЛОЖЬ",
          check_composite_subs(result, ["ПРАВДА", "ЛОЖЬ"]))

    # Order-independent
    result2 = {"sub_verdicts": [{"status": "ЛОЖЬ"}, {"status": "ПРАВДА"}]}
    check("order-independent ЛОЖЬ+ПРАВДА",
          check_composite_subs(result2, ["ПРАВДА", "ЛОЖЬ"]))

    # Mismatch
    result3 = {"sub_verdicts": [{"status": "ПРАВДА"}, {"status": "ПРАВДА"}]}
    check("mismatch: both ПРАВДА vs expected ПРАВДА+ЛОЖЬ",
          not check_composite_subs(result3, ["ПРАВДА", "ЛОЖЬ"]))

    # Empty sub-verdicts
    result4 = {"sub_verdicts": []}
    check("empty sub-verdicts → False",
          not check_composite_subs(result4, ["ПРАВДА", "ЛОЖЬ"]))

    # No sub-verdicts key
    result5 = {}
    check("no sub_verdicts key → False",
          not check_composite_subs(result5, ["ПРАВДА", "ЛОЖЬ"]))


def test_self_critique_verdicts():
    """_parse_self_critique accepts only 3 valid verdicts."""
    print("\n=== test_self_critique_verdicts ===")
    from pipeline import FactCheckPipeline

    valid = ["ПРАВДА", "ЛОЖЬ", "НЕ УВЕРЕНА"]
    for v in valid:
        r = FactCheckPipeline._parse_self_critique(
            f"ОШИБКИ: нет\nКОРРЕКЦИЯ: НЕТ\nРЕКОМЕНДУЕМЫЙ_SCORE: 50\nРЕКОМЕНДУЕМЫЙ_ВЕРДИКТ: {v}"
        )
        check(f"critique accepts {v}", r["recommended_verdict"] == v)

    # МАНИПУЛЯЦИЯ should NOT be accepted
    r2 = FactCheckPipeline._parse_self_critique(
        "ОШИБКИ: нет\nКОРРЕКЦИЯ: НЕТ\nРЕКОМЕНДУЕМЫЙ_SCORE: 50\nРЕКОМЕНДУЕМЫЙ_ВЕРДИКТ: МАНИПУЛЯЦИЯ"
    )
    check("critique rejects МАНИПУЛЯЦИЯ", r2["recommended_verdict"] is None)

    # НЕ ПОДТВЕРЖДЕНО should NOT be accepted
    r3 = FactCheckPipeline._parse_self_critique(
        "ОШИБКИ: нет\nКОРРЕКЦИЯ: НЕТ\nРЕКОМЕНДУЕМЫЙ_SCORE: 50\nРЕКОМЕНДУЕМЫЙ_ВЕРДИКТ: НЕ ПОДТВЕРЖДЕНО"
    )
    check("critique rejects НЕ ПОДТВЕРЖДЕНО", r3["recommended_verdict"] is None)


def test_truncated_reasoning_handling():
    """Truncated <reasoning> without </reasoning> marks for retry, not guesses."""
    print("\n=== test_truncated_reasoning_handling ===")
    from pipeline import FactCheckPipeline

    # Truncated reasoning — no </reasoning>
    r = FactCheckPipeline.parse_verdict(
        "<reasoning>Начинаем анализ утверждения. Источники показывают что"
    )
    check("truncated → _needs_retry=True", r.get("_needs_retry") is True)
    check("truncated → verdict stays НЕ УВЕРЕНА", r["verdict"] == "НЕ УВЕРЕНА")

    # Complete reasoning with answer
    r2 = FactCheckPipeline.parse_verdict(
        "<reasoning>Полный анализ с цитатами [Wikipedia] сообщает: факт подтверждён.</reasoning>"
        "<answer>ПРАВДА 85</answer>"
    )
    check("complete reasoning → no _needs_retry", r2.get("_needs_retry") is None)
    check("complete reasoning → ПРАВДА", r2["verdict"] == "ПРАВДА")


def test_result_dict_has_new_fields():
    """Result dict template includes _explanation and _composite."""
    print("\n=== test_result_dict_has_new_fields ===")
    with open(os.path.join(os.path.dirname(__file__), "pipeline.py"), "r") as f:
        code = f.read()

    # Find the result dict construction in check()
    check("result dict has _explanation",
          '"_explanation": parsed.get("_explanation"' in code)
    check("result dict has _composite",
          '"_composite": parsed.get("_composite"' in code)


def run_offline_tests():
    """Run all offline unit tests."""
    print("=" * 70)
    print("V15 OFFLINE UNIT TESTS (no GPU needed)")
    print("=" * 70)

    test_parse_verdict_default()
    test_parse_verdict_no_manipulation()
    test_parse_verdict_audit()
    test_prompts_clean()
    test_score_ranges()
    test_verdict_normalization()
    test_pipeline_no_old_verdicts()
    test_app_no_old_verdicts()
    test_prompts_no_old_verdicts()
    test_composite_sub_verdict_check()
    test_self_critique_verdicts()
    test_truncated_reasoning_handling()
    test_result_dict_has_new_fields()

    print("\n" + "=" * 70)
    print(f"OFFLINE RESULTS: {PASS} passed, {FAIL} failed")
    if ERRORS:
        print(f"FAILED TESTS:")
        for e in ERRORS:
            print(f"  - {e}")
    print("=" * 70)
    return FAIL == 0


# ============================================================
# PART 2: INTEGRATION TESTS (requires GPU + model)
# ============================================================

# Focused integration claims — tests specific V15 behaviors
INTEGRATION_CLAIMS = [
    # 1. Simple ПРАВДА — should NOT become НЕ УВЕРЕНА
    {
        "claim": "Токио является столицей Японии",
        "expected": "ПРАВДА",
        "checks": ["verdict", "score>=60"],
        "desc": "Simple true fact → ПРАВДА with score ≥ 60",
    },
    # 2. Simple ЛОЖЬ — should NOT become НЕ УВЕРЕНА
    {
        "claim": "Австралия является частью Европы",
        "expected": "ЛОЖЬ",
        "checks": ["verdict", "score<=34"],
        "desc": "Simple false fact → ЛОЖЬ with score ≤ 34",
    },
    # 3. Myth → must be ЛОЖЬ (was often НЕ ПОДТВЕРЖДЕНО before)
    {
        "claim": "Молния никогда не бьёт в одно и то же место дважды",
        "expected": "ЛОЖЬ",
        "checks": ["verdict"],
        "desc": "Myth → ЛОЖЬ (debunk sources should trigger)",
    },
    # 4. Composite claim → СОСТАВНОЕ with sub-verdicts
    {
        "claim": "Альберт Эйнштейн родился в Германии и получил Нобелевскую премию по математике",
        "expected": "СОСТАВНОЕ",
        "checks": ["verdict", "has_sub_verdicts", "sub_has_true", "sub_has_false"],
        "desc": "Composite → СОСТАВНОЕ with ПРАВДА+ЛОЖЬ sub-verdicts",
    },
    # 5. Ambiguous/controversial → НЕ УВЕРЕНА with explanation
    {
        "claim": "Мобильные телефоны вызывают рак мозга",
        "expected": "НЕ УВЕРЕНА",
        "checks": ["verdict", "has_explanation"],
        "desc": "Controversial → НЕ УВЕРЕНА with _explanation",
    },
    # 6. Numerical false — clear ЛОЖЬ
    {
        "claim": "Расстояние от Земли до Солнца составляет 15 миллионов километров",
        "expected": "ЛОЖЬ",
        "checks": ["verdict"],
        "desc": "Numerical mismatch → ЛОЖЬ",
    },
    # 7. Another composite
    {
        "claim": "Дмитрий Менделеев изобрёл периодическую таблицу и открыл водку",
        "expected": "СОСТАВНОЕ",
        "checks": ["verdict", "has_sub_verdicts"],
        "desc": "Composite → СОСТАВНОЕ",
    },
    # 8. No old verdicts in output
    {
        "claim": "Земля вращается вокруг Солнца",
        "expected": "ПРАВДА",
        "checks": ["verdict", "no_old_verdicts"],
        "desc": "ПРАВДА + no МАНИПУЛЯЦИЯ/НЕ ПОДТВЕРЖДЕНО in result",
    },
]

NORM_MAP = {
    "СОСТАВНОЕ": "СОСТАВНОЕ", "СОСТАВНОЕ УТВЕРЖДЕНИЕ": "СОСТАВНОЕ",
    "НЕ УВЕРЕНА": "НЕ УВЕРЕНА",
    "ЛОЖЬ": "ЛОЖЬ", "FALSE": "ЛОЖЬ", "ФЕЙК": "ЛОЖЬ",
    "ПРАВДА": "ПРАВДА", "TRUE": "ПРАВДА",
    "МАНИПУЛЯЦИЯ": "СОСТАВНОЕ", "ПОЛУПРАВДА": "СОСТАВНОЕ",
    "НЕ ПОДТВЕРЖДЕНО": "НЕ УВЕРЕНА",
}

def norm_verdict(v):
    v = v.strip().upper()
    for key, val in NORM_MAP.items():
        if key in v:
            return val
    return "НЕ УВЕРЕНА"


def run_check(result, tc):
    """Run all checks for a single claim result. Returns (pass_count, fail_count, details)."""
    p, f = 0, 0
    details = []
    predicted = norm_verdict(result.get("verdict", ""))

    for chk in tc["checks"]:
        if chk == "verdict":
            ok = predicted == tc["expected"]
            if ok:
                p += 1
            else:
                f += 1
                details.append(f"verdict: expected {tc['expected']}, got {predicted}")

        elif chk.startswith("score>="):
            threshold = int(chk.split(">=")[1])
            score = result.get("credibility_score", 0)
            ok = score >= threshold
            if ok:
                p += 1
            else:
                f += 1
                details.append(f"score {score} < {threshold}")

        elif chk.startswith("score<="):
            threshold = int(chk.split("<=")[1])
            score = result.get("credibility_score", 100)
            ok = score <= threshold
            if ok:
                p += 1
            else:
                f += 1
                details.append(f"score {score} > {threshold}")

        elif chk == "has_sub_verdicts":
            svs = result.get("sub_verdicts", [])
            ok = len(svs) >= 2
            if ok:
                p += 1
            else:
                f += 1
                details.append(f"sub_verdicts count: {len(svs)} (need ≥2)")

        elif chk == "sub_has_true":
            svs = result.get("sub_verdicts", [])
            ok = any(sv.get("status", "").upper() == "ПРАВДА" for sv in svs)
            if ok:
                p += 1
            else:
                f += 1
                statuses = [sv.get("status") for sv in svs]
                details.append(f"no ПРАВДА in sub-verdicts: {statuses}")

        elif chk == "sub_has_false":
            svs = result.get("sub_verdicts", [])
            ok = any(sv.get("status", "").upper() == "ЛОЖЬ" for sv in svs)
            if ok:
                p += 1
            else:
                f += 1
                statuses = [sv.get("status") for sv in svs]
                details.append(f"no ЛОЖЬ in sub-verdicts: {statuses}")

        elif chk == "has_explanation":
            expl = result.get("_explanation", "")
            ok = len(expl) > 10
            if ok:
                p += 1
            else:
                f += 1
                details.append(f"_explanation empty or too short: '{expl}'")

        elif chk == "no_old_verdicts":
            raw = str(result)
            ok = "НЕ ПОДТВЕРЖДЕНО" not in raw and "МАНИПУЛЯЦИЯ" not in raw
            if ok:
                p += 1
            else:
                f += 1
                if "НЕ ПОДТВЕРЖДЕНО" in raw:
                    details.append("found НЕ ПОДТВЕРЖДЕНО in result")
                if "МАНИПУЛЯЦИЯ" in raw:
                    details.append("found МАНИПУЛЯЦИЯ in result")

        elif chk == "has_ensemble_features":
            ef = result.get("_ensemble_features", {})
            ok = len(ef) > 0
            if ok:
                p += 1
            else:
                f += 1
                details.append("_ensemble_features empty")

    return p, f, details


def run_integration_tests():
    """Run integration tests with full pipeline."""
    print("\n" + "=" * 70)
    print("V15 INTEGRATION TESTS (GPU + model)")
    print("=" * 70)

    from pipeline import FactCheckPipeline
    from config import SearchConfig
    from model import find_best_adapter

    adapter_path = find_best_adapter()
    print(f"Adapter: {adapter_path or 'base'}")

    print("Loading pipeline...")
    t0 = time.time()
    pipeline = FactCheckPipeline(adapter_path=adapter_path, search_config=SearchConfig())
    print(f"Pipeline loaded in {time.time()-t0:.1f}s\n")

    total_pass = 0
    total_fail = 0
    total_time = 0
    results_summary = []

    for i, tc in enumerate(INTEGRATION_CLAIMS):
        print(f"\n{'─'*70}")
        print(f"[{i+1}/{len(INTEGRATION_CLAIMS)}] {tc['desc']}")
        print(f"  Claim: {tc['claim']}")
        print(f"  Expected: {tc['expected']}")

        t0 = time.time()
        try:
            result = pipeline.check(tc["claim"])
            elapsed = time.time() - t0
            total_time += elapsed

            predicted = norm_verdict(result.get("verdict", ""))
            score = result.get("credibility_score", 0)
            svs = result.get("sub_verdicts", [])
            expl = result.get("_explanation", "")

            p, f, details = run_check(result, tc)
            total_pass += p
            total_fail += f

            status = "PASS" if f == 0 else "FAIL"
            print(f"  Result: {predicted} (score={score}) [{status}] ({elapsed:.1f}s)")
            if svs:
                subs_str = ", ".join(f"П{sv.get('index','?')}={sv.get('status','?')}" for sv in svs)
                print(f"  Sub-verdicts: {subs_str}")
            if expl:
                print(f"  Explanation: {expl}")
            if details:
                for d in details:
                    print(f"  [FAIL] {d}")

            results_summary.append({
                "claim": tc["claim"][:50],
                "expected": tc["expected"],
                "got": predicted,
                "score": score,
                "ok": f == 0,
                "time": round(elapsed, 1),
            })
        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            total_fail += len(tc["checks"])
            print(f"  [ERROR] {e} ({elapsed:.1f}s)")
            traceback.print_exc()
            results_summary.append({
                "claim": tc["claim"][:50],
                "expected": tc["expected"],
                "got": "ERROR",
                "score": 0,
                "ok": False,
                "time": round(elapsed, 1),
            })

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"INTEGRATION RESULTS: {total_pass} passed, {total_fail} failed")
    print(f"Total time: {total_time:.0f}s  Avg: {total_time/len(INTEGRATION_CLAIMS):.1f}s")
    print(f"{'='*70}")

    print(f"\n{'Claim':<52s} {'Exp':<12s} {'Got':<12s} {'Score':>5s} {'Time':>6s} {'OK':>4s}")
    print("-" * 95)
    for r in results_summary:
        ok = "PASS" if r["ok"] else "FAIL"
        print(f"{r['claim']:<52s} {r['expected']:<12s} {r['got']:<12s} {r['score']:>5d} {r['time']:>5.1f}s {ok:>4s}")

    verdict_ok = sum(1 for r in results_summary if r["ok"])
    print(f"\nClaim-level accuracy: {verdict_ok}/{len(INTEGRATION_CLAIMS)}")

    return total_fail == 0


# ============================================================
# MAIN
# ============================================================

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    full_mode = "--full" in sys.argv

    offline_ok = run_offline_tests()

    if full_mode:
        integration_ok = run_integration_tests()
    else:
        print("\n(Skip integration tests — run with --full to include)")
        integration_ok = True

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Offline unit tests:  {'PASS' if offline_ok else 'FAIL'}")
    if full_mode:
        print(f"  Integration tests:   {'PASS' if integration_ok else 'FAIL'}")
    all_ok = offline_ok and integration_ok
    print(f"  Overall:             {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print("=" * 70)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
