"""Wikidata SPARQL integration for structured fact verification.

Проверяет структурированные факты через Wikidata Knowledge Graph:
- Основатель компании (P112)
- Естественные спутники (P397, P398)
- Местоположение (P276, P17)
- Столица (P36)
- Дата основания (P571)
- Река → материк/страна (P30, P17)

Использует публичный SPARQL endpoint (без ключа API).
"""

import re
import json
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Any


SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "AntifakeBot/4.0 (fact-checking; Python)"

# Маппинг свойств Wikidata для типичных проверок
PROPERTY_MAP = {
    "founder": "P112",       # основатель
    "founded_by": "P112",
    "inception": "P571",     # дата основания
    "country": "P17",        # страна
    "continent": "P30",      # материк
    "capital": "P36",        # столица
    "location": "P276",      # местоположение
    "parent_body": "P397",   # орбита вокруг (для спутников)
    "satellite": "P398",     # естественные спутники
    "mouth": "P403",         # устье реки
    "origin": "P740",        # место происхождения
}


def _sparql_query(query: str) -> List[Dict[str, str]]:
    """Выполняет SPARQL-запрос к Wikidata."""
    try:
        url = f"{SPARQL_ENDPOINT}?query={urllib.parse.quote(query)}&format=json"
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
        })
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        return data.get("results", {}).get("bindings", [])
    except Exception as e:
        print(f"  [Wikidata] SPARQL ошибка: {e}")
        return []


def resolve_entity(name: str, lang: str = "ru") -> Optional[str]:
    """Разрешает имя сущности в Wikidata QID.

    "Луна" → "Q405"
    "Microsoft" → "Q2283"
    "Титаник" → "Q25173"
    """
    try:
        q = urllib.parse.quote(name)
        url = (
            f"https://www.wikidata.org/w/api.php"
            f"?action=wbsearchentities&search={q}&language={lang}"
            f"&limit=3&format=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        resp = urllib.request.urlopen(req, timeout=8)
        data = json.loads(resp.read())

        results = data.get("search", [])
        if results:
            return results[0].get("id")
        return None
    except Exception as e:
        print(f"  [Wikidata] Entity resolution ошибка для '{name}': {e}")
        return None


def get_entity_properties(qid: str, properties: List[str],
                          lang: str = "ru") -> Dict[str, List[str]]:
    """Получает значения свойств для сущности.

    Args:
        qid: Wikidata QID (e.g., "Q2283")
        properties: Список свойств (e.g., ["P112", "P571"])
        lang: Язык меток

    Returns:
        {"P112": ["Билл Гейтс", "Пол Аллен"], "P571": ["4 апреля 1975"]}
    """
    result: Dict[str, List[str]] = {}

    for prop in properties:
        query = f"""
        SELECT ?valueLabel WHERE {{
            wd:{qid} wdt:{prop} ?value.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
        }}
        LIMIT 10
        """
        bindings = _sparql_query(query)
        values = []
        for b in bindings:
            val = b.get("valueLabel", {}).get("value", "")
            if val and not val.startswith("http"):
                values.append(val)
        if values:
            result[prop] = values

    return result


def check_structured_facts(claim: str, entities: List[str],
                           lang: str = "ru") -> Dict[str, Any]:
    """Проверяет структурированные факты через Wikidata.

    Анализирует claim, определяет какие свойства проверить,
    и возвращает структурированный результат.

    Returns:
        {
            "found": True/False,
            "facts": [
                {"entity": "Microsoft", "property": "основатель",
                 "expected": "Стив Джобс", "actual": ["Билл Гейтс", "Пол Аллен"],
                 "match": False}
            ],
            "snippet": "Wikidata: Microsoft — основатель: Билл Гейтс, Пол Аллен"
        }
    """
    facts: List[Dict[str, Any]] = []
    claim_lower = claim.lower()

    # Определяем какие свойства проверять
    checks: List[tuple] = []  # (entity_name, property_id, property_label, expected_value)

    for entity in entities[:4]:
        ent_lower = entity.lower()

        # Основатель / создатель
        if any(w in claim_lower for w in ['основан', 'создан', 'учрежд', 'основал', 'создал']):
            checks.append((entity, "P112", "основатель", None))
            checks.append((entity, "P571", "дата основания", None))

        # Спутники / орбита
        if any(w in claim_lower for w in ['спутник', 'орбит']):
            checks.append((entity, "P397", "обращается вокруг", None))

        # Река / материк / страна
        if any(w in claim_lower for w in ['протекает', 'река', 'впадает']):
            checks.append((entity, "P30", "материк", None))
            checks.append((entity, "P17", "страна", None))

        # Столица
        if any(w in claim_lower for w in ['столиц']):
            checks.append((entity, "P36", "столица", None))

        # Планета — свойства
        if any(w in claim_lower for w in ['планет']):
            checks.append((entity, "P398", "спутники", None))

        # Место события (для "затонул в", "произошёл в")
        if any(w in claim_lower for w in ['затону', 'крушен', 'гибел']):
            checks.append((entity, "P276", "место", None))
            checks.append((entity, "P625", "координаты", None))

    if not checks:
        return {"found": False, "facts": [], "snippet": ""}

    # Разрешаем сущности → QID
    entity_cache: Dict[str, Optional[str]] = {}
    for entity, _, _, _ in checks:
        if entity not in entity_cache:
            entity_cache[entity] = resolve_entity(entity, lang)

    # Запрашиваем свойства
    snippets = []
    for entity, prop_id, prop_label, expected in checks:
        qid = entity_cache.get(entity)
        if not qid:
            continue

        props = get_entity_properties(qid, [prop_id], lang)
        values = props.get(prop_id, [])

        if values:
            # Проверяем совпадение с claim
            match = None
            if expected:
                match = any(expected.lower() in v.lower() for v in values)

            facts.append({
                "entity": entity,
                "property": prop_label,
                "wikidata_values": values,
                "match": match,
            })
            val_str = ", ".join(values[:5])
            snippets.append(f"{entity} → {prop_label}: {val_str}")

    snippet = ""
    if snippets:
        snippet = "WIKIDATA ФАКТЫ:\n" + "\n".join(snippets)
        print(f"  [Wikidata] Найдено {len(facts)} фактов: {'; '.join(s[:60] for s in snippets[:3])}")

    return {
        "found": bool(facts),
        "facts": facts,
        "snippet": snippet,
    }


def format_wikidata_hint(wikidata_result: Dict[str, Any]) -> str:
    """Форматирует результат Wikidata для LLM-промпта."""
    if not wikidata_result.get("found"):
        return ""
    return wikidata_result.get("snippet", "")
