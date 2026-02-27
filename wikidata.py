"""Wikidata SPARQL integration for structured fact verification.

V8: Расширенный маппинг свойств (~20), русские метки,
    транзитивные запросы, NER-интеграция через Natasha.

Проверяет структурированные факты через Wikidata Knowledge Graph:
- Основатель (P112), дата основания (P571)
- Спутники (P397, P398), координаты (P625)
- Страна (P17), материк (P30), столица (P36)
- Население (P1082), высота (P2044)
- Дата рождения/смерти (P569, P570)
- Должность (P39), место рождения (P19)
- Автор (P50), жанр (P136)
- Значимое событие (P793)

Использует публичный SPARQL endpoint (без ключа API).
"""

import re
import json
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Any


SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "AntifakeBot/5.0 (fact-checking; Python)"

# V8: Расширенный маппинг свойств Wikidata
PROPERTY_MAP = {
    # Организации
    "founder": "P112",          # основатель
    "founded_by": "P112",
    "inception": "P571",        # дата основания
    "headquarters": "P159",     # штаб-квартира
    # География
    "country": "P17",           # страна
    "continent": "P30",         # материк
    "capital": "P36",           # столица
    "location": "P276",         # местоположение
    "located_in": "P131",       # расположен в (адм. единица)
    # Астрономия
    "parent_body": "P397",      # орбита вокруг (для спутников)
    "satellite": "P398",        # естественные спутники
    # Гидрография
    "mouth": "P403",            # устье реки
    "origin": "P740",           # место происхождения
    # Биография
    "birth_date": "P569",       # дата рождения
    "death_date": "P570",       # дата смерти
    "birth_place": "P19",       # место рождения
    "death_place": "P20",       # место смерти
    "occupation": "P106",       # род занятий
    "position": "P39",          # должность
    "employer": "P108",         # работодатель
    # Статистика
    "population": "P1082",      # население
    "elevation": "P2044",       # высота над уровнем моря
    "area": "P2046",            # площадь
    # Творчество
    "author": "P50",            # автор
    "genre": "P136",            # жанр
    "publication_date": "P577", # дата публикации
    # События
    "significant_event": "P793",  # значимое событие
    "participants": "P710",       # участники
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

        # V8: Fallback — поиск на английском если русский не нашёл
        if lang == "ru":
            return resolve_entity(name, lang="en")
        return None
    except Exception as e:
        print(f"  [Wikidata] Entity resolution ошибка для '{name}': {e}")
        return None


def get_entity_properties(qid: str, properties: List[str],
                          lang: str = "ru") -> Dict[str, List[str]]:
    """Получает значения свойств для сущности.

    V8: Единый SPARQL запрос для всех свойств (вместо N запросов).
    """
    result: Dict[str, List[str]] = {}

    if not properties:
        return result

    # V8: Batch query — все свойства в одном запросе
    union_parts = []
    for prop in properties:
        union_parts.append(
            f'{{ wd:{qid} wdt:{prop} ?value_{prop}. '
            f'BIND("{prop}" AS ?prop) BIND(?value_{prop} AS ?value) }}'
        )

    if len(union_parts) == 1:
        query = f"""
        SELECT ?prop ?valueLabel WHERE {{
            {union_parts[0]}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
        }}
        LIMIT 20
        """
    else:
        query = f"""
        SELECT ?prop ?valueLabel WHERE {{
            {{ {' UNION '.join(union_parts)} }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
        }}
        LIMIT 30
        """

    bindings = _sparql_query(query)
    for b in bindings:
        prop = b.get("prop", {}).get("value", "")
        val = b.get("valueLabel", {}).get("value", "")
        if prop and val and not val.startswith("http"):
            if prop not in result:
                result[prop] = []
            if val not in result[prop]:
                result[prop].append(val)

    # Fallback: если batch не вернул — пробуем по одному
    if not result:
        for prop in properties[:3]:
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

    V8: Расширенный набор проверок (~20 свойств).
    """
    facts: List[Dict[str, Any]] = []
    claim_lower = claim.lower()

    # Определяем какие свойства проверять
    checks: List[tuple] = []  # (entity_name, property_id, property_label, expected_value)

    for entity in entities[:5]:
        # Основатель / создатель
        if any(w in claim_lower for w in ['основан', 'создан', 'учрежд', 'основал', 'создал']):
            checks.append((entity, "P112", "основатель", None))
            checks.append((entity, "P571", "дата основания", None))

        # Спутники / орбита
        if any(w in claim_lower for w in ['спутник', 'орбит', 'вращается вокруг']):
            checks.append((entity, "P397", "обращается вокруг", None))

        # Река / материк / страна
        if any(w in claim_lower for w in ['протекает', 'река', 'впадает']):
            checks.append((entity, "P30", "материк", None))
            checks.append((entity, "P17", "страна", None))
            checks.append((entity, "P403", "устье", None))

        # Столица
        if any(w in claim_lower for w in ['столиц']):
            checks.append((entity, "P36", "столица", None))

        # Планета — свойства
        if any(w in claim_lower for w in ['планет']):
            checks.append((entity, "P398", "спутники", None))

        # Место события (затонул, крушение)
        if any(w in claim_lower for w in ['затону', 'крушен', 'гибел', 'катастроф']):
            checks.append((entity, "P276", "место", None))

        # V8: Население
        if any(w in claim_lower for w in ['населен', 'жител', 'человек проживает',
                                            'млн человек', 'млрд человек', 'популяц']):
            checks.append((entity, "P1082", "население", None))

        # V8: Высота / длина
        if any(w in claim_lower for w in ['высот', 'метров', 'км высот', 'возвышается']):
            checks.append((entity, "P2044", "высота", None))

        # V8: Дата рождения / смерти
        if any(w in claim_lower for w in ['родил', 'рождён', 'рождения', 'родился']):
            checks.append((entity, "P569", "дата рождения", None))
            checks.append((entity, "P19", "место рождения", None))
        if any(w in claim_lower for w in ['умер', 'скончал', 'погиб', 'смерт']):
            checks.append((entity, "P570", "дата смерти", None))
            checks.append((entity, "P20", "место смерти", None))

        # V8: Должность / профессия
        if any(w in claim_lower for w in ['президент', 'министр', 'глав', 'директор',
                                            'канцлер', 'премьер', 'мэр', 'губернатор']):
            checks.append((entity, "P39", "должность", None))
        if any(w in claim_lower for w in ['профессия', 'работает', 'является', 'род занятий']):
            checks.append((entity, "P106", "род занятий", None))

        # V8: Автор / творчество
        if any(w in claim_lower for w in ['написал', 'автор', 'сочинил', 'создатель',
                                            'режиссёр', 'композитор']):
            checks.append((entity, "P50", "автор", None))

        # V8: Страна (общий)
        if any(w in claim_lower for w in ['находится в', 'расположен', 'расположена',
                                            'в стране', 'страна']):
            checks.append((entity, "P17", "страна", None))

        # V8: Событие с датой (высадка, запуск, битва)
        if any(w in claim_lower for w in ['высадил', 'высадка', 'запуск', 'полёт',
                                            'битва', 'сражени', 'война']):
            checks.append((entity, "P793", "значимое событие", None))
            checks.append((entity, "P585", "момент времени", None))

    # Дедупликация checks
    seen_checks = set()
    unique_checks = []
    for c in checks:
        key = (c[0], c[1])
        if key not in seen_checks:
            seen_checks.add(key)
            unique_checks.append(c)
    checks = unique_checks

    if not checks:
        return {"found": False, "facts": [], "snippet": ""}

    # Разрешаем сущности → QID
    entity_cache: Dict[str, Optional[str]] = {}
    for entity, _, _, _ in checks:
        if entity not in entity_cache:
            entity_cache[entity] = resolve_entity(entity, lang)

    # Группируем свойства по QID для batch-запросов
    qid_props: Dict[str, List[tuple]] = {}
    for entity, prop_id, prop_label, expected in checks:
        qid = entity_cache.get(entity)
        if not qid:
            continue
        if qid not in qid_props:
            qid_props[qid] = []
        qid_props[qid].append((entity, prop_id, prop_label, expected))

    # Запрашиваем свойства (batch per entity)
    snippets = []
    for qid, prop_list in qid_props.items():
        prop_ids = list(set(p[1] for p in prop_list))
        all_props = get_entity_properties(qid, prop_ids, lang)

        for entity, prop_id, prop_label, expected in prop_list:
            values = all_props.get(prop_id, [])
            if values:
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
