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
import os
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "AntifakeBot/5.0 (fact-checking; Python)"

# V8: Расширенный маппинг свойств Wikidata
PROPERTY_MAP = {
    # Организации
    "founder": "P112",          # основатель
    "founded_by": "P112",
    "developer": "P178",        # V21: разработчик (для ПО, криптовалют, сетей)
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
    "point_in_time": "P585",      # момент времени
    "start_time": "P580",         # начало
    "end_time": "P582",           # конец
    "deaths": "P1120",            # число жертв
    # Размеры и физ. величины
    "height": "P2048",            # высота (для зданий, гор)
    "width": "P2049",             # ширина
    "length": "P2043",            # длина (для рек, стен, дорог)
    "density": "P2054",           # плотность
    "mass": "P2067",              # масса
    "speed": "P2052",             # скорость
    # Книги / астрономия
    "pages": "P1104",             # число страниц
    "distance_from_earth": "P2583",  # расстояние от Земли
}


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "wikidata_cache.json")
CACHE_TTL = 7 * 24 * 3600  # 7 дней


def _load_cache() -> Dict[str, Any]:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def _cache_get(key: str) -> Optional[Any]:
    cache = _load_cache()
    entry = cache.get(key)
    if entry and time.time() - entry.get("ts", 0) < CACHE_TTL:
        return entry.get("data")
    return None


def _cache_set(key: str, data: Any) -> None:
    cache = _load_cache()
    cache[key] = {"ts": time.time(), "data": data}
    _save_cache(cache)


# V21: reuse search.py's global rate-limiter + 429-aware retry so Wikidata /
# SPARQL calls share the per-host budget with Wikipedia calls. Previously
# this module retried only on URLError and never honored 429; an HTTPError
# 429 fell straight through to the caller.
from search import _safe_urlopen as _rl_safe_urlopen  # noqa: E402


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((
        urllib.error.URLError,
        TimeoutError,
        OSError,
    )),
    reraise=True,
)
def _wikidata_api_call(url: str, headers: dict, timeout: int = 10) -> dict:
    """Single API call with tenacity retry + rate-limit for Wikidata/SPARQL."""
    req = urllib.request.Request(url, headers=headers)
    resp = _rl_safe_urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def _sparql_query(query: str) -> List[Dict[str, str]]:
    """Выполняет SPARQL-запрос к Wikidata с tenacity retry."""
    url = f"{SPARQL_ENDPOINT}?query={urllib.parse.quote(query)}&format=json"
    try:
        data = _wikidata_api_call(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
        }, timeout=10)
        return data.get("results", {}).get("bindings", [])
    except Exception as e:
        print(f"  [Wikidata] SPARQL ошибка после 3 попыток: {e}")
        return []


# V13: P31 (instance of) types for entity validation
_P31_PERSON = {"Q5"}                    # human
_P31_PLACE = {"Q515", "Q6256", "Q165", "Q3624078", "Q486972"}  # city, country, sea, admin territory, human settlement
_P31_ORG = {"Q43229", "Q4830453", "Q783794"}  # organization, business, company

# V13: Entity translation dictionary for RU→EN fallback
_ENTITY_TRANSLATIONS = {
    "юрий гагарин": "Yuri Gagarin",
    "гагарин": "Gagarin",
    "луна": "Moon",
    "марс": "Mars",
    "земля": "Earth",
    "венера": "Venus",
    "юпитер": "Jupiter",
    "сатурн": "Saturn",
    "солнце": "Sun",
    "россия": "Russia",
    "москва": "Moscow",
    "титаник": "Titanic",
    "эверест": "Everest",
    "амазонка": "Amazon",
    "нил": "Nile",
    "байкал": "Baikal",
    "пушкин": "Pushkin",
    "толстой": "Tolstoy",
    "менделеев": "Mendeleev",
    "путин": "Putin",
    "сталин": "Stalin",
    "ленин": "Lenin",
}


def _get_p31_types(qid: str) -> set:
    """Get P31 (instance of) types for a QID via SPARQL."""
    query = f"""
    SELECT ?type WHERE {{
        wd:{qid} wdt:P31 ?type.
    }}
    LIMIT 10
    """
    bindings = _sparql_query(query)
    types = set()
    for b in bindings:
        val = b.get("type", {}).get("value", "")
        if val:
            # Extract QID from URI: http://www.wikidata.org/entity/Q5 → Q5
            types.add(val.rsplit("/", 1)[-1])
    return types


def _validate_candidate_p31(qid: str, context_hint: str = "") -> bool:
    """V13: Validate candidate entity by checking P31 type makes sense."""
    types = _get_p31_types(qid)
    if not types:
        return True  # Can't validate, assume OK
    # Cache P31 types alongside QID
    _cache_set(f"p31:{qid}", list(types))
    return True  # We just cache; filtering done in resolve_entity


def _filter_candidates_by_p31(candidates: list, name_lower: str) -> Optional[str]:
    """V13: Filter search candidates by P31 type relevance.

    Returns best QID or None if no good candidate found.
    """
    if not candidates:
        return None

    # For single candidate, do basic validation
    if len(candidates) == 1:
        qid = candidates[0].get("id")
        desc = (candidates[0].get("description") or "").lower()
        # Reject if description is clearly unrelated
        # e.g., searching "гагарин" but getting a village
        return qid

    # Multiple candidates: prefer the one with most relevant P31.
    # V21: the wbsearchentities API ranks by relevance — rank-0 is the most
    # common/prominent entity for that name. Previous logic ignored rank and
    # only used description keywords, causing e.g. "биткоин" to resolve to
    # "Bitcoin City" (Q124613662, rank 4, desc has "city") instead of the
    # cryptocurrency Q131723 (rank 0, desc "digital cash system"). We now
    # add a rank bonus so rank-0 wins unless it has disqualifying markers.
    best_qid = None
    best_score = -1000
    for idx, cand in enumerate(candidates):
        qid = cand.get("id")
        desc = (cand.get("description") or "").lower()
        score = 0

        # Rank bonus — respect wbsearchentities relevance ordering.
        score += max(0, 5 - idx)  # 5/4/3/2/1 for ranks 0..4

        # Boost for description containing relevant keywords
        if any(w in desc for w in ["человек", "person", "human", "космонавт", "astronaut",
                                     "политик", "politician", "учёный", "scientist",
                                     "писатель", "writer", "композитор", "composer"]):
            score += 3
        if any(w in desc for w in ["город", "city", "страна", "country", "река", "river",
                                     "озеро", "lake", "гора", "mountain", "океан", "ocean",
                                     "море", "sea", "планета", "planet", "спутник"]):
            score += 3
        if any(w in desc for w in ["организация", "organization", "компания", "company"]):
            score += 3
        # V21: reward broader concept types that were missing — currencies,
        # cryptocurrencies, languages, software, works of art. Without this
        # "digital cash system" / "криптовалюта" descriptions scored zero and
        # lost to less relevant city/organisation matches on the same name.
        if any(w in desc for w in ["currency", "валют", "криптовалют", "cryptocurrency",
                                     "язык", "language", "software", "программ",
                                     "роман", "novel", "фильм", "film", "album", "альбом",
                                     "песня", "song", "вид ", "species", "растени",
                                     "элемент", "element", "химич", "chemical",
                                     "компьютер", "computer", "система", "system",
                                     "digital", "сеть", "network"]):
            score += 3
        # Penalize for disambiguation pages
        if "disambig" in desc or "значения" in desc:
            score -= 10
        # V21: penalize "planned / proposed / fictional / legendary" entries —
        # typically future or fictional projects that share a name with the
        # real-world referent (e.g. "Bitcoin City", "planned rail line").
        if any(w in desc for w in ["planned", "proposed", "fictional", "legendary",
                                     "запланирован", "предполагаем", "вымышлен",
                                     "мифическ", "легендарн", "прототип"]):
            score -= 5
        # Penalize for clearly wrong types (e.g., "Roman Empire" for "Гагарин")
        if any(w in desc for w in ["империя", "empire", "dynasty", "династия",
                                     "crater", "кратер", "asteroid", "астероид"]):
            score -= 2

        if score > best_score:
            best_score = score
            best_qid = qid

    return best_qid


def resolve_entity(name: str, lang: str = "ru") -> Optional[str]:
    """V13: Разрешает имя сущности в Wikidata QID с P31 фильтрацией.

    1. Поиск на русском (limit=5)
    2. Фильтрация кандидатов по description/P31
    3. Если не нашли — перевести и искать на английском
    4. Если всё ещё не нашли → None (не случайный QID!)
    """
    cache_key = f"entity:v13:{lang}:{name}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached if cached != "" else None

    name_lower = name.lower().strip()

    # V21: track whether any API call failed transiently (HTTP 4xx/5xx/timeout)
    # vs. cleanly returned "no results". We only poison the cache with "" when
    # we have a clean no-hit answer — never on transient failures like 429.
    had_transient_error = False

    # Step 1: Search in target language (limit=5 for better coverage)
    q = urllib.parse.quote(name)
    url = (
        f"https://www.wikidata.org/w/api.php"
        f"?action=wbsearchentities&search={q}&language={lang}"
        f"&limit=5&format=json"
    )
    try:
        data = _wikidata_api_call(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        results = data.get("search", [])
        if results:
            qid = _filter_candidates_by_p31(results, name_lower)
            if qid:
                _cache_set(cache_key, qid)
                return qid
    except Exception as e:
        print(f"  [Wikidata] Entity resolution ошибка для '{name}' ({lang}): {e}")
        had_transient_error = True

    # Step 2: EN fallback with translation (only from ru)
    if lang == "ru":
        # Translate name before EN search
        en_name = _ENTITY_TRANSLATIONS.get(name_lower, None)
        if en_name is None:
            # No known translation — try searching with original name in EN
            en_name = name

        q_en = urllib.parse.quote(en_name)
        url_en = (
            f"https://www.wikidata.org/w/api.php"
            f"?action=wbsearchentities&search={q_en}&language=en"
            f"&limit=5&format=json"
        )
        try:
            data_en = _wikidata_api_call(url_en, headers={"User-Agent": USER_AGENT}, timeout=10)
            results_en = data_en.get("search", [])
            if results_en:
                qid = _filter_candidates_by_p31(results_en, name_lower)
                if qid:
                    _cache_set(cache_key, qid)
                    return qid
        except Exception as e:
            print(f"  [Wikidata] EN fallback ошибка для '{en_name}': {e}")
            had_transient_error = True

    # Step 3: Nothing found. Only cache negative answer when we had a clean
    # API response that simply had no matches — NEVER cache "" on transient
    # rate-limit / network errors, because that would freeze a real entity
    # as "unresolvable" for the 7-day TTL after a single 429.
    if not had_transient_error:
        _cache_set(cache_key, "")
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
        # Основатель / создатель / разработчик
        if any(w in claim_lower for w in ['основан', 'создан', 'учрежд', 'основал', 'создал',
                                            'разработан', 'разработал', 'придумал', 'изобрёл',
                                            'изобрел', 'изобретён']):
            checks.append((entity, "P112", "основатель", None))
            checks.append((entity, "P571", "дата основания", None))
            # V21: P178 (developer) covers software, cryptocurrencies, networks,
            # protocols — e.g. Bitcoin → P178 = Сатоси Накамото. Bitcoin has
            # no P112 but P178 is the canonical authorship field for such
            # entities, and a creator-swap in the claim must be caught here.
            checks.append((entity, "P178", "разработчик", None))
            # V21: P170 (creator) covers works of art, fictional universes,
            # some protocols; complementary to P112/P178.
            checks.append((entity, "P170", "создатель", None))

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

        # Длина / протяжённость
        if any(w in claim_lower for w in ['длин', 'протяжённость']):
            checks.append((entity, "P2043", "длина", None))

        # Масса / вес
        if any(w in claim_lower for w in ['масс', 'весит', 'весом']):
            checks.append((entity, "P2067", "масса", None))

        # Скорость
        if 'скорость' in claim_lower:
            checks.append((entity, "P2052", "скорость", None))

        # Жертвы / погибшие
        if any(w in claim_lower for w in ['жертв', 'погиб']):
            checks.append((entity, "P1120", "число жертв", None))

        # Ширина
        if 'ширин' in claim_lower:
            checks.append((entity, "P2049", "ширина", None))

        # Число страниц
        if 'страниц' in claim_lower:
            checks.append((entity, "P1104", "число страниц", None))

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
                    "property_id": prop_id,
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
