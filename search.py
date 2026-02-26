"""Advanced fact-checking search pipeline v4.

Новое в v4:
- Матрица доверенных источников (TRUSTED_SOURCES) по категориям
- QueryClassifier: rule-based классификация + Helsinki-NLP/opus-mt-ru-en перевод (CPU, ~300MB)
  Никаких внешних API — только локальные модели
- Умный генератор DDG-запросов с site: операторами (build_search_queries)
- Жёсткий URL-фильтр (clean_results): blacklist / trusted TLD / unverified
- Логика «нулевой выдачи» как признак фейка (ZERO_RESULTS_CONTEXT)
- Standalone LLM-судья (get_verdict):
    * Если передан уже загруженный Mistral (model + tokenizer) → полноценный AI-вердикт
    * Иначе → детерминистический анализ по структуре источников
- Полная обратная совместимость с pipeline.py (FactCheckSearcher API)
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request
import warnings
from typing import Any, Dict, List, Optional

# Подавляем предупреждение о переименовании duckduckgo_search → ddgs.
# Старый пакет может быть установлен в системном Python — приложение корректно
# работает с любым из них, предупреждение не несёт пользы.
warnings.filterwarnings(
    "ignore",
    message=r"This package.*renamed.*ddgs",
    category=RuntimeWarning,
)

# Импорт DDG-клиента на уровне модуля (один раз, не при каждом вызове _search_ddg).
# Предпочитаем новое имя пакета (ddgs), откатываемся на старое при отсутствии.
try:
    from ddgs import DDGS as _DDGS
except ImportError:
    from duckduckgo_search import DDGS as _DDGS  # type: ignore[no-redef]

_DDG_TIMEOUT = 20  # секунд — достаточно для медленного Bing-бэкенда

from cache import SearchCache
from config import SearchConfig
from source_credibility import boost_by_credibility


# ============================================================
# ШАГ 1: МАТРИЦА ДОВЕРЕННЫХ ИСТОЧНИКОВ
# ============================================================

TRUSTED_SOURCES: Dict[str, List[str]] = {
    "reference": [
        "wikipedia.org",
        "britannica.com",
        "wikimedia.org",
    ],
    "factcheckers_global": [
        "snopes.com",
        "reuters.com",        # reuters.com/fact-check — поддомен, базовый домен тот же
        "apnews.com",
        "politifact.com",
        "fullfact.org",
        "leadstories.com",
        "factcheck.org",
    ],
    "factcheckers_ru": [
        "provereno.media",
        "stopfake.org",
        "lapsha.media",
        "fakenews.ru",
    ],
    "news_agencies_global": [
        "reuters.com",
        "apnews.com",
        "afp.com",
        "bloomberg.com",
        "bbc.com",
        "aljazeera.com",
        "ft.com",
        "wsj.com",
    ],
    "news_agencies_ru": [
        "tass.ru",
        "interfax.ru",
        "rbc.ru",
        "kommersant.ru",
        "vedomosti.ru",
        "ria.ru",
        "rg.ru",
        "forbes.ru",
    ],
    "official_intl": [
        "european-union.europa.eu",
        "ec.europa.eu",
        "consilium.europa.eu",
        "europarl.europa.eu",
        "europa.eu",
        "un.org",
        "imf.org",
        "worldbank.org",
        "nato.int",
        "schengenvisainfo.com",
    ],
    "science_med_global": [
        "nature.com",
        "science.org",
        "thelancet.com",
        "nejm.org",
        "who.int",
        "cdc.gov",
        "nih.gov",
        "scientificamerican.com",
    ],
    "science_med_ru": [
        "nplus1.ru",
        "cyberleninka.ru",
        "elementy.ru",
        "nauka.tass.ru",
        "minzdrav.gov.ru",
    ],
    "tech_crypto_global": [
        "techcrunch.com",
        "wired.com",
        "theverge.com",
        "coindesk.com",
        "cointelegraph.com",
        "technologyreview.com",
    ],
    "tech_crypto_ru": [
        "habr.com",
        "3dnews.ru",
        "ixbt.com",
        "vc.ru",
    ],
}

# Плоское множество всех доверенных доменов — для быстрого поиска O(1)
_ALL_TRUSTED_DOMAINS: set = {
    domain
    for domains in TRUSTED_SOURCES.values()
    for domain in domains
}

# Строгий чёрный список: UGC, соцсети, форумы
BLACKLIST_DOMAINS: List[str] = [
    "reddit.com",
    "quora.com",
    "github.com",
    "stackoverflow.com",
    "vk.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "instagram.com",
    "pikabu.ru",
    "dzen.ru",
    "livejournal.com",
    "youtube.com",
    "t.me",
    "telegram.org",
    "medium.com",
    "substack.com",
    "wordpress.com",
    "blogger.com",
    "tumblr.com",
    "pinterest.com",
    "linkedin.com",
]
_BLACKLIST_SET: set = set(BLACKLIST_DOMAINS)

# Доверенные TLD: .gov / .mil / .edu — всегда оставляем
TRUSTED_TLDS: List[str] = [".gov", ".mil", ".edu"]

# Паттерны URL, характерные для форумов и блогов (UGC-мусор)
_FORUM_BLOG_PATH_PATTERNS: List[str] = [
    "/forum/", "/forums/", "/community/", "/thread/", "/threads/",
    "/discuss/", "/discussion/", "/blog/", "/blogs/",
    "/user/", "/users/", "/profile/", "/profiles/",
    "/comment/", "/comments/", "/reply/", "/replies/",
    "/question/", "/questions/", "/answer/", "/answers/",
    "/topic/", "/topics/", "/post/", "/posts/",
]

# Маппинг: категория → ключи в TRUSTED_SOURCES
_CATEGORY_SOURCE_KEYS: Dict[str, List[str]] = {
    "science":    ["science_med_global",   "science_med_ru"],
    "politics":   ["news_agencies_global", "news_agencies_ru"],
    "technology": ["tech_crypto_global",   "tech_crypto_ru"],
    "crypto":     ["tech_crypto_global",   "tech_crypto_ru"],
    "general":    ["news_agencies_global", "news_agencies_ru"],
}


# ============================================================
# ШАГ 2: МАРШРУТИЗАТОР ЗАПРОСОВ (QueryClassifier)
#
# Классификация топика (science/politics/technology/general):
#   1. Приоритет — локальный Mistral (inject через set_generate_fn)
#   2. Fallback — regex по целым словам (границы \b, не substring)
#
# Перевод RU→EN:
#   Helsinki-NLP/opus-mt-ru-en (MarianMT, ~300MB, CPU, lazy load)
#   Fallback — оригинальный запрос (DDG понимает кириллицу)
#
# Никаких внешних API. Всё локально.
# ============================================================

# Промпт для Mistral — классификация topic в 1 токен, быстро
_CLASSIFY_PROMPT = """\
[INST]Classify the topic of this claim. Answer with ONLY one word from the list.

Topics:
- science  (medicine, health, biology, physics, chemistry, vaccines, viruses)
- politics (government, war, elections, military, sanctions, diplomacy)
- technology (tech, crypto, AI, software, blockchain, space, gadgets)
- general  (economy, culture, sport, celebrity, business, other)

Claim: "{claim}"

Topic:[/INST]"""

_VALID_TOPICS = {"science", "politics", "technology", "general"}


class QueryClassifier:
    """Классифицирует утверждение по теме и переводит на английский.

    Классификация:
      - Если передана generate_fn (Mistral через inject_model) → 1-токенный ответ модели
      - Иначе → regex-классификатор с границами слов (не substring)

    Перевод RU→EN:
      - Helsinki-NLP/opus-mt-ru-en, CPU, lazy load
      - Fallback: оригинал

    inject_model(fn) вызывается из FactCheckSearcher.set_generate_fn()
    уже после загрузки Mistral в pipeline.py.
    """

    _MARIAN_MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"

    def __init__(self):
        self._generate_fn  = None   # callable: str → str (Mistral)
        self._marian_tok   = None
        self._marian_model = None
        self._marian_ready = False

    # ----------------------------------------------------------
    # Инъекция Mistral (вызывается из FactCheckSearcher)
    # ----------------------------------------------------------

    def inject_model(self, generate_fn):
        """Принимает callable(prompt: str) -> str — уже загруженный Mistral.

        Вызывать после загрузки модели в pipeline.py:
            searcher._classifier.inject_model(lambda p: llm.invoke(p))
        """
        self._generate_fn = generate_fn
        print("  [QueryClassifier] Mistral подключён для классификации ✓")

    # ----------------------------------------------------------
    # Публичный метод
    # ----------------------------------------------------------

    def classify(self, claim: str) -> Dict[str, str]:
        """Возвращает {"category": str, "ru_query": str, "en_query": str}."""
        category = self._classify_topic(claim)
        en_query = self._translate(claim)
        return {
            "category": category,
            "ru_query": claim,
            "en_query": en_query,
        }

    # ----------------------------------------------------------
    # Классификация топика
    # ----------------------------------------------------------

    def _classify_topic(self, claim: str) -> str:
        """Определяет тему. Mistral → regex fallback."""
        if self._generate_fn is not None:
            return self._classify_with_mistral(claim)
        return self._classify_with_regex(claim)

    def _classify_with_mistral(self, claim: str) -> str:
        """1-токенная классификация через уже загруженный Mistral."""
        try:
            prompt = _CLASSIFY_PROMPT.format(claim=claim[:300])
            raw = self._generate_fn(prompt)
            # Берём первое слово из ответа, приводим к нижнему регистру
            word = raw.strip().split()[0].lower().strip(".,:")
            if word in _VALID_TOPICS:
                return word
            # Если модель вернула что-то другое — ищем topic в первых 30 символах
            for topic in _VALID_TOPICS:
                if topic in raw.lower()[:30]:
                    return topic
        except Exception as e:
            print(f"  [QueryClassifier] Mistral ошибка: {e} — regex fallback")
        return self._classify_with_regex(claim)

    def _classify_with_regex(self, claim: str) -> str:
        """Regex-классификатор с границами слов — не substring matching.

        Паттерны скомпилированы один раз при первом вызове.
        """
        if not hasattr(self, "_re_science"):
            self._re_science = re.compile(
                r"\b(вакцин\w*|вирус\w*|covid|ковид\w*|болезн\w*|лечени\w*|"
                r"медицин\w*|учёный|ученый|исследован\w*|наук[аие]\w*|научн\w*|"
                r"клиническ\w*|биолог\w*|химия|физика|генет\w*|пандемия|эпидемия|"
                r"минздрав|вакцинац\w*|мутация|штамм\w*|антител\w*|днк|рнк|"
                r"онколог\w*|фармацевт\w*|лекарств\w*|who|воз)\b",
                re.IGNORECASE,
            )
            self._re_tech = re.compile(
                r"\b(биткоин|bitcoin|крипт\w*|crypto|ethereum|блокчейн|blockchain|"
                r"spacex|tesla|apple|google|microsoft|технолог\w*|искусственн\w*|"
                r"нейросет\w*|программ\w*|хакер\w*|кибер\w*|стартап|процессор\w*|"
                r"смартфон\w*|криптовалют\w*|майнинг|токен\w*|nft|искусств\w* интеллект|"
                r"\bии\b|\bai\b)\b",
                re.IGNORECASE,
            )
            self._re_politics = re.compile(
                r"\b(президент\w*|министр\w*|парламент\w*|выбор\w*|санкци\w*|"
                r"войн\w*|конфликт\w*|нато|nato|украин\w*|госдум\w*|закон\w*|"
                r"правительств\w*|арми\w*|войска|войск\w*|удар\w*|обстрел\w*|"
                r"мобилизац\w*|дипломат\w*|посол\w*|саммит\w*|оон|un\b|"
                r"байден|путин|трамп|зеленск\w*|"
                r"цб\s+рф|центральн\w*\s+банк|ключев\w*\s+ставк|"
                r"ставк\w*\s+цб|минфин|госбюджет|федеральн\w*\s+резерв|"
                r"санкци\w*|эмбарго|национализац\w*)\b",
                re.IGNORECASE,
            )

        if self._re_science.search(claim):
            return "science"
        if self._re_tech.search(claim):
            return "technology"
        if self._re_politics.search(claim):
            return "politics"
        return "general"

    # ----------------------------------------------------------
    # Перевод через Helsinki-NLP/opus-mt-ru-en (MarianMT, CPU)
    # ----------------------------------------------------------

    def _load_marian(self):
        """Lazy-загрузка MarianMT. Вызывается один раз."""
        if self._marian_ready:
            return
        self._marian_ready = True
        try:
            from transformers import MarianMTModel, MarianTokenizer
            import torch
            print(f"  [Translator] Загрузка {self._MARIAN_MODEL_NAME} (CPU)...")
            self._marian_tok = MarianTokenizer.from_pretrained(self._MARIAN_MODEL_NAME)
            self._marian_model = MarianMTModel.from_pretrained(
                self._MARIAN_MODEL_NAME,
                dtype=torch.float32,
            )
            self._marian_model.eval()
            print("  [Translator] Helsinki-NLP/opus-mt-ru-en загружен ✓")
        except Exception as e:
            print(f"  [Translator] Не удалось загрузить: {e}")
            print("  [Translator] Fallback: будем передавать оригинальный запрос")
            self._marian_tok   = None
            self._marian_model = None

    # Таблица раскрытия аббревиатур — ПОРЯДОК ВАЖЕН:
    # сначала составные (МВД РФ, ЦБ РФ), потом одиночные (РФ, ООН)
    # иначе "РФ" заменяется раньше "МВД РФ" → дублирование
    _ABBREV_MAP = [
        # --- Составные (обязательно первыми) ---
        (r"\bЦБ\s+РФ\b",          "Центральный банк России"),
        (r"\bМВД\s+РФ\b",         "Министерство внутренних дел России"),
        (r"\bФСБ\s+РФ\b",         "Федеральная служба безопасности России"),
        (r"\bМИД\s+РФ\b",         "Министерство иностранных дел России"),
        (r"\bМинфин\s+РФ\b",      "Министерство финансов России"),
        (r"\bСБ\s+ООН\b",         "Совет Безопасности ООН"),
        # --- Одиночные ---
        (r"\bМВД\b",              "Министерство внутренних дел России"),
        (r"\bФСБ\b",              "Федеральная служба безопасности России"),
        (r"\bМИД\b",              "Министерство иностранных дел России"),
        (r"\bМинфин\b",           "Министерство финансов России"),
        (r"\bРФ\b",               "Россия"),
        (r"\bВОЗ\b",              "Всемирная организация здравоохранения"),
        (r"\bГосдума\b",          "Государственная дума России"),
        (r"\bЦИК\b",              "Центральная избирательная комиссия России"),
        (r"\bСВО\b",              "специальная военная операция"),
        (r"\bКНДР\b",             "Северная Корея"),
    ]

    def _expand_abbreviations(self, text: str) -> str:
        """Раскрывает русские аббревиатуры перед переводом.
        Порядок: составные аббревиатуры → одиночные.
        """
        result = text
        for pattern, replacement in self._ABBREV_MAP:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

    def _translate(self, text: str) -> str:
        """Переводит RU→EN. Ограничение входа: 120 символов (поисковый запрос).
        Раскрывает аббревиатуры перед переводом для лучшего качества.
        При ошибке возвращает оригинальный текст.
        """
        self._load_marian()

        # Раскрываем аббревиатуры даже если MarianMT недоступен
        expanded = self._expand_abbreviations(text)

        if self._marian_model is None:
            return expanded  # Хотя бы аббревиатуры раскрыты

        try:
            import torch
            inputs = self._marian_tok(
                [expanded[:120]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            with torch.no_grad():
                ids = self._marian_model.generate(**inputs, max_new_tokens=80, num_beams=4)
            return self._marian_tok.decode(ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"  [Translator] Ошибка перевода: {e}")
            return expanded


# ============================================================
# ШАГ 3: ГЕНЕРАТОР DDG-ЗАПРОСОВ С SITE: ОПЕРАТОРАМИ
# ============================================================

def _build_site_filter(domains: List[str], max_sites: int = 6) -> str:
    """Строит строку 'site:a.com OR site:b.com ...' из списка доменов.

    Берём только базовый домен (без пути) и ограничиваем max_sites,
    чтобы не превысить длину запроса и не снизить качество выдачи DDG.
    """
    seen: set = set()
    parts: List[str] = []
    for d in domains:
        base = d.split("/")[0]  # убираем путь вида reuters.com/fact-check → reuters.com
        if base not in seen:
            seen.add(base)
            parts.append(f"site:{base}")
        if len(parts) >= max_sites:
            break
    return " OR ".join(parts)


def build_search_queries(classified: Dict[str, str]) -> List[Dict[str, str]]:
    """Формирует 3 параллельных DDG-запроса по матрице источников.

    Args:
        classified: результат QueryClassifier.classify()

    Returns:
        Список словарей: [{"query": str, "label": str, "priority": int}, ...]

    Три запроса:
      1. factcheckers — ищем разоблачения на EN-запрос
      2. ru_sources   — российские СМИ + профильная категория (RU-запрос)
      3. global_sources — мировые СМИ + профильная категория (EN-запрос)
    """
    category = classified.get("category", "general")
    ru_q = classified.get("ru_query", "")
    en_q = classified.get("en_query", ru_q)

    # Определяем специфичные для категории ключи
    cat_keys = _CATEGORY_SOURCE_KEYS.get(category, _CATEGORY_SOURCE_KEYS["general"])
    cat_global_key = next((k for k in cat_keys if "global" in k), "news_agencies_global")
    cat_ru_key = next((k for k in cat_keys if "ru" in k), "news_agencies_ru")

    # Запрос 1: Фактчекеры (все глобальные + российские)
    fact_domains = (
        TRUSTED_SOURCES["factcheckers_global"]
        + TRUSTED_SOURCES["factcheckers_ru"]
    )
    q1 = f'{en_q} {_build_site_filter(fact_domains, max_sites=7)}'

    # Запрос 2: Российские источники + профильная категория RU
    ru_extra = TRUSTED_SOURCES.get(cat_ru_key, [])
    ru_all = list(dict.fromkeys(TRUSTED_SOURCES["news_agencies_ru"] + ru_extra))
    q2 = f'{ru_q} {_build_site_filter(ru_all, max_sites=7)}'

    # Запрос 3: Мировые источники + профильная категория Global
    global_extra = TRUSTED_SOURCES.get(cat_global_key, [])
    global_all = list(dict.fromkeys(TRUSTED_SOURCES["news_agencies_global"] + global_extra))
    q3 = f'{en_q} {_build_site_filter(global_all, max_sites=7)}'

    return [
        {"query": q1, "label": "factcheckers",   "priority": 1},
        {"query": q2, "label": "ru_sources",      "priority": 2},
        {"query": q3, "label": "global_sources",  "priority": 3},
    ]


# ============================================================
# ШАГ 4: ЖЁСТКИЙ ПАРСЕР URL (HARD FILTER PIPELINE)
# ============================================================

def _extract_base_domain(url: str) -> str:
    """Извлекает базовый домен из URL без www-префикса.

    'https://www.reuters.com/fact-check/story' → 'reuters.com'
    """
    try:
        netloc = urllib.parse.urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def _domain_in_trusted(domain: str) -> bool:
    """Проверяет домен (и его родительские домены) в матрице TRUSTED_SOURCES.

    'nauka.tass.ru' → 'tass.ru' → True
    """
    if domain in _ALL_TRUSTED_DOMAINS:
        return True
    # Проверяем родительские домены (поддомены)
    parts = domain.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[i:])
        if parent in _ALL_TRUSTED_DOMAINS:
            return True
    return False


def _has_trusted_tld(domain: str) -> bool:
    """Проверяет, заканчивается ли домен на .gov / .edu / .mil."""
    return any(domain.endswith(tld) for tld in TRUSTED_TLDS)


def _is_blacklisted(domain: str) -> bool:
    """Точная проверка: домен или его родитель в чёрном списке."""
    if domain in _BLACKLIST_SET:
        return True
    # Субдомены: 'old.reddit.com' → 'reddit.com' → True
    parts = domain.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[i:])
        if parent in _BLACKLIST_SET:
            return True
    return False


def _looks_like_ugc(url: str) -> bool:
    """Проверяет, похож ли URL-путь на форум/блог/UGC."""
    path = urllib.parse.urlparse(url).path.lower()
    return any(pat in path for pat in _FORUM_BLOG_PATH_PATTERNS)


def clean_results(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Фильтрует список статей по матрице доверенных источников.

    Логика (для каждого URL):
      1. Домен в BLACKLIST_DOMAINS → DROP
      2. TLD in (.gov, .edu, .mil) → KEEP
      3. Домен в TRUSTED_SOURCES → KEEP
      4. Неизвестный домен, без UGC-признаков → KEEP с флагом [UNVERIFIED_SOURCE]
      5. Неизвестный домен + UGC-признаки в пути → DROP

    Args:
        results: сырые статьи из DDG/SerpAPI

    Returns:
        Очищенный список. Недоверенные источники помечены флагом _unverified=True
        и префиксом [UNVERIFIED_SOURCE] в заголовке.
    """
    cleaned: List[Dict[str, str]] = []

    for article in results:
        url = article.get("link", "")
        if not url:
            continue

        domain = _extract_base_domain(url)
        if not domain:
            continue

        # 1. Чёрный список → DROP
        if _is_blacklisted(domain):
            print(f"    [Filter:BLACKLIST] {domain}")
            continue

        # 2. Доверенный TLD → KEEP
        if _has_trusted_tld(domain):
            print(f"    [Filter:TLD_KEEP] {domain}")
            cleaned.append(article)
            continue

        # 3. В матрице источников → KEEP
        if _domain_in_trusted(domain):
            print(f"    [Filter:TRUSTED] {domain}")
            cleaned.append(article)
            continue

        # 4. Неизвестный домен → DROP всегда (строгий режим)
        # Только доверенные источники и trusted TLD проходят фильтр.
        # Чешские sberdat.uiv.cz, вьетнамские ChatGPT-блоги и прочий мусор — сюда.
        print(f"    [Filter:DROP] {domain} — неизвестный домен")
        continue

    return cleaned


# ============================================================
# ШАГ 5: ЛОГИКА «НУЛЕВОЙ ВЫДАЧИ» (SILENCE AS EVIDENCE)
# ============================================================

ZERO_RESULTS_CONTEXT: str = (
    "СИСТЕМНОЕ СООБЩЕНИЕ: Поиск по базам фактчекеров (Snopes, Reuters Fact-Check, "
    "Provereno.media), мировым агентствам (Reuters, AP, Bloomberg, BBC) и российским "
    "СМИ (ТАСС, Интерфакс, РБК) дал 0 результатов. "
    "Для события такого масштаба полное молчание авторитетных СМИ — "
    "это признак выдумки, маргинальной теории или вирусного фейка. "
    "Чем сенсационнее утверждение, тем сильнее подозрение в его ложности."
)


# ============================================================
# ШАГ 6: LLM-СУДЬЯ (VERDICT AGENT)
# ============================================================

# Промпт для Mistral-вердикта в get_verdict (standalone режим)
# Компактный — max 350 токенов ответа, чистый JSON
_VERDICT_PROMPT = """\
[INST]Ты — главный редактор отдела фактчекинга. Проверь утверждение на основе источников.

УТВЕРЖДЕНИЕ: {query}

ИСТОЧНИКИ:
{context}

ПРАВИЛА:
1. Фактчекеры (Snopes, Reuters Fact-Check, Provereno.media) опровергают → FALSE (85-100).
2. Авторитетные СМИ молчат о сенсации → FALSE (70-80).
3. [UNVERIFIED_SOURCE] — слабое доказательство, используй со скепсисом.
4. Надёжные СМИ подтверждают → TRUE (75-100).
5. Данных мало → NEEDS_CONTEXT (40-60).

Ответь ТОЛЬКО валидным JSON без markdown:
{{"verdict": "TRUE" или "FALSE" или "NEEDS_CONTEXT", "confidence_score": число 0-100, "explanation": "3-4 предложения на русском", "trusted_sources_found": ["url1", "url2"]}}[/INST]"""


def get_verdict(
    query: str,
    cleaned_context: List[Dict[str, str]],
    model=None,
    tokenizer=None,
    generate_fn=None,
) -> Dict[str, Any]:
    """Standalone LLM-судья: выносит вердикт по утверждению.

    Три режима:
      1. generate_fn передан (callable str→str) → Mistral через готовый пайплайн
      2. model + tokenizer переданы → прямая генерация через transformers
      3. Ничего не передано → детерминистический анализ по структуре источников

    Args:
        query: утверждение для проверки
        cleaned_context: результат clean_results() — отфильтрованные статьи
        model: опционально — уже загруженная transformers модель
        tokenizer: опционально — tokenizer к model
        generate_fn: опционально — callable(prompt: str) -> str (напр. keyword_llm.invoke)

    Returns:
        {"verdict": "TRUE"|"FALSE"|"NEEDS_CONTEXT",
         "confidence_score": int,
         "explanation": str,
         "trusted_sources_found": List[str]}
    """
    # Формируем текстовый контекст для промпта
    context_text = _build_verdict_context(cleaned_context)

    # Режим 1 или 2: есть Mistral — используем
    if generate_fn is not None or (model is not None and tokenizer is not None):
        return _verdict_with_mistral(query, context_text, generate_fn, model, tokenizer)

    # Режим 3: детерминистический анализ
    return _verdict_deterministic(query, cleaned_context)


def _build_verdict_context(cleaned_context: List[Dict[str, str]]) -> str:
    """Формирует текст контекста для LLM-судьи."""
    if not cleaned_context:
        return ZERO_RESULTS_CONTEXT
    lines: List[str] = [f"Найдено {len(cleaned_context)} источников:\n"]
    for i, art in enumerate(cleaned_context[:7], 1):
        title   = art.get("title", "Без заголовка")
        snippet = art.get("snippet", "")[:350]
        source  = art.get("source", art.get("link", ""))
        link    = art.get("link", "")
        date    = art.get("date", "")
        lines.append(f"{i}. {title}")
        if source:
            lines.append(f"   Источник: {source}")
        if date:
            lines.append(f"   Дата: {date}")
        if snippet:
            lines.append(f"   {snippet}")
        if link:
            lines.append(f"   URL: {link}")
        lines.append("")
    return "\n".join(lines)


def _verdict_with_mistral(
    query: str,
    context_text: str,
    generate_fn,
    model,
    tokenizer,
) -> Dict[str, Any]:
    """Вердикт через локальный Mistral. Парсит JSON из ответа."""
    prompt = _VERDICT_PROMPT.format(
        query=query[:400],
        context=context_text[:2000],
    )
    try:
        # Режим 1: generate_fn (уже обёрнутый HuggingFacePipeline.invoke)
        if generate_fn is not None:
            raw = generate_fn(prompt)
        else:
            # Режим 2: прямая генерация через transformers
            import torch
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                ids = model.generate(
                    **inputs,
                    max_new_tokens=350,
                    do_sample=False,
                    repetition_penalty=1.1,
                )
            raw = tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return _parse_verdict_json(raw)

    except Exception as e:
        print(f"  [get_verdict] Mistral ошибка: {e} → детерминистический fallback")
        # Восстанавливаем cleaned_context из context_text не получится,
        # поэтому возвращаем минимальный детерминистический результат
        return {
            "verdict": "NEEDS_CONTEXT",
            "confidence_score": 30,
            "explanation": f"Ошибка LLM-судьи: {e}. Результат требует ручной проверки.",
            "trusted_sources_found": [],
        }


def _parse_verdict_json(raw: str) -> Dict[str, Any]:
    """Надёжный парсер JSON из ответа Mistral.

    Обрабатывает:
    - Чистый JSON: {"verdict": ...}
    - Markdown-обёртка: ```json {...} ```
    - JSON после пояснительного текста: "Анализ показывает... {"verdict":...}"
    - Полный провал: ищем ключевые слова в тексте
    """
    _valid_verdicts = {"TRUE", "FALSE", "NEEDS_CONTEXT"}

    # 1. Убираем markdown-блоки (``` и ```json)
    clean = re.sub(r"```(?:json)?\s*", "", raw)
    clean = re.sub(r"```\s*", "", clean).strip()

    # 2. Ищем САМЫЙ БОЛЬШОЙ JSON-объект (жадный поиск от { до последнего })
    #    Это корректно обрабатывает вложенные списки внутри JSON
    m = re.search(r"\{[\s\S]*\}", clean)
    if m:
        candidate = m.group(0)
        try:
            result = json.loads(candidate)
            verdict = str(result.get("verdict", "NEEDS_CONTEXT")).strip().upper()
            if verdict not in _valid_verdicts:
                verdict = "NEEDS_CONTEXT"
            return {
                "verdict": verdict,
                "confidence_score": max(0, min(100, int(result.get("confidence_score", 50)))),
                "explanation": str(result.get("explanation", "")),
                "trusted_sources_found": list(result.get("trusted_sources_found", [])),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Последний fallback: ищем ключевые слова в тексте
    raw_up = raw.upper()
    # Проверяем наличие "FALSE" прежде "NEEDS_CONTEXT" (т.к. он длиннее)
    if '"VERDICT": "FALSE"' in raw_up or "'VERDICT': 'FALSE'" in raw_up:
        verdict, conf = "FALSE", 60
    elif '"VERDICT": "TRUE"' in raw_up or "'VERDICT': 'TRUE'" in raw_up:
        verdict, conf = "TRUE", 60
    elif "FALSE" in raw_up and "TRUE" not in raw_up:
        verdict, conf = "FALSE", 50
    elif "TRUE" in raw_up and "FALSE" not in raw_up:
        verdict, conf = "TRUE", 50
    else:
        verdict, conf = "NEEDS_CONTEXT", 40

    return {
        "verdict": verdict,
        "confidence_score": conf,
        "explanation": raw[:500] if raw else "Не удалось разобрать ответ модели.",
        "trusted_sources_found": [],
    }


def _verdict_deterministic(
    query: str,
    cleaned_context: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Детерминистический вердикт без LLM — анализ структуры источников.

    Логика:
    - 0 источников → FALSE (молчание = признак фейка)
    - URL содержит путь /fact-check/ или домен — чистый фактчекер → FALSE с высоким confidence
    - Много доверенных источников → NEEDS_CONTEXT (финальный вердикт за основным pipeline)
    - Только UNVERIFIED → NEEDS_CONTEXT с низким confidence
    """
    # Только чистые фактчекеры — не включаем reuters/apnews (они дублируются в news_agencies)
    _pure_factcheckers = set(
        TRUSTED_SOURCES["factcheckers_ru"] + [
            "snopes.com", "politifact.com", "fullfact.org",
            "leadstories.com", "factcheck.org",
        ]
    )
    # Плюс ловим путь /fact-check/ на крупных агентствах
    _factcheck_path_markers = ["/fact-check", "/fact_check", "/factcheck", "/razoblachenie"]

    if not cleaned_context:
        return {
            "verdict": "FALSE",
            "confidence_score": 65,
            "explanation": ZERO_RESULTS_CONTEXT,
            "trusted_sources_found": [],
        }

    trusted_arts   = [a for a in cleaned_context if not a.get("_unverified")]
    unverified_arts = [a for a in cleaned_context if a.get("_unverified")]
    trusted_urls   = [a.get("link", "") for a in trusted_arts if a.get("link")]

    # Проверяем наличие фактчекеров (чистые домены + /fact-check/ пути)
    factcheck_hits = [
        u for u in trusted_urls
        if any(fc in _extract_base_domain(u) for fc in _pure_factcheckers)
        or any(marker in u.lower() for marker in _factcheck_path_markers)
    ]

    if factcheck_hits:
        return {
            "verdict": "FALSE",
            "confidence_score": 80,
            "explanation": (
                f"Найдены материалы фактчекеров ({len(factcheck_hits)} источника). "
                "Присутствие фактчекерских публикаций по данному запросу — "
                "признак того, что утверждение разбиралось и опровергалось. "
                "Рекомендуется просмотреть найденные материалы."
            ),
            "trusted_sources_found": factcheck_hits[:5],
        }

    if not trusted_arts:
        return {
            "verdict": "NEEDS_CONTEXT",
            "confidence_score": 25,
            "explanation": (
                f"Найдено {len(unverified_arts)} источников, но все из неизвестных доменов "
                "[UNVERIFIED_SOURCE]. Авторитетные СМИ по данной теме не найдены. "
                "Это слабый сигнал достоверности."
            ),
            "trusted_sources_found": [],
        }

    # Контентный анализ заголовков и сниппетов — ищем противоречия с claim
    content_signal = _scan_titles_for_contradiction(query, trusted_arts)
    if content_signal:
        return {
            "verdict": "FALSE",
            "confidence_score": content_signal["confidence"],
            "explanation": content_signal["explanation"],
            "trusted_sources_found": trusted_urls[:5],
        }

    return {
        "verdict": "NEEDS_CONTEXT",
        "confidence_score": 45,
        "explanation": (
            f"Найдено {len(trusted_arts)} доверенных источников и "
            f"{len(unverified_arts)} непроверенных. "
            "Для финального вердикта необходим анализ содержимого источников через LLM. "
            "Передайте model + tokenizer в get_verdict() для полного анализа Mistral."
        ),
        "trusted_sources_found": trusted_urls[:5],
    }


def _scan_titles_for_contradiction(
    query: str,
    trusted_arts: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """Сканирует заголовки и сниппеты источников на противоречия с claim.

    Ищет конкретные паттерны опровержения:
    - Подмена города: claim "в Токио" → источники говорят "Paris" / "в Париже"
    - Подмена роли/персоны: claim "Цукерберг CEO Tesla" → источники "Musk's Tesla"
    - Подмена страны/организации: известные пары
    - Явное отрицание: "did not", "не является", "ban" при claim "признал"

    Возвращает dict с confidence и explanation если нашли противоречие, иначе None.
    """
    query_lower = query.lower()
    # Собираем весь текст источников (заголовки + сниппеты)
    source_text = " ".join(
        (a.get("title", "") + " " + a.get("snippet", "")).lower()
        for a in trusted_arts[:10]
    )

    # --- Пары подмены городов/стран ---
    _location_pairs = [
        # (что в claim, что в источниках → опровержение)
        (["токио", "tokyo"],  ["paris", "париж", "paris 2024"]),
        (["париж", "paris"],  ["tokyo", "токио", "tokyo 2024"]),
        (["москва", "moscow"], ["киев", "kyiv", "kiev"]),
        (["лондон", "london"], ["paris", "париж", "berlin", "берлин"]),
        (["пекин", "beijing"], ["tokyo", "токио", "paris", "париж"]),
    ]
    for claim_locs, contra_locs in _location_pairs:
        if any(loc in query_lower for loc in claim_locs):
            contra_hits = [loc for loc in contra_locs if loc in source_text]
            if contra_hits:
                claim_loc = next(l for l in claim_locs if l in query_lower)
                return {
                    "confidence": 75,
                    "explanation": (
                        f"Утверждение указывает на локацию «{claim_loc}», "
                        f"однако источники Reuters/AP/BBC упоминают «{contra_hits[0]}». "
                        "Это признак подмены места события."
                    ),
                }

    # --- Подмена персоны-роли ---
    _role_pairs = [
        # (персона в claim, компания в claim, реальный владелец роли в источниках)
        (["цукерберг", "zuckerberg"], ["tesla"], ["musk", "маск", "elon"]),
        (["маск", "musk", "илон"],    ["meta", "facebook", "instagram"], ["zuckerberg", "цукерберг"]),
        (["байден", "biden"],          ["tesla", "spacex", "microsoft"], ["musk", "маск", "gates"]),
        (["трамп", "trump"],           ["apple", "google", "amazon"],    ["cook", "bezos", "pichai"]),
    ]
    for claim_persons, claim_companies, real_owners in _role_pairs:
        has_person  = any(p in query_lower for p in claim_persons)
        has_company = any(c in query_lower for c in claim_companies)
        if has_person and has_company:
            owner_hits = [o for o in real_owners if o in source_text]
            if owner_hits:
                person = next(p for p in claim_persons if p in query_lower)
                company = next(c for c in claim_companies if c in query_lower)
                return {
                    "confidence": 80,
                    "explanation": (
                        f"Утверждение приписывает роль «{person}» в компании «{company}». "
                        f"Источники упоминают реального руководителя «{owner_hits[0]}». "
                        "Это классическая подмена персоны."
                    ),
                }

    # --- Явные отрицания в источниках при утверждении recognition/acceptance ---
    _affirmative_kw = ["признал", "одобрил", "легализовал", "разрешил", "подтвердил", "принял"]
    _negation_signals = ["ban", "banned", "banning", "reject", "prohibit", "not legal",
                         "запрет", "не признал", "заблокировал", "отверг", "but ban"]
    if any(kw in query_lower for kw in _affirmative_kw):
        neg_hits = [s for s in _negation_signals if s in source_text]
        if len(neg_hits) >= 2:  # Минимум 2 сигнала чтобы избежать ложных срабатываний
            return {
                "confidence": 70,
                "explanation": (
                    f"Утверждение говорит о принятии/признании, "
                    f"однако источники содержат сигналы запрета/отказа: {neg_hits[:3]}. "
                    "Возможно утверждение искажает реальное решение."
                ),
            }

    return None


# ============================================================
# ШАГ 7: POST-CROSSENCODER БУСТИНГ ФАКТЧЕК-ИСТОЧНИКОВ (Direction 1)
# ============================================================

# Домены, специализирующиеся на разоблачениях — строгий список
FACTCHECK_DOMAINS: frozenset = frozenset(
    TRUSTED_SOURCES["factcheckers_global"] + TRUSTED_SOURCES["factcheckers_ru"]
)

# Слова-маркеры разоблачения в заголовках/сниппетах
DEBUNK_KEYWORDS: List[str] = [
    # Русские
    "миф", "мифы", "разоблач", "фейк", "дезинформ", "неправда",
    "заблуждение", "провер", "опроверг", "опровержение", "недостоверн",
    # Английские
    "debunk", "myth", "false claim", "mislead", "fact-check", "fact check",
    "hoax", "disinformation", "misinformation", "busted",
]

# Паттерны URL, типичные для фактчек-материалов
_FACTCHECK_PATH_MARKERS: List[str] = [
    "/fact-check", "/fact_check", "/factcheck", "/razoblachenie",
    "/myths", "/hoax", "/debunk", "/checking",
]

FACTCHECK_BOOST: float = 2.0  # прибавляется к cross_encoder_score


def _domain_in_factcheckers(domain: str) -> bool:
    """Проверяет, является ли домен специализированным фактчекером."""
    if domain in FACTCHECK_DOMAINS:
        return True
    parts = domain.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[i:])
        if parent in FACTCHECK_DOMAINS:
            return True
    return False


def boost_factcheck_scores(
    ranked_docs: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Post-CrossEncoder бустинг: поднимает фактчек-источники в топ выдачи.

    Вызывать ПОСЛЕ cross_encoder.rerank() и ДО склейки контекста для LLM.
    Решает проблему Semantic Overgeneralization: общие новостные статьи получают
    высокий CrossEncoder-скор из-за лексического перекрытия, вытесняя специализированные
    статьи-разоблачения, которые аппаратно опровергают конкретный миф.

    Критерии бустинга (любой из):
      1. Домен — фактчекер из FACTCHECK_DOMAINS (snopes, provereno, politifact, …)
      2. Заголовок/сниппет содержит DEBUNK_KEYWORDS (миф, разоблачение, debunk, …)
      3. URL содержит паттерн фактчек-пути (/fact-check/, /debunk/, …)

    После бустинга выполняется финальная пересортировка.

    Args:
        ranked_docs: список документов с полем cross_encoder_score (или semantic_score)

    Returns:
        Пересортированный список: фактчек-источники гарантированно вытесняют новости.
    """
    boosted_count = 0
    for doc in ranked_docs:
        url = doc.get("link", "")
        domain = _extract_base_domain(url)
        text = (doc.get("title", "") + " " + doc.get("snippet", "")).lower()

        is_fc_domain = _domain_in_factcheckers(domain)
        has_debunk_kw = any(kw in text for kw in DEBUNK_KEYWORDS)
        has_fc_path = any(marker in url.lower() for marker in _FACTCHECK_PATH_MARKERS)

        if is_fc_domain or has_debunk_kw or has_fc_path:
            current = doc.get("cross_encoder_score", doc.get("semantic_score", 0.0))
            doc["cross_encoder_score"] = current + FACTCHECK_BOOST
            boosted_count += 1

    if boosted_count:
        print(f"  [FC-Boost] +{FACTCHECK_BOOST:.1f} к скору для {boosted_count} фактчек-источников")

    # Финальная пересортировка — разоблачения вытесняют лайфстайл-новости на Топ-1
    ranked_docs.sort(
        key=lambda x: x.get("cross_encoder_score", x.get("semantic_score", 0.0)),
        reverse=True,
    )
    return ranked_docs


# ============================================================
# ШАГ 8: ENTITY MATCHING — ПРЕД-ФИЛЬТР КОНТЕКСТА (Direction 2)
# ============================================================

# Стоп-слова для entity extraction: частицы, предлоги, союзы, местоимения
_ENTITY_STOPWORDS: frozenset = frozenset([
    # Русские
    "что", "как", "это", "есть", "был", "была", "было", "были",
    "его", "её", "их", "также", "при", "для", "она", "они", "он",
    "который", "которая", "которое", "которые", "себя", "себе",
    "свой", "своя", "своё", "свои", "этот", "эта", "этих", "этим",
    "такой", "такая", "таком", "такие", "всего", "всей", "всем",
    "когда", "если", "чтобы", "потому", "хотя", "после", "перед",
    "между", "через", "около", "кроме", "вместо", "среди", "против",
    "один", "одна", "одно", "одни", "года", "году", "лет", "раз",
    "мире", "мира", "страны", "страна", "город", "города", "стал",
    "стала", "имеет", "может", "будет", "было", "были", "иметь",
    "более", "менее", "очень", "уже", "ещё", "даже", "почти",
    "тоже", "также", "кроме", "затем", "потом", "теперь", "здесь",
    "всего", "самый", "самая", "самое", "самые", "каждый", "любой",
    # Английские
    "the", "and", "that", "this", "was", "were", "have", "been",
    "from", "with", "which", "their", "they", "about", "more",
    "also", "when", "what", "who", "how", "his", "her", "its",
    "will", "would", "could", "should", "does", "did", "not",
    "but", "for", "are", "has", "had", "been", "than", "into",
])


def validate_context_entities(
    atomic_fact: str,
    context_string: str,
) -> float:
    """Оценивает покрытие уникальных сущностей из факта в контексте источника.

    Решает проблему Semantic Overgeneralization: если в факте слово «математика»,
    а в контексте только «вступительные экзамены» — это другой факт, и модель
    должна получать пенализированный контекст, а не вводиться в заблуждение.

    Алгоритм:
      1. Извлекаем значимые слова из atomic_fact (длина ≥ 4, не стоп-слово)
      2. Для каждого слова проверяем его 5-char псевдо-стем в context_string
         (перекрывает русские падежные формы: математика → матем)
      3. Coverage = доля покрытых слов
      4. Penalty = 1.0 - coverage

    Args:
        atomic_fact: атомарное утверждение (напр. «провалил экзамен по математике»)
        context_string: заголовок + сниппет источника

    Returns:
        penalty: 0.0 = нет штрафа (полное покрытие),
                 1.0 = максимальный штраф (контекст не содержит ни одной сущности).
        Рекомендация: при penalty > 0.5 — сильно пенализировать или отбросить контекст.
    """
    fact_lower = atomic_fact.lower()
    fact_words = re.findall(r"[а-яёА-ЯЁa-zA-Z]{4,}", fact_lower)
    significant = [w for w in fact_words if w not in _ENTITY_STOPWORDS]

    if not significant:
        return 0.0  # нечего проверять

    context_lower = context_string.lower()
    covered = sum(
        1 for word in significant
        if word[:5] in context_lower  # псевдо-стем 5 символов
    )

    coverage = covered / len(significant)
    return max(0.0, 1.0 - coverage)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ (обратная совместимость)
# ============================================================

class RateLimiter:
    """Ограничение частоты запросов к внешним API."""

    def __init__(self, calls_per_second: float = 0.5):
        self.interval = 1.0 / calls_per_second
        self.last_call = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()


# Пороги для bag-of-words fallback (если sentence-transformers недоступен)
RELEVANCE_THRESHOLD = 0.35
RELATED_THRESHOLD   = 0.20


def _simple_tokenize(text: str) -> Dict[str, int]:
    """Токенизация с псевдо-стеммингом для русского языка (fallback)."""
    words = text.lower().split()
    freq: Dict[str, int] = {}
    for w in words:
        w = w.strip(".,!?;:\"'()-[]{}»«")
        if len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
            if len(w) > 4:
                stem = w[:4]
                freq[stem] = freq.get(stem, 0) + 1
    return freq


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Bag-of-words косинусное сходство с псевдо-стеммингом (fallback)."""
    va = _simple_tokenize(text_a)
    vb = _simple_tokenize(text_b)
    if not va or not vb:
        return 0.0
    common = set(va) & set(vb)
    dot = sum(va[k] * vb[k] for k in common)
    na = sum(v ** 2 for v in va.values()) ** 0.5
    nb = sum(v ** 2 for v in vb.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ============================================================
# ОСНОВНОЙ КЛАСС: FactCheckSearcher (улучшенный v3)
# ============================================================

class FactCheckSearcher:
    """Поисковый движок для фактчекинга с маршрутизацией по источникам.

    v3: routing pipeline + жёсткий URL-фильтр + обратная совместимость.

    Порядок поиска в search_all_keywords():
      1. Routing pipeline (QueryClassifier → 3 DDG-запроса с site: → clean_results)
      2. Wikipedia (веб-факты, всегда для claim)
      3. Контрастивный поиск (X — Y → ищем X для нахождения реального Y)
      4. Fallback: ключевые слова без фильтрации (только если результатов < 3)

    API совместим с pipeline.py: search_all_keywords / rank_by_relevance / format_results.
    """

    def __init__(self, config: SearchConfig = None):
        if config is None:
            config = SearchConfig()
        self.config = config
        self._serpapi_failed = False
        self._cache = SearchCache()
        self._rate_limiter = RateLimiter(calls_per_second=0.5)

        # LLM-маршрутизатор: MarianMT перевод + Mistral классификация (если подключён)
        # Mistral подключается позже через set_generate_fn() из pipeline.py
        self._classifier = QueryClassifier()

        # Семантический ранкер (опциональный)
        self._semantic_ranker = None
        try:
            from embeddings import get_ranker
            self._semantic_ranker = get_ranker()
            print("Семантическое ранжирование: включено (sentence-transformers)")
        except ImportError:
            print("Семантическое ранжирование: выключено (pip install sentence-transformers)")

    # ----------------------------------------------------------
    # Внутренние методы поиска
    # ----------------------------------------------------------

    def _search_ddg(self, query: str, max_results: int = 8) -> List[Dict[str, str]]:
        """DDG поиск с exponential backoff (3 попытки: 2 / 4 / 8 сек)."""
        for attempt in range(3):
            try:
                raw = _DDGS(timeout=_DDG_TIMEOUT).text(
                    query.strip(),
                    max_results=max_results,
                    backend="html",   # детерминистичный бэкенд
                )
                articles: List[Dict[str, str]] = []
                for item in raw or []:
                    articles.append({
                        "title":   item.get("title", ""),
                        "snippet": item.get("body", item.get("snippet", "")),
                        "source":  item.get("source", ""),
                        "link":    item.get("url", item.get("href", "")),
                        "date":    item.get("date", ""),
                    })
                return articles
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"  [DDG] Попытка {attempt + 1}/3 неудачна: {e}. "
                          f"Ожидание {wait}с...")
                    time.sleep(wait)
                else:
                    print(f"  [DDG] Все 3 попытки неудачны: {e}")
                    return []
        return []

    def _search_serpapi(self, keyword: str, web_mode: bool = False) -> List[Dict[str, str]]:
        """SerpAPI Google поиск (платный, основной при наличии API-ключа)."""
        from serpapi import GoogleSearch

        params = {
            "q":       keyword.strip(),
            "engine":  "google",
            "gl":      self.config.gl,
            "hl":      self.config.hl,
            "num":     self.config.num_results,
            "api_key": self.config.api_key,
        }
        if not web_mode:
            params["tbm"] = self.config.tbm

        results = GoogleSearch(params).get_dict()
        articles: List[Dict[str, str]] = []

        for item in results.get("news_results", []):
            articles.append({
                "title":   item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "source":  item.get("source", ""),
                "link":    item.get("link", ""),
                "date":    item.get("date", ""),
            })
        for item in results.get("organic_results", []):
            articles.append({
                "title":   item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "source":  item.get("displayed_link", ""),
                "link":    item.get("link", ""),
                "date":    "",
            })
        return articles

    def _search_wikipedia(self, keyword: str, lang: str = "ru") -> List[Dict[str, str]]:
        """Wikipedia API поиск (публичный, без ключа)."""
        try:
            q = urllib.parse.quote(keyword.strip())
            headers = {"User-Agent": "AntifakeBot/3.0 (fact-checking)"}
            url = (
                f"https://{lang}.wikipedia.org/w/api.php"
                f"?action=query&list=search&srsearch={q}&format=json"
                f"&srlimit=3&srprop=snippet"
            )
            req  = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=8)
            data = json.loads(resp.read())
            hits = data.get("query", {}).get("search", [])

            articles: List[Dict[str, str]] = []
            for item in hits[:3]:
                title = item.get("title", "")
                raw_snip = item.get("snippet", "")
                snippet  = re.sub(r"<[^>]+>", "", raw_snip)
                snippet  = re.sub(r"\s+", " ", snippet).strip()
                link = (
                    f"https://{lang}.wikipedia.org/wiki/"
                    + urllib.parse.quote(title.replace(" ", "_"))
                )
                articles.append({
                    "title":   title,
                    "snippet": snippet,
                    "source":  f"{lang}.wikipedia.org",
                    "link":    link,
                    "date":    "",
                })
            return articles
        except Exception as e:
            print(f"  [Wikipedia] Ошибка: {e}")
            return []

    def set_generate_fn(self, generate_fn):
        """Подключает уже загруженный Mistral к маршрутизатору.

        Вызывать из pipeline.py после load_base_model / load_finetuned_model:
            # В FactCheckPipeline.__init__ после build_langchain_llm:
            self.searcher.set_generate_fn(lambda p: self.keyword_llm.invoke(p))

        generate_fn: callable(prompt: str) -> str
        """
        self._classifier.inject_model(generate_fn)

    # ----------------------------------------------------------
    # Новый метод: поиск с маршрутизацией
    # ----------------------------------------------------------

    def search_with_routing(self, claim: str) -> List[Dict[str, str]]:
        """Полный routing pipeline:
          1. QueryClassifier → категория + EN-перевод
          2. build_search_queries → 3 DDG-запроса с site: операторами
          3. DDG выполнение каждого запроса (с кэшем и rate limit)
          4. clean_results → жёсткая фильтрация по матрице источников

        Returns:
            Очищенный список статей. Недоверенные помечены _unverified=True.
        """
        # Шаг 2: Классификация
        classified = self._classifier.classify(claim)
        print(
            f"  [Router] Категория: {classified['category']} | "
            f"EN: {classified['en_query'][:70]}"
        )

        # Шаг 3: Генерация запросов и их выполнение
        queries = build_search_queries(classified)
        all_raw: List[Dict[str, str]] = []
        seen_urls: set = set()

        for q_info in queries:
            label = q_info["label"]
            query = q_info["query"]
            print(f"  [DDG:{label}] {query[:90]}...")

            cache_key = f"routed:{query}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                raw = cached
            else:
                self._rate_limiter.wait()
                raw = self._search_ddg(query, max_results=self.config.num_results)
                self._cache.set(cache_key, raw)

            print(f"  [DDG:{label}] Сырых результатов: {len(raw)}")

            for art in raw:
                url = art.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_raw.append(art)

        # Шаг 4: Жёсткая фильтрация
        cleaned = clean_results(all_raw)
        trusted_cnt   = sum(1 for a in cleaned if not a.get("_unverified"))
        unverified_cnt = sum(1 for a in cleaned if a.get("_unverified"))
        dropped = len(all_raw) - len(cleaned)
        print(
            f"  [Filter] {len(all_raw)} → {len(cleaned)} "
            f"(доверенных: {trusted_cnt}, непроверенных: {unverified_cnt}, "
            f"отброшено: {dropped})"
        )
        return cleaned

    # ----------------------------------------------------------
    # Методы обратной совместимости с pipeline.py
    # ----------------------------------------------------------

    def search_keyword(self, keyword: str, web_mode: bool = False) -> List[Dict[str, str]]:
        """Поиск одного запроса: кэш → SerpAPI → DDG (без routing-фильтрации).

        Используется для Wikipedia web_mode и как fallback в search_all_keywords.
        """
        cache_key = f"{'web' if web_mode else 'news'}:{keyword}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        self._rate_limiter.wait()

        wiki_results: List[Dict[str, str]] = []
        if web_mode:
            wiki_results = self._search_wikipedia(keyword, lang=self.config.hl)

        if self.config.api_key and not self._serpapi_failed:
            try:
                serpapi_results = self._search_serpapi(keyword, web_mode=web_mode)
                results = wiki_results + serpapi_results
                self._cache.set(cache_key, results)
                return results
            except Exception as e:
                print(f"  [SerpAPI] Ошибка: {e} — переключаюсь на DDG для всех запросов")
                self._serpapi_failed = True

        ddg_results = self._search_ddg(keyword)
        results = wiki_results + ddg_results
        self._cache.set(cache_key, results)
        return results

    def search_all_keywords(
        self, keywords: List[str], claim: str = ""
    ) -> List[Dict[str, str]]:
        """Поиск по всем ключевым словам с routing pipeline.

        Стратегия (приоритет убывает):
          1. Routing pipeline (3 DDG-запроса с site: + clean_results) — ОСНОВНОЙ
          2. Wikipedia — фактические веб-данные
          3. Контрастивный поиск — ловит подмену деталей (X — Y → ищем X)
          4. Fallback: ключевые слова напрямую — только при < 3 результатах

        Совместим с вызовами из pipeline.py.
        """
        seen_urls: set = set()
        all_results: List[Dict[str, str]] = []

        def _add(articles: List[Dict[str, str]]):
            for art in articles:
                url = art.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(art)
                elif url and "wikipedia.org" in url:
                    # Wikipedia: обновляем сниппет если новый длиннее
                    new_snip = art.get("snippet", "")
                    for ex in all_results:
                        if ex.get("link") == url:
                            if len(new_snip) > len(ex.get("snippet", "")):
                                ex["snippet"] = new_snip
                            break

        # 1. Routing pipeline (приоритетный блок)
        if claim.strip():
            _add(self.search_with_routing(claim))

        # 2. Wikipedia (веб-факты)
        # clean_results применяется и здесь — search_keyword возвращает DDG без фильтра
        if claim.strip():
            _add(clean_results(self.search_keyword(claim.strip()[:120], web_mode=True)))

        # 2.5. Myth-buster запросы: ищем разоблачения и мифы
        if claim.strip():
            short_claim = claim.strip()[:80]
            mythbuster_suffixes = [
                "миф OR разоблачение OR заблуждение",
                "fact check OR debunk OR фейк",
            ]
            mb_total_before = len(all_results)
            for suffix in mythbuster_suffixes:
                mb_query = f"{short_claim} {suffix}"
                mb_cleaned = clean_results(self._search_ddg(mb_query, max_results=5))

                # Direction 3: резервируем минимум 2–3 слота для фактчекеров.
                # Новостные сайты не должны вытеснять специализированные разоблачения
                # на этапе первичного сбора ссылок — до CrossEncoder.
                fc_hits = [
                    r for r in mb_cleaned
                    if _domain_in_factcheckers(_extract_base_domain(r.get("link", "")))
                ]
                non_fc = [r for r in mb_cleaned if r not in fc_hits]
                # Фактчекеры — первыми в пул (до 3 слотов), затем остальные
                _add(fc_hits[:3] + non_fc)

            mb_added = len(all_results) - mb_total_before
            print(f"  [Myth-buster] +{mb_added} результатов после myth-buster запросов")

        # 3. Контрастивный поиск: «X — Y» → ищем «X», находим реальный Y
        if claim.strip():
            _eq_seps = [" — ", " - ", " является ", " это ", " составляет "]
            _event_seps = [
                " прошли в ", " прошёл в ", " прошла в ",
                " состоялись в ", " состоялся в ", " состоялась в ",
                " проходили в ", " проходил в ", " проходила в ",
                " пройдёт в ", " пройдут в ",
            ]
            _matched = False
            for sep in _eq_seps:
                if sep in claim:
                    part = claim.split(sep)[0].strip()
                    if part and len(part) >= 10:
                        _add(clean_results(self.search_keyword(part, web_mode=True)))
                    _matched = True
                    break
            if not _matched:
                for sep in _event_seps:
                    if sep in claim:
                        part = claim.split(sep)[0].strip()
                        if part and len(part) >= 10:
                            _add(clean_results(self.search_keyword(part, web_mode=True)))
                        break

        # 4. Fallback: ключевые слова (только при недостатке результатов)
        # ВАЖНО: прогоняем через clean_results — иначе сюда попадает мусор (чешские/вьетнамские сайты)
        if len(all_results) < 3:
            print(f"  [Fallback] Мало результатов ({len(all_results)}) — запускаем keyword fallback (с фильтром)")
            combined = " ".join(kw.strip() for kw in keywords if kw.strip())
            if combined:
                _add(clean_results(self.search_keyword(combined)))
            if claim.strip():
                direct = claim.strip()[:120]
                if direct != combined:
                    _add(clean_results(self.search_keyword(direct)))
            for kw in keywords:
                if kw.strip():
                    _add(clean_results(self.search_keyword(kw)))

        return all_results

    def rank_by_relevance(
        self,
        claim: str,
        results: List[Dict[str, str]],
        min_semantic_score: float = 0.35,
    ) -> List[Dict[str, str]]:
        """Ранжирование: семантическое (если доступно) или bag-of-words fallback."""
        if not results:
            return []

        if self._semantic_ranker is not None:
            results = self._semantic_ranker.rank_results(claim, results, top_k=10)
            results = boost_by_credibility(results)

            before = len(results)
            filtered = [r for r in results if r.get("semantic_score", 0) >= min_semantic_score]
            if len(filtered) < 3 and before >= 3:
                filtered = results[:3]
            elif len(filtered) < before:
                dropped = before - len(filtered)
                print(f"  Отфильтровано нерелевантных: {dropped}/{before} "
                      f"(порог {min_semantic_score})")
            return filtered if filtered else results[:3]

        # Fallback: bag-of-words
        for art in results:
            text = f"{art.get('title', '')} {art.get('snippet', '')}"
            score = cosine_similarity(claim, text)
            art["relevance_score"] = round(score, 3)
            art["is_confirming"]   = score >= RELEVANCE_THRESHOLD
            art["is_related"]      = RELATED_THRESHOLD <= score < RELEVANCE_THRESHOLD

        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return results

    @staticmethod
    def format_results(results: List[Dict[str, str]]) -> str:
        """Форматирует результаты для LLM — нейтрально, без предрешения вердикта.

        При пустом списке возвращает ZERO_RESULTS_CONTEXT (сигнал потенциального фейка).
        Непроверенные источники помечаются предупреждением для LLM-судьи.
        """
        if not results:
            return ZERO_RESULTS_CONTEXT

        lines = [
            f"Найдено {len(results)} источников. Прочитай каждый ВНИМАТЕЛЬНО и "
            f"определи, подтверждает ли он утверждение или нет."
        ]
        lines.append("")

        for i, art in enumerate(results[:7], 1):
            parts = [f"{i}. {art.get('title', '')}"]
            if art.get("source"):
                parts.append(f"   Источник: {art['source']}")
            if art.get("date"):
                parts.append(f"   Дата: {art['date']}")
            if art.get("snippet"):
                parts.append(f"   {art['snippet'][:500]}")
            if art.get("_unverified"):
                parts.append(
                    "   ПРЕДУПРЕЖДЕНИЕ: [НЕПРОВЕРЕННЫЙ ИСТОЧНИК] — "
                    "неизвестный домен, низкое доверие. Не использовать как "
                    "основное доказательство."
                )
            lines.append("\n".join(parts))

        return "\n\n".join(lines)
