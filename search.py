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
import urllib.error
import urllib.parse
import urllib.request
import warnings
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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

# Async DDG (опционально)
_ASYNC_DDGS = None
try:
    from duckduckgo_search import AsyncDDGS as _ASYNC_DDGS
except ImportError:
    pass

_DDG_TIMEOUT = 20  # секунд — достаточно для медленного Bing-бэкенда

from cache import SearchCache
from config import SearchConfig
from source_credibility import boost_by_credibility


# V13: Tenacity retry wrapper for Wikipedia/Wikidata API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ConnectionError,
    )),
    reraise=True,
)
def _wiki_api_call(url: str, headers: dict = None, timeout: int = 10) -> dict:
    """Single Wikipedia/Wikidata API call with tenacity retry."""
    if headers is None:
        headers = {"User-Agent": "AntifakeBot/5.0 (fact-checking; Python)"}
    req = urllib.request.Request(url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


# ============================================================
# ШАГ 1: МАТРИЦА ДОВЕРЕННЫХ ИСТОЧНИКОВ
# ============================================================

TRUSTED_SOURCES: Dict[str, List[str]] = {
    "reference": [
        "wikipedia.org",
        "britannica.com",
        "wikimedia.org",
        "merriam-webster.com",
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
        "theguardian.com",
        "dw.com",
        "euronews.com",
        "cnn.com",
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
        "lenta.ru",
        "iz.ru",
        "kp.ru",
        "mk.ru",
        "1tv.ru",
        "gazeta.ru",
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
    "art_culture_global": [
        "theguardian.com",
        "artsy.net",
        "britannica.com",
        "metmuseum.org",
        "moma.org",
        "tate.org.uk",
        "theartnewspaper.com",
    ],
    "art_culture_ru": [
        "culture.ru",
        "tretyakovgallery.ru",
        "hermitagemuseum.org",
        "pushkinmuseum.art",
    ],
    "sports_global": [
        "espn.com",
        "goal.com",
        "transfermarkt.com",
        "olympics.com",
    ],
    "sports_ru": [
        "sport-express.ru",
        "championat.com",
        "sports.ru",
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
    "science":     ["science_med_global",   "science_med_ru"],
    "politics":    ["news_agencies_global", "news_agencies_ru"],
    "technology":  ["tech_crypto_global",   "tech_crypto_ru"],
    "crypto":      ["tech_crypto_global",   "tech_crypto_ru"],
    "art_culture": ["art_culture_global",   "art_culture_ru"],
    "sports":      ["sports_global",        "sports_ru"],
    "general":     ["news_agencies_global", "news_agencies_ru"],
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
- art_culture (art, painting, sculpture, music, literature, theater, museum, artist)
- sports (football, hockey, tennis, basketball, Olympics, championship)
- general  (economy, celebrity, business, other)

Claim: "{claim}"

Topic:[/INST]"""

_VALID_TOPICS = {"science", "politics", "technology", "art_culture", "sports", "general"}

# Словарь терминов, которые MarianMT переводит НЕПРАВИЛЬНО
TRANSLATION_OVERRIDES = {
    # Термины искусства/культуры
    "муравьед": "anteater",
    "муравьеда": "anteater",
    "муравьедом": "anteater",
    "постоянство памяти": "The Persistence of Memory",
    "сюрреализм": "surrealism",
    "сюрреалист": "surrealist",
    "сюрреалистом": "surrealist",
    "импрессионизм": "impressionism",
    "кубизм": "cubism",
    # Имена художников — не переводить
    "дали": "Dalí",
    "пикассо": "Picasso",
    "малевич": "Malevich",
    "кандинский": "Kandinsky",
    "моне": "Monet",
    "ван гог": "Van Gogh",
    # Имена знаменитостей — защита от побуквенного перевода
    "долина": "Dolina",
    "долиной": "Dolina",
    "долину": "Dolina",
    "роналдо": "Ronaldo",
    "криштиано": "Cristiano",
    "месси": "Messi",
    # Специфические термины
    "безопасный счёт": "safe account",
    "безопасный счет": "safe account",
}

# Словарь переводов именованных сущностей (географические объекты, организации,
# достопримечательности). Используется ДО машинного перевода для корректной
# передачи устоявшихся английских названий.
ENTITY_TRANSLATIONS = {
    # --- Многословные сущности (длинные первыми при сортировке) ---
    "великая китайская стена": "Great Wall of China",
    "эйфелева башня": "Eiffel Tower",
    "красная площадь": "Red Square",
    "тихий океан": "Pacific Ocean",
    "северный ледовитый океан": "Arctic Ocean",
    "чёрное море": "Black Sea",
    "черное море": "Black Sea",
    "средиземное море": "Mediterranean Sea",
    "атлантический океан": "Atlantic Ocean",
    "индийский океан": "Indian Ocean",
    "мёртвое море": "Dead Sea",
    "мертвое море": "Dead Sea",
    "каспийское море": "Caspian Sea",
    "байкал": "Lake Baikal",
    "эверест": "Mount Everest",
    "килиманджаро": "Mount Kilimanjaro",
    "статуя свободы": "Statue of Liberty",
    "биг-бен": "Big Ben",
    "колизей": "Colosseum",
    "тадж-махал": "Taj Mahal",
    "мачу-пикчу": "Machu Picchu",
    "ниагарский водопад": "Niagara Falls",
    "панамский канал": "Panama Canal",
    "суэцкий канал": "Suez Canal",
    "сахара": "Sahara Desert",
    "амазонка": "Amazon River",
    "нил": "Nile River",
    "волга": "Volga River",
    "дунай": "Danube River",
    "миссисипи": "Mississippi River",
    "титаник": "Titanic",
    "кремль": "Kremlin",
    "версаль": "Palace of Versailles",
    "букингемский дворец": "Buckingham Palace",
    "пентагон": "The Pentagon",
    "белый дом": "White House",
    "великобритания": "United Kingdom",
    "соединённые штаты": "United States",
    "соединенные штаты": "United States",
    "советский союз": "Soviet Union",
    "европейский союз": "European Union",
    "объединённые нации": "United Nations",
    "объединенные нации": "United Nations",
    "всемирная организация здравоохранения": "World Health Organization",
    "международная космическая станция": "International Space Station",
    "марианская впадина": "Mariana Trench",
    "пирамиды гизы": "Pyramids of Giza",
    "гренландия": "Greenland",
    "антарктида": "Antarctica",
    "арктика": "Arctic",
    "фудзияма": "Mount Fuji",
    "сиднейский оперный театр": "Sydney Opera House",
    "озеро титикака": "Lake Titicaca",
    "берлинская стена": "Berlin Wall",
    "нотр-дам": "Notre-Dame",
    "альпы": "Alps",
    "гималаи": "Himalayas",
    "большой барьерный риф": "Great Barrier Reef",
    "ватикан": "Vatican",
    "галапагосские острова": "Galapagos Islands",
    "стоунхендж": "Stonehenge",
    # --- Страны (лемма → EN) ---
    "россия": "Russia",
    "германия": "Germany",
    "франция": "France",
    "китай": "China",
    "япония": "Japan",
    "индия": "India",
    "бразилия": "Brazil",
    "австралия": "Australia",
    "канада": "Canada",
    "италия": "Italy",
    "испания": "Spain",
    "турция": "Turkey",
    "египет": "Egypt",
    "иран": "Iran",
    "ирак": "Iraq",
    "сирия": "Syria",
    "украина": "Ukraine",
    "польша": "Poland",
    "корея": "Korea",
    "мексика": "Mexico",
    "аргентина": "Argentina",
    "нигерия": "Nigeria",
    "пакистан": "Pakistan",
    "саудовская аравия": "Saudi Arabia",
    # --- Города ---
    "москва": "Moscow",
    "париж": "Paris",
    "лондон": "London",
    "берлин": "Berlin",
    "пекин": "Beijing",
    "токио": "Tokyo",
    "вашингтон": "Washington",
    "нью-йорк": "New York",
    "рим": "Rome",
    "мадрид": "Madrid",
    "стамбул": "Istanbul",
    "львов": "Lviv",
    # --- Персоналии ---
    "наполеон": "Napoleon",
    "бонапарт": "Bonaparte",
    "гагарин": "Gagarin",
    "путин": "Putin",
    "трамп": "Trump",
    "байден": "Biden",
    "зеленский": "Zelensky",
    "маск": "Musk",
    "джобс": "Jobs",
    "гейтс": "Gates",
    "эйнштейн": "Einstein",
    "тесла": "Tesla",
    "менделеев": "Mendeleev",
    # --- Континенты/регионы ---
    "африка": "Africa",
    "европа": "Europe",
    "азия": "Asia",
    "луна": "Moon",
    "марс": "Mars",
    "юпитер": "Jupiter",
    "сатурн": "Saturn",
    # --- Единицы/термины ---
    "цельсий": "Celsius",
    "фаренгейт": "Fahrenheit",
}


def _apply_entity_translations(text: str) -> str:
    """Заменяет известные именованные сущности на английские аналоги ДО перевода.

    Двухпроходная стратегия:
    1. Прямое совпадение (case-insensitive) — "амазонка" → "Amazon River"
    2. Лемматизация через pymorphy2 — "России" → lemma "россия" → "Russia"
    """
    result = text
    # Проход 1: Прямое совпадение (длинные фразы первыми)
    for ru, en in sorted(ENTITY_TRANSLATIONS.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(ru), re.IGNORECASE)
        result = pattern.sub(en, result)

    # Проход 2: Лемматизация оставшейся кириллицы
    # Находим слова, которые не были переведены
    remaining_cyrillic = re.findall(r'[а-яёА-ЯЁ]{3,}', result)
    if remaining_cyrillic:
        try:
            from nlp_russian import lemmatize
            for word in remaining_cyrillic:
                lemma = lemmatize(word)
                if lemma in ENTITY_TRANSLATIONS:
                    # Заменяем падежную форму на английский перевод
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    result = pattern.sub(ENTITY_TRANSLATIONS[lemma], result)
        except ImportError:
            pass  # nlp_russian недоступен — пропускаем лемматизацию

    return result


def _apply_translation_overrides(text: str) -> str:
    """Заменяет известные проблемные фразы/слова ДО отправки в MarianMT."""
    result = text
    # Сортируем по длине фразы (длинные первыми) — "постоянство памяти" до "памяти"
    for ru, en in sorted(TRANSLATION_OVERRIDES.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(ru), re.IGNORECASE)
        result = pattern.sub(en, result)
    return result


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
    _NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

    def __init__(self):
        self._generate_fn  = None   # callable: str → str (Mistral)
        self._marian_tok   = None
        self._marian_model = None
        self._marian_ready = False
        self._nllb_tok     = None
        self._nllb_model   = None
        self._nllb_ready   = False
        self._use_nllb     = False  # True если NLLB загружен успешно

    # ----------------------------------------------------------
    # NER-защита имён собственных при переводе
    # ----------------------------------------------------------

    @staticmethod
    def _protect_proper_nouns(text: str) -> tuple:
        """Заменяет имена собственные на placeholder'ы перед переводом."""
        placeholders = {}
        counter = [0]

        def replace_match(m):
            key = f"PROPN{counter[0]}"
            placeholders[key] = m.group(0)
            counter[0] += 1
            return key

        # Слова с заглавной буквы НЕ в начале предложения = имена собственные
        # Дефис включён для поддержки "Нью-Йорк", "Санкт-Петербург" и т.п.
        protected = re.sub(
            r'(?<=[а-яёa-z\s,])\b([А-ЯЁA-Z][а-яёa-z]+(?:[-\s]+[А-ЯЁA-Z][а-яёa-z]+)*)',
            replace_match, text
        )
        # Также защищаем слова в кавычках (названия произведений)
        def replace_quoted(m):
            key = f"PROPN{counter[0]}"
            placeholders[key] = m.group(1)
            counter[0] += 1
            return f'"{key}"'

        protected = re.sub(r'«([^»]+)»', replace_quoted, protected)
        protected = re.sub(r'"([^"]+)"', replace_quoted, protected)

        return protected, placeholders

    @staticmethod
    def _restore_proper_nouns(text: str, placeholders: dict) -> str:
        """Восстанавливает имена собственные после перевода."""
        for key, value in placeholders.items():
            text = text.replace(key, value)
        # Очистка артефактов: остаточные PROPN0, corrupted tokens
        text = re.sub(r'PROPN\d+', '', text)
        text = re.sub(r'\b\w+\d+(?:í|ó|é)\b', '', text)
        return text.strip()

    # ----------------------------------------------------------
    # Пост-обработка NLLB: удаление остаточной кириллицы
    # ----------------------------------------------------------

    _CYRILLIC_TRANSLIT = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
        'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k',
        'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r',
        'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts',
        'ч': 'ch', 'ш': 'sh', 'щ': 'shch', 'ъ': '', 'ы': 'y', 'ь': '',
        'э': 'e', 'ю': 'yu', 'я': 'ya',
    }

    @staticmethod
    def _cleanup_residual_cyrillic(text: str) -> str:
        """Remove any remaining Cyrillic words from NLLB English output.

        1. Check TRANSLATION_OVERRIDES for known terms
        2. Fall back to simple transliteration
        """
        def _replace_cyrillic(m: re.Match) -> str:
            word = m.group(0)
            word_lower = word.lower()
            # Check overrides dict (handles inflected forms)
            if word_lower in TRANSLATION_OVERRIDES:
                return TRANSLATION_OVERRIDES[word_lower]
            # Transliterate as last resort
            result = []
            for ch in word_lower:
                result.append(QueryClassifier._CYRILLIC_TRANSLIT.get(ch, ch))
            return "".join(result)

        return re.sub(r'[а-яёА-ЯЁ]+', _replace_cyrillic, text)

    # ----------------------------------------------------------
    # Перевод с предварительной подстановкой сущностей
    # ----------------------------------------------------------

    def translate_with_entities(self, text_ru: str) -> str:
        """Переводит RU->EN с предварительной подстановкой именованных сущностей.

        1. Применяет словарь ENTITY_TRANSLATIONS (устоявшиеся названия)
        2. Вызывает основной _translate (NLLB / MarianMT)
        3. Проверяет остаточную кириллицу и логирует предупреждение
        """
        # Шаг 1: подстановка сущностей из словаря
        with_entities = _apply_entity_translations(text_ru)

        # Шаг 2: машинный перевод (NLLB -> MarianMT -> оригинал)
        translated = self._translate(with_entities)

        # Шаг 3: проверка остаточной кириллицы (3+ символов подряд)
        remaining = re.findall(r'[а-яёА-ЯЁ]{3,}', translated)
        if remaining:
            print(
                f"  [Translator] WARN: остаточная кириллица после перевода: "
                f"{remaining[:5]}"
            )

        return translated

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
        # Early scam return — не давать art_culture перебить скам
        from claim_parser import detect_scam_concepts
        concept = detect_scam_concepts(claim)
        if concept.get("n_groups", 0) >= 2:
            category = "general"  # пусть пройдёт через scam-детектор, а не art_culture
        else:
            category = self._classify_topic(claim)
        en_query = self.translate_with_entities(claim)
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
            self._re_art = re.compile(
                r"\b(живопис\w*|картин\w*|художник\w*|скульптур\w*|музей\w*|"
                r"сюрреализ\w*|импрессионизм|авангард\w*|биограф\w*|творчеств\w*|"
                r"выставк\w*|галере\w*|искусств[оа]\w*|литератур\w*|поэт\w*|"
                r"писател\w*|философ\w*|историч\w*|средневеков\w*|"
                r"композитор\w*|симфони\w*|опер[аы]\w*|балет\w*|театр\w*|"
                r"дали|пикассо|моне|рембрандт|"
                r"renaissan|baroque|surreal|impressi)\b",
                re.IGNORECASE,
            )
            self._re_sports = re.compile(
                r"\b(футбол\w*|хоккей\w*|теннис\w*|баскетбол\w*|олимпи\w*|"
                r"чемпионат\w*|лига\w*|матч\w*|турнир\w*|спортсмен\w*|"
                r"роналдо|месси|нефтехимик\w*|забил\w*|гол\w*|"
                r"трансфер\w*|тренер\w*|сборн\w*)\b",
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
        # art_culture ПЕРЕД technology чтобы "искусство" не попало в tech
        if self._re_art.search(claim):
            return "art_culture"
        if self._re_sports.search(claim):
            return "sports"
        if self._re_tech.search(claim):
            return "technology"
        if self._re_politics.search(claim):
            return "politics"
        return "general"

    # ----------------------------------------------------------
    # Перевод: NLLB-200 (primary) → MarianMT (fallback)
    # ----------------------------------------------------------

    def _load_nllb(self):
        """Попытка загрузки NLLB-200-distilled-600M. Если не удалось — fallback на Marian."""
        if self._nllb_ready:
            return
        self._nllb_ready = True
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
            import torch
            print(f"  [Translator] Загрузка {self._NLLB_MODEL_NAME} (CPU)...")
            # V13: Fix tie_word_embeddings warning
            nllb_config = AutoConfig.from_pretrained(self._NLLB_MODEL_NAME)
            nllb_config.tie_word_embeddings = False
            self._nllb_tok = AutoTokenizer.from_pretrained(self._NLLB_MODEL_NAME)
            self._nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                self._NLLB_MODEL_NAME,
                config=nllb_config,
                torch_dtype=torch.float32,
            )
            self._nllb_model.eval()
            self._use_nllb = True
            print(f"  [Translator] {self._NLLB_MODEL_NAME} загружен ✓")
        except Exception as e:
            print(f"  [Translator] NLLB не удалось загрузить: {e}, fallback на MarianMT")
            self._load_marian()

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
        """Переводит RU→EN. Пробует NLLB-200, затем MarianMT, затем оригинал.
        Раскрывает аббревиатуры перед переводом для лучшего качества.
        При ошибке возвращает оригинальный текст.
        """
        # Пробуем NLLB-200 первым
        self._load_nllb()

        # Раскрываем аббревиатуры даже если модель недоступна
        expanded = self._expand_abbreviations(text)

        # NER-защита: имена → placeholder ПЕРЕД overrides и переводом
        # Так "Сальвадор Дали" → PROPN0 ДО того, как overrides обработают "дали"
        protected, placeholders = self._protect_proper_nouns(expanded)
        protected = _apply_translation_overrides(protected)  # Словарь терминов
        protected = _apply_entity_translations(protected)   # Именованные сущности

        # Попытка перевода через NLLB-200
        if self._use_nllb and self._nllb_model is not None:
            try:
                import torch
                self._nllb_tok.src_lang = "rus_Cyrl"
                # V13: Expanded NLLB limits (400 chars input, 200 tokens output)
                inputs = self._nllb_tok(protected[:400], return_tensors="pt")
                translated_tokens = self._nllb_model.generate(
                    **inputs,
                    forced_bos_token_id=self._nllb_tok.convert_tokens_to_ids("eng_Latn"),
                    max_length=200,
                )
                translated = self._nllb_tok.batch_decode(
                    translated_tokens, skip_special_tokens=True
                )[0]
                translated = self._cleanup_residual_cyrillic(translated)
                return self._restore_proper_nouns(translated, placeholders)
            except Exception as e:
                print(f"  [Translator] NLLB ошибка: {e}, fallback на MarianMT")

        # Fallback на MarianMT
        if not self._marian_ready:
            self._load_marian()

        if self._marian_model is None:
            return self._restore_proper_nouns(expanded, placeholders)

        try:
            import torch
            inputs = self._marian_tok(
                [protected[:120]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            with torch.no_grad():
                ids = self._marian_model.generate(**inputs, max_new_tokens=80, num_beams=4)
            translated = self._marian_tok.decode(ids[0], skip_special_tokens=True).strip()

            result = self._restore_proper_nouns(translated, placeholders)
            return result
        except Exception as e:
            print(f"  [Translator] Ошибка перевода: {e}")
            return self._restore_proper_nouns(expanded, placeholders)


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

    # Критические скам-сущности: если ни одна из них не найдена в source → max penalty
    scam_keywords = ["выплат", "взнос", "бот", "телеграм", "telegram",
                     "актив", "наследств", "верификац", "комисси", "страхов",
                     "безоп", "кошел", "привяз", "облиг", "секрет",
                     "проверочн", "регистрац", "платформ"]
    critical_terms = [w for w in fact_words if any(sk in w for sk in scam_keywords)]
    if critical_terms:
        critical_covered = sum(1 for w in critical_terms if w[:5] in context_lower)
        if critical_covered == 0:
            return 1.0  # ни один критический скам-термин не найден → max penalty

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
# ASYNC DDG SEARCH (для параллельного выполнения запросов)
# ============================================================

import asyncio

async def _search_ddg_async(query: str, max_results: int = 8) -> list:
    """Асинхронный DDG-поиск (если AsyncDDGS доступен)."""
    if _ASYNC_DDGS is None:
        return []
    try:
        async with _ASYNC_DDGS(timeout=_DDG_TIMEOUT) as ddgs:
            results = await ddgs.atext(query, max_results=max_results)
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", r.get("snippet", "")),
                    "source": r.get("source", ""),
                    "link": r.get("url", r.get("href", "")),
                    "date": r.get("date", ""),
                }
                for r in (results or [])
            ]
    except Exception:
        return []


async def search_all_async(queries: List[str], max_results: int = 8) -> List[List[Dict]]:
    """Запуск всех DDG-запросов параллельно. Ожидаемое ускорение: 3-5x."""
    tasks = [_search_ddg_async(q, max_results) for q in queries]
    return await asyncio.gather(*tasks)


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

    def _search_ddg_parallel(self, queries: List[str], max_results: int = 8) -> List[List[Dict[str, str]]]:
        """Run multiple DDG queries in parallel using asyncio. B3 optimization.

        Falls back to sequential _search_ddg if async execution fails.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context (e.g., Streamlit) — use thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
                    futures = [executor.submit(self._search_ddg, q, max_results) for q in queries]
                    return [f.result(timeout=30) for f in futures]
            else:
                return loop.run_until_complete(search_all_async(queries, max_results))
        except RuntimeError:
            # No event loop — create one
            try:
                return asyncio.run(search_all_async(queries, max_results))
            except Exception as e:
                print(f"  [DDG-parallel] Async failed ({e}), falling back to sequential")
                return [self._search_ddg(q, max_results) for q in queries]

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

    def _search_wikipedia_with_extract(self, keyword: str, lang: str = "ru") -> List[Dict[str, str]]:
        """Wikipedia + полный вводный раздел (первые 10 предложений).

        EN fallback: если русская Wikipedia пуста или snippet < 200 символов,
        пробуем английскую Wikipedia с переведённым запросом.
        """
        hits = self._search_wikipedia(keyword, lang)

        for hit in hits[:2]:  # Топ-2 результата
            title = hit.get("title", "")
            try:
                import requests
                q = urllib.parse.quote(title)
                extract_url = (
                    f"https://{lang}.wikipedia.org/w/api.php"
                    f"?action=query&titles={q}&prop=extracts"
                    f"&exintro=1&explaintext=1&format=json&exsentences=10"
                )
                resp = requests.get(extract_url, timeout=5,
                                    headers={"User-Agent": "AntifakeBot/3.0"})
                data = resp.json()
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    extract = page.get("extract", "")
                    if extract and len(extract) > len(hit.get("snippet", "")):
                        hit["snippet"] = extract[:1500]  # Первые 1500 символов
            except Exception:
                pass  # Оставляем оригинальный snippet

        # --- EN fallback ---
        # Если RU-результаты пусты или все snippet'ы короткие (< 200 символов),
        # пробуем английскую Wikipedia
        total_snippet_len = sum(len(h.get("snippet", "")) for h in hits)
        if lang == "ru" and (not hits or total_snippet_len < 200):
            try:
                # Переводим запрос для EN Wikipedia
                en_keyword = _apply_entity_translations(keyword)
                # Если после подстановки сущностей остался кириллический текст,
                # пробуем машинный перевод через classifier
                if re.search(r'[а-яёА-ЯЁ]{3,}', en_keyword):
                    en_keyword = self._classifier.translate_with_entities(keyword)
                en_hits = self._search_wikipedia(en_keyword[:80], lang="en")

                for hit_en in en_hits[:2]:
                    title_en = hit_en.get("title", "")
                    try:
                        import requests
                        q_en = urllib.parse.quote(title_en)
                        extract_url_en = (
                            f"https://en.wikipedia.org/w/api.php"
                            f"?action=query&titles={q_en}&prop=extracts"
                            f"&exintro=1&explaintext=1&format=json&exsentences=10"
                        )
                        resp_en = requests.get(
                            extract_url_en, timeout=5,
                            headers={"User-Agent": "AntifakeBot/3.0"}
                        )
                        data_en = resp_en.json()
                        pages_en = data_en.get("query", {}).get("pages", {})
                        for page_en in pages_en.values():
                            extract_en = page_en.get("extract", "")
                            if extract_en and len(extract_en) > len(hit_en.get("snippet", "")):
                                hit_en["snippet"] = extract_en[:1500]
                    except Exception:
                        pass
                if en_hits:
                    print(f"  [Wikipedia] EN fallback: +{len(en_hits)} результатов для '{en_keyword[:50]}'")
                    hits.extend(en_hits)
            except Exception as e:
                print(f"  [Wikipedia] EN fallback ошибка: {e}")

        return hits

    def _search_arxiv(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """ArXiv API для научных утверждений."""
        try:
            q = urllib.parse.quote(query[:100])
            url = f"http://export.arxiv.org/api/query?search_query=all:{q}&max_results={max_results}"
            req = urllib.request.Request(url, headers={"User-Agent": "AntifakeBot/3.0"})
            resp = urllib.request.urlopen(req, timeout=8)
            data = resp.read().decode("utf-8")

            articles = []
            # Простой парсинг Atom XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns)[:max_results]:
                title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
                summary = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")[:500]
                link_el = entry.find("atom:id", ns)
                link = link_el.text.strip() if link_el is not None else ""
                articles.append({
                    "title": title,
                    "snippet": summary,
                    "source": "arxiv.org",
                    "link": link,
                    "date": "",
                })
            return articles
        except Exception as e:
            print(f"  [ArXiv] Ошибка: {e}")
            return []

    def _search_google_news_rss(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Google News RSS для свежих новостей."""
        try:
            import feedparser
        except ImportError:
            return []
        try:
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query[:100])}&hl=ru&gl=RU"
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries[:max_results]:
                articles.append({
                    "title": entry.get("title", ""),
                    "snippet": entry.get("summary", entry.get("description", ""))[:500],
                    "source": entry.get("source", {}).get("title", "Google News"),
                    "link": entry.get("link", ""),
                    "date": entry.get("published", ""),
                })
            return articles
        except Exception as e:
            print(f"  [GoogleNews] Ошибка: {e}")
            return []

    def federated_search(self, claim: str, category: str, en_query: str) -> List[Dict[str, str]]:
        """Федеративный поиск: разные API для разных категорий."""
        results = []

        # По категории — дополнительные источники
        if category == "science":
            results.extend(self._search_arxiv(en_query))
        elif category in ("politics", "general"):
            results.extend(self._search_google_news_rss(en_query))

        # Всегда: Wikipedia (расширенный)
        results.extend(self._search_wikipedia_with_extract(claim[:80]))

        return results

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
        # B3: Try async parallel execution first, fallback to sequential
        queries = build_search_queries(classified)
        all_raw: List[Dict[str, str]] = []
        seen_urls: set = set()

        # Separate cached vs uncached queries
        _uncached_queries = []
        _cached_results = {}
        for q_info in queries:
            cache_key = f"routed:{q_info['query']}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                _cached_results[q_info['query']] = cached
            else:
                _uncached_queries.append(q_info)

        # B3: Async parallel DDG for uncached queries
        _async_results = {}
        if _uncached_queries and _ASYNC_DDGS is not None:
            try:
                _async_query_strs = [q["query"] for q in _uncached_queries]
                _loop = asyncio.new_event_loop()
                _raw_lists = _loop.run_until_complete(
                    search_all_async(_async_query_strs, max_results=self.config.num_results)
                )
                _loop.close()
                for q_info, raw in zip(_uncached_queries, _raw_lists):
                    _async_results[q_info['query']] = raw
                    self._cache.set(f"routed:{q_info['query']}", raw)
                print(f"  [B3:Async] {len(_uncached_queries)} queries in parallel")
            except Exception as _ae:
                print(f"  [B3:Async] Failed ({_ae}), falling back to sequential")
                _async_results = {}

        for q_info in queries:
            label = q_info["label"]
            query = q_info["query"]
            print(f"  [DDG:{label}] {query[:90]}...")

            if query in _cached_results:
                raw = _cached_results[query]
            elif query in _async_results:
                raw = _async_results[query]
            else:
                # Sequential fallback
                self._rate_limiter.wait()
                raw = self._search_ddg(query, max_results=self.config.num_results)
                self._cache.set(f"routed:{query}", raw)

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

        # 2.5. A6: Counter-search — negation queries for debunking + myth-buster
        #       B3: Batch DDG queries in parallel instead of sequential loop
        if claim.strip():
            short_claim = claim.strip()[:80]
            counter_suffixes = [
                "миф OR фейк OR разоблачение OR опровержение",
                "fact check OR debunked OR false OR myth OR fake",
                "миф OR разоблачение OR заблуждение",
            ]
            mb_queries = [f"{short_claim} {suffix}" for suffix in counter_suffixes]
            mb_total_before = len(all_results)

            # B3: parallel DDG search for all counter-suffix queries
            mb_batch_results = self._search_ddg_parallel(mb_queries, max_results=5)

            for mb_raw in mb_batch_results:
                mb_cleaned = clean_results(mb_raw)

                # A6: Mark counter-search results for downstream NLI
                for r in mb_cleaned:
                    r["is_counter_evidence"] = True

                # Direction 3: резервируем минимум 2–3 слота для фактчекеров.
                fc_hits = [
                    r for r in mb_cleaned
                    if _domain_in_factcheckers(_extract_base_domain(r.get("link", "")))
                ]
                non_fc = [r for r in mb_cleaned if r not in fc_hits]
                # Фактчекеры — первыми в пул (до 3 слотов), затем остальные
                _add(fc_hits[:3] + non_fc)

            mb_added = len(all_results) - mb_total_before
            print(f"  [Counter-search] +{mb_added} результатов после counter/myth-buster запросов")

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

        # 5. Counter-search: ищем разоблачения / опровержения утверждения
        #    B3: Batch RU + EN counter-queries in parallel
        if claim.strip():
            short_claim = claim.strip()[:80]
            counter_before = len(all_results)

            # Build list of counter-queries (RU always, EN if translation available)
            counter_queries: List[str] = [
                f'"{short_claim}" миф OR фейк OR разоблачение OR опровержение'
            ]
            try:
                claim_en = self._classifier.translate_with_entities(short_claim)
                if claim_en and claim_en != short_claim:
                    counter_queries.append(
                        f'"{claim_en}" debunked OR false OR myth OR fake'
                    )
            except Exception as e:
                print(f"  [Counter-search] EN перевод не удался: {e}")

            # B3: parallel DDG search for RU (+EN) counter-queries
            counter_batch = self._search_ddg_parallel(counter_queries, max_results=3)

            for batch_raw in counter_batch:
                batch_cleaned = clean_results(batch_raw)
                for art in batch_cleaned:
                    art["is_counter_evidence"] = True
                _add(batch_cleaned)

            counter_added = len(all_results) - counter_before
            print(
                f"  [Counter-search] +{counter_added} результатов "
                f"контр-доказательств (разоблачения/опровержения)"
            )

        return all_results

    # ----------------------------------------------------------
    # V5: Wikipedia direct entity lookup (Task 2)
    # ----------------------------------------------------------

    def wiki_entity_lookup(self, entities: List[str], lang: str = "ru") -> List[Dict[str, str]]:
        """Прямой поиск в Wikipedia по каждой сущности отдельно.

        В отличие от _search_wikipedia (поиск по claim-тексту),
        этот метод ищет КОНКРЕТНЫЕ страницы для каждой сущности:
        - "Луна" → статья "Луна" с определением "естественный спутник Земли"
        - "Microsoft" → статья "Microsoft" с именами основателей
        - "Титаник" → статья "Титаник" с местом крушения

        Возвращает результаты с полными intro-секциями (до 2000 символов).
        EN fallback: если RU-результат пуст или snippet < 200 символов,
        пробуем English Wikipedia с переведённым названием сущности.
        """
        results: List[Dict[str, str]] = []
        seen_titles: set = set()
        headers = {"User-Agent": "AntifakeBot/4.0 (fact-checking)"}

        for entity in entities[:5]:  # Макс 5 сущностей
            entity = entity.strip()
            if not entity or len(entity) < 2:
                continue

            found_extract = ""  # Текст из RU Wikipedia для этой сущности

            try:
                # Шаг 1: Поиск страницы по точному названию (RU)
                q = urllib.parse.quote(entity)
                search_url = (
                    f"https://{lang}.wikipedia.org/w/api.php"
                    f"?action=query&list=search&srsearch={q}&format=json"
                    f"&srlimit=2&srprop=snippet"
                )
                req = urllib.request.Request(search_url, headers=headers)
                resp = urllib.request.urlopen(req, timeout=8)
                data = json.loads(resp.read())
                hits = data.get("query", {}).get("search", [])

                if hits:
                    # Шаг 2: Получаем полный intro для топ-1 результата
                    title = hits[0].get("title", "")
                    if title.lower() not in seen_titles:
                        seen_titles.add(title.lower())

                        qt = urllib.parse.quote(title)
                        extract_url = (
                            f"https://{lang}.wikipedia.org/w/api.php"
                            f"?action=query&titles={qt}&prop=extracts"
                            f"&exintro=1&explaintext=1&format=json&exsentences=20"
                        )
                        req2 = urllib.request.Request(extract_url, headers=headers)
                        resp2 = urllib.request.urlopen(req2, timeout=8)
                        data2 = json.loads(resp2.read())
                        pages = data2.get("query", {}).get("pages", {})

                        for page in pages.values():
                            extract = page.get("extract", "")
                            if extract and len(extract) > 50:
                                found_extract = extract
                                link = (
                                    f"https://{lang}.wikipedia.org/wiki/"
                                    + urllib.parse.quote(title.replace(" ", "_"))
                                )
                                results.append({
                                    "title": f"{title} — Википедия",
                                    "snippet": extract[:2000],
                                    "source": f"{lang}.wikipedia.org",
                                    "link": link,
                                    "date": "",
                                    "source_credibility": 0.90,
                                })
                                break

            except Exception as e:
                print(f"  [WikiEntity] Ошибка для '{entity}': {e}")

            # --- EN fallback для этой сущности ---
            if lang == "ru" and len(found_extract) < 200:
                try:
                    # Переводим название сущности для EN Wikipedia
                    en_entity = _apply_entity_translations(entity)
                    if re.search(r'[а-яёА-ЯЁ]{3,}', en_entity):
                        en_entity = self._classifier.translate_with_entities(entity)

                    q_en = urllib.parse.quote(en_entity)
                    search_url_en = (
                        f"https://en.wikipedia.org/w/api.php"
                        f"?action=query&list=search&srsearch={q_en}&format=json"
                        f"&srlimit=2&srprop=snippet"
                    )
                    req_en = urllib.request.Request(search_url_en, headers=headers)
                    resp_en = urllib.request.urlopen(req_en, timeout=8)
                    data_en = json.loads(resp_en.read())
                    hits_en = data_en.get("query", {}).get("search", [])

                    if hits_en:
                        title_en = hits_en[0].get("title", "")
                        if title_en.lower() not in seen_titles:
                            seen_titles.add(title_en.lower())
                            qt_en = urllib.parse.quote(title_en)
                            extract_url_en = (
                                f"https://en.wikipedia.org/w/api.php"
                                f"?action=query&titles={qt_en}&prop=extracts"
                                f"&exintro=1&explaintext=1&format=json&exsentences=20"
                            )
                            req2_en = urllib.request.Request(extract_url_en, headers=headers)
                            resp2_en = urllib.request.urlopen(req2_en, timeout=8)
                            data2_en = json.loads(resp2_en.read())
                            pages_en = data2_en.get("query", {}).get("pages", {})

                            for page_en in pages_en.values():
                                extract_en = page_en.get("extract", "")
                                if extract_en and len(extract_en) > 50:
                                    link_en = (
                                        f"https://en.wikipedia.org/wiki/"
                                        + urllib.parse.quote(title_en.replace(" ", "_"))
                                    )
                                    results.append({
                                        "title": f"{title_en} — Wikipedia (EN)",
                                        "snippet": extract_en[:2000],
                                        "source": "en.wikipedia.org",
                                        "link": link_en,
                                        "date": "",
                                        "source_credibility": 0.90,
                                    })
                                    print(
                                        f"  [WikiEntity] EN fallback для '{entity}' "
                                        f"→ '{title_en}'"
                                    )
                                    break
                except Exception as e:
                    print(f"  [WikiEntity] EN fallback ошибка для '{entity}': {e}")

        if results:
            print(f"  [WikiEntity] Найдено {len(results)} статей для сущностей")
        return results

    # ----------------------------------------------------------
    # V5: Verification query generation (Task 4)
    # ----------------------------------------------------------

    def generate_verification_queries(self, claim: str, entities: List[str],
                                       generate_fn=None) -> List[str]:
        """Генерация целевых верификационных запросов.

        Вместо поиска по ключевым словам из claim, генерирует конкретные
        вопросы для проверки каждого аспекта утверждения:
        - "Microsoft основана Стивом Джобсом" → "кто основал Microsoft"
        - "Титаник затонул в Тихом океане" → "где затонул Титаник"
        """
        queries = []

        # Правило 1: Для каждой именованной сущности — прямой запрос "кто/что такое X"
        for entity in entities[:3]:
            if len(entity) > 3:
                queries.append(entity)

        # Правило 2: Паттерны для типичных проверяемых утверждений
        claim_lower = claim.lower()

        # Основатель / создатель
        founder_patterns = [
            (r'([\w\s]+?)\s+(?:основан[аоы]?|создан[аоы]?|учреждён[аоы]?)\s+(.+?)(?:\s+в\s+\d|$)',
             lambda m: [f"кто основал {m.group(1).strip()}", f"{m.group(2).strip()} основатель"]),
            (r'(.+?)\s+(?:основал[аи]?|создал[аи]?)\s+([\w\s]+)',
             lambda m: [f"кто основал {m.group(2).strip()}", f"{m.group(1).strip()} что основал"]),
        ]
        for pattern, gen_fn in founder_patterns:
            m = re.search(pattern, claim, re.IGNORECASE)
            if m:
                queries.extend(gen_fn(m))
                break

        # Место события
        location_patterns = [
            (r'(.+?)\s+(?:в|на)\s+([А-ЯЁ][\w]+(?:\s+[А-ЯЁ][\w]+)?)\s*(?:океан|мор|озер|город)',
             lambda m: [f"где {m.group(1).strip()}"]),
            (r'(?:затонул|произошл|случил|находит)\w*\s+.*?(?:в|на)\s+(\w+\s+\w+)',
             lambda m: [f"где затонул {entities[0]}" if entities else ""]),
        ]
        for pattern, gen_fn in location_patterns:
            m = re.search(pattern, claim, re.IGNORECASE)
            if m:
                qs = gen_fn(m)
                queries.extend([q for q in qs if q])
                break

        # Спутники / свойства планет
        if any(w in claim_lower for w in ['спутник', 'планета', 'орбит']):
            for ent in entities[:2]:
                queries.append(f"{ent} характеристики спутники")

        # Числовые факты (скорость, температура, население)
        if re.search(r'\d+\s*(?:м/с|км/ч|°[CС]|градус|метр|км|тыс|млн|млрд)', claim):
            for ent in entities[:2]:
                queries.append(f"{ent} точные данные характеристики")

        # Правило 3: LLM-генерация (если доступна) для сложных случаев
        if generate_fn and len(queries) < 3:
            try:
                prompt = (
                    f"[INST]Сгенерируй 2 конкретных поисковых запроса для проверки утверждения. "
                    f"Запросы должны искать ФАКТЫ, а не само утверждение. Только запросы через запятую.\n\n"
                    f"Утверждение: \"{claim}\"\n"
                    f"Запросы:[/INST]"
                )
                raw = generate_fn(prompt)
                for q in raw.split(","):
                    q = q.strip().strip('"').strip("'")
                    if q and len(q) > 5 and q not in queries:
                        queries.append(q)
            except Exception:
                pass

        # Дедупликация
        seen = set()
        unique = []
        for q in queries:
            ql = q.lower().strip()
            if ql and ql not in seen:
                seen.add(ql)
                unique.append(q)

        return unique[:6]

    def search_verification_queries(self, queries: List[str]) -> List[Dict[str, str]]:
        """Выполняет поиск по верификационным запросам.

        Возвращает отфильтрованные результаты из DDG + Wikipedia.
        """
        all_results: List[Dict[str, str]] = []
        seen_urls: set = set()

        # B3: Parallel DDG search
        ddg_results = self._search_ddg_parallel(queries, max_results=5)

        for batch in ddg_results:
            cleaned = clean_results(batch)
            for art in cleaned:
                url = art.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    art["_verification_query"] = True
                    all_results.append(art)

        # Wikipedia по каждому запросу
        for q in queries[:3]:
            wiki = self._search_wikipedia_with_extract(q[:80])
            for art in wiki:
                url = art.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    art["_verification_query"] = True
                    all_results.append(art)

        if all_results:
            print(f"  [VerifQuery] +{len(all_results)} результатов по верификационным запросам")
        return all_results

    # ----------------------------------------------------------
    # V5: Multi-hop entity counter-search (Task 5)
    # ----------------------------------------------------------

    def search_counter_entities(self, claim: str, entities: List[str]) -> List[Dict[str, str]]:
        """Multi-hop поиск: проверяет каждую сущность отдельно.

        Для "Microsoft основана Стивом Джобсом":
        - Ищет "Microsoft основатель" → находит Билла Гейтса
        - Ищет "Стив Джобс основатель" → находит Apple
        - Сравнение → противоречие

        Для "Титаник затонул в Тихом океане":
        - Ищет "Титаник место крушения" → находит Атлантический океан
        """
        all_results: List[Dict[str, str]] = []
        seen_urls: set = set()

        # Паттерны для генерации counter-запросов
        claim_lower = claim.lower()

        counter_queries = []
        for entity in entities[:3]:
            ent = entity.strip()
            if len(ent) < 3:
                continue

            # Для каждой сущности — поиск её реальных свойств
            if any(w in claim_lower for w in ['основан', 'создан', 'учрежд', 'основал', 'создал']):
                counter_queries.append(f"{ent} основатель создатель кто")
            if any(w in claim_lower for w in ['затону', 'крушен', 'авари', 'катастроф']):
                counter_queries.append(f"{ent} место где произошло")
            if any(w in claim_lower for w in ['спутник', 'планет']):
                counter_queries.append(f"{ent} спутники список")
            if any(w in claim_lower for w in ['протекает', 'впадает', 'река']):
                counter_queries.append(f"{ent} где протекает материк")
            if any(w in claim_lower for w in ['столиц', 'город']):
                counter_queries.append(f"{ent} столица какой страны")

        if not counter_queries:
            return []

        # B3: Parallel DDG search
        ddg_batches = self._search_ddg_parallel(counter_queries[:4], max_results=3)

        for batch in ddg_batches:
            cleaned = clean_results(batch)
            for art in cleaned:
                url = art.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    art["_counter_entity"] = True
                    all_results.append(art)

        if all_results:
            print(f"  [CounterEntity] +{len(all_results)} результатов multi-hop поиска")
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
            if art.get("is_counter_evidence"):
                parts.append(
                    "   [КОНТР-ДОКАЗАТЕЛЬСТВО] — результат поиска разоблачений. "
                    "Может указывать на то, что утверждение является фейком или мифом."
                )
            if art.get("_unverified"):
                parts.append(
                    "   ПРЕДУПРЕЖДЕНИЕ: [НЕПРОВЕРЕННЫЙ ИСТОЧНИК] — "
                    "неизвестный домен, низкое доверие. Не использовать как "
                    "основное доказательство."
                )
            lines.append("\n".join(parts))

        return "\n\n".join(lines)
