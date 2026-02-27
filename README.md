# Antifake — Система определения достоверности новостей

Система автоматической проверки утверждений на русском языке. Принимает текстовое утверждение, выполняет многоэтапный анализ через поиск, NLI и LLM, возвращает вердикт с источниками.

---

## Архитектура

```
Утверждение
    │
    ▼
[Декомпозиция]  ←  Mistral 7B разбивает составное на атомарные факты
    │
    ▼
[QueryClassifier]  ←  Helsinki-NLP MarianMT (RU→EN) + rule-based роутер
    │                    категории: science / politics / technology / general
    ▼
[DDG Search × 3]  ←  factcheckers | ru_sources | global_sources
    │                 site: операторы по матрице доверенных источников
    │   + Myth-buster:  "<claim> миф OR разоблачение OR debunk"
    │   + Wikipedia API (direct entity lookup)
    │   + Verification queries (целевые проверочные вопросы)
    │   + Multi-hop counter-entity search (ловит подмену персон)
    ▼
[clean_results]  ←  URL-фильтр: blacklist → trusted TLD → TRUSTED_SOURCES → DROP
    │
    ▼
[SemanticRanker]  ←  intfloat/multilingual-e5-base (bi-encoder, CPU)
    │
    ▼
[CrossEncoder]  ←  cross-encoder/ms-marco-MiniLM-L-6-v2 (top-7)
    │
    ├── [boost_factcheck_scores]  ←  +2.0 к скору для snopes/provereno/debunk-текстов
    │
    └── [validate_context_entities]  ←  штраф за отсутствие сущностей факта в контексте
            │                            фильтр при penalty > 0.35 (мин. 3 документа)
            ▼
[NLI Analysis]  ←  mDeBERTa-v3-base-xnli (entailment / contradiction, CPU)
    │                 + NLI per sub-claim (отдельный анализ каждого подпункта)
    │
    ▼
[Wikidata SPARQL]  ←  Структурированные факты из Knowledge Graph
    │                   основатель, столица, спутники, материк, дата основания
    │
    ▼
[Числа / Даты / Локации]  ←  детерминистическая проверка claim vs sources
    │
    ▼
[Mistral 7B Verdict]  ←  GRPO/SFT адаптер, Chain-of-Thought <reasoning>
    │
    ▼
[Ensemble Vote]  ←  числа(±3) + NLI(±2) + Wikidata(±2) + детали(±2) + LLM(±2) + скам(−4)
    │
    ▼
Вердикт + Скор + Источники
```

---

## Вердикты

| Вердикт | Описание |
|---------|----------|
| **ПРАВДА** | Утверждение подтверждено источниками |
| **ЛОЖЬ / ФЕЙК** | Утверждение опровергнуто фактчекерами или противоречит источникам |
| **МАНИПУЛЯЦИЯ / ПОЛУПРАВДА** | Часть фактов верна, часть — нет (составные утверждения) |
| **ЧАСТИЧНО ПОДТВЕРЖДЕНО** | Некоторые пункты подтверждены, по остальным данных нет |
| **НЕ ПОДТВЕРЖДЕНО** | Данных недостаточно для вердикта |

---

## Структура проекта

```
antifake/
├── app.py                  # Streamlit веб-интерфейс
├── pipeline.py             # LangChain LCEL цепочка (основной пайплайн)
├── search.py               # DDG-поиск, роутер запросов, фильтрация источников
├── nli_checker.py          # NLI анализ (mDeBERTa)
├── embeddings.py           # SemanticRanker (bi-encoder) + ReRanker (CrossEncoder)
├── claim_parser.py         # Извлечение чисел, дат, локаций, скам-паттернов
├── source_credibility.py   # Бустинг по кредитности домена
├── model.py                # Загрузка Mistral, LoRA адаптеров, LangChain wrapper
├── prompts.py              # Шаблоны промптов (SFT / GRPO / аудиторский)
├── wikidata.py             # Wikidata SPARQL — структурированная верификация фактов
├── adversarial.py          # Adversarial debate (защитник vs обвинитель)
├── satire_detector.py      # Детектор сатиры и юмора
├── fact_cache.py           # Кэш проверенных фактов
├── utils.py                # Стоп-слова, стемминг, транслитерация
├── config.py               # Конфигурация (ModelConfig, PipelineConfig, ...)
├── train_grpo.py           # Обучение с GRPO (reward-based)
├── data/                   # Датасеты для обучения и оценки
└── adapters/               # LoRA/GRPO адаптеры (локально)
```

---

## Требования

- Python 3.10+
- CUDA GPU рекомендован для LLM (оптимизировано под RTX 5070 12GB, Blackwell sm_120)
- CPU достаточен для NLI, CrossEncoder, SemanticRanker

```
torch>=2.1.0
transformers>=4.36.0
unsloth[colab-new]>=2024.1
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.25.0
trl>=0.15.0
langchain-core>=0.1.0
langchain-huggingface>=0.0.1
sentence-transformers>=2.3.0
ddgs>=9.0
sacremoses>=0.0.53
streamlit>=1.30.0
datasets>=2.16.0
```

> **Важно:** `xformers` намеренно исключён — не поддерживает Blackwell sm_120 (вызывает segfault). Если установлен, выполните `pip uninstall xformers`.

---

## Установка

```bash
git clone <repo>
cd antifake
pip install -r requirements.txt
```

---

## Запуск

### Веб-интерфейс (Streamlit)

```bash
streamlit run app.py
```

Опционально — Google Search через SerpAPI (без ключа работает через DuckDuckGo):

```bash
export SERPAPI_API_KEY="ваш_ключ"
streamlit run app.py
```

### Python API

```python
from pipeline import FactCheckPipeline

pipeline = FactCheckPipeline()
result = pipeline.check("Эйнштейн провалил экзамен по математике в школе")

print(result["verdict"])           # ЛОЖЬ / ФЕЙК
print(result["credibility_score"]) # 15
print(result["reasoning"])         # обоснование
for src in result["sources"]:
    print(src["link"])             # источники
```

### Структура ответа

```python
{
    "claim": str,                  # исходное утверждение
    "sub_claims": list[str],       # атомарные факты (если декомпозиция)
    "verdict": str,                # ПРАВДА / ЛОЖЬ / МАНИПУЛЯЦИЯ / ...
    "credibility_score": int,      # 0–100
    "confidence": int,             # уверенность модели, 0–100
    "reasoning": str,              # обоснование вердикта
    "chain_of_thought": str,       # <reasoning> блок (GRPO модель)
    "sub_verdicts": list[dict],    # per-fact результаты (составные утверждения)
    "sources": list[dict],         # список источников с URL
    "keywords": list[str],         # извлечённые ключевые слова
    "total_time": float,           # время выполнения, сек
}
```

---

## Матрица доверенных источников

Поиск работает только по верифицированным доменам. Все остальные домены отбрасываются на этапе `clean_results`.

| Категория | Примеры |
|-----------|---------|
| Фактчекеры (global) | snopes.com, reuters.com, politifact.com, factcheck.org, fullfact.org |
| Фактчекеры (RU) | provereno.media, stopfake.org, lapsha.media |
| Новостные агентства (global) | reuters.com, apnews.com, bbc.com, bloomberg.com, ft.com |
| Новостные агентства (RU) | tass.ru, interfax.ru, rbc.ru, kommersant.ru, vedomosti.ru |
| Наука и медицина | nature.com, who.int, nih.gov, nplus1.ru, cyberleninka.ru |
| Технологии / Крипто | techcrunch.com, wired.com, habr.com, coindesk.com |
| Официальные источники | europa.eu, un.org, imf.org, nato.int |
| Справочники | wikipedia.org, britannica.com |

Дополнительно: домены с TLD `.gov`, `.edu`, `.mil` всегда принимаются.

---

## Ensemble-вердикт: система голосования

Итоговый вердикт определяется суммой взвешенных сигналов:

| Сигнал | Вес | Условие |
|--------|-----|---------|
| Числа совпадают | +3.0 | claim-число = source-число (±10%) |
| Числа расходятся | −3.0 | несовпадение при отсутствии match |
| NLI entailment | до +2.0 | mDeBERTa entailment × count |
| NLI contradiction | до −2.0 | mDeBERTa contradiction × count |
| KEY_DETAIL_MISSING | −2.0 | уникальные слова claim отсутствуют в sources (stem matching) |
| ENTITY_SATURATION | +1.0 | ≥3 из 7 sources содержат сущности claim |
| WIKIDATA_CONFIRMED | до +2.0 | Wikidata подтверждает факты из claim |
| WIKIDATA_CONTRADICTED | до −3.0 | Wikidata опровергает факты из claim |
| LLM ПРАВДА | до +2.0 | пропорционально credibility_score |
| LLM ЛОЖЬ | до −2.0 | пропорционально credibility_score |
| SCAM_PATTERN | −4.0 | обнаружены мошеннические паттерны |
| DATE_YEAR_MISMATCH | −2.0 | год в claim ≠ годам в источниках |
| LOCATION_MISMATCH | −2.5 | все локации из claim отсутствуют в источниках |
| LOCATION_CONFIRMED | +0.5 | все локации найдены в источниках |
| HALF_TRUTH_ROLE | −1.0 | роль персоны искажена (со-основатель → основатель) |
| PERSON_ENTITY_MISMATCH | −3.0 | персона подменена (Джобс вместо Гейтса) |

`vote ≥ +1.0` → ПРАВДА · `vote ≤ −1.0` → ЛОЖЬ · иначе → adversarial debate → fallback на LLM

---

## Обнаружение манипуляций в составных утверждениях

Для составных утверждений (через «и», «а также», «однако», «кроме того») система:

1. Декомпозирует на атомарные факты через LLM
2. Проверяет каждый факт независимо
3. Выносит аудиторский вердикт по матрице:

| Результат | Вердикт |
|-----------|---------|
| Все факты ПРАВДА | ПРАВДА |
| Есть ПРАВДА + есть ЛОЖЬ | МАНИПУЛЯЦИЯ / ПОЛУПРАВДА |
| Только ЛОЖЬ | ЛОЖЬ |
| Все НЕТ ДАННЫХ | НЕ ПОДТВЕРЖДЕНО |

Пример: *«Эйнштейн получил Нобелевскую премию, однако в школе был двоечником и утверждал, что человек использует мозг на 10%»* → МАНИПУЛЯЦИЯ (3 ПРАВДА + 1 ЛОЖЬ из 4).

---

## Wikidata SPARQL — структурированная верификация

Для фактов с именованными сущностями система обращается к Wikidata Knowledge Graph через SPARQL endpoint (без API-ключа):

| Свойство | Wikidata ID | Пример |
|----------|-------------|--------|
| Основатель | P112 | Microsoft → Билл Гейтс, Пол Аллен |
| Дата основания | P571 | Microsoft → 1975-04-04 |
| Столица | P36 | Россия → Москва |
| Материк | P30 | Амазонка → Южная Америка |
| Спутник / орбита | P397, P398 | Луна → обращается вокруг Земли |
| Местоположение | P276 | Титаник → Атлантический океан |
| Страна | P17 | Амазонка → Бразилия, Перу, Колумбия |

Wikidata-факты используются двояко:
1. Как текстовая подсказка для LLM в промпте
2. Как **голос в ансамбле** (до +2.0 за подтверждение, до −3.0 за противоречие)

Stem matching (4-char) учитывает русскую морфологию: «Земля» (Wikidata) → «земл» ≈ «Земли» (claim).

---

## Расширенный поиск

Помимо стандартного DDG Search × 3, система использует:

- **Wikipedia entity lookup** — прямое обращение к Wikipedia API для каждой сущности из claim (до 2000 символов intro)
- **Verification queries** — LLM генерирует целевые проверочные вопросы (напр. «кто основал Microsoft?»), по ним выполняется дополнительный поиск
- **Multi-hop counter-entity search** — независимый поиск каждой сущности claim для обнаружения подмен (напр. «Стив Джобс основатель» → Apple, не Microsoft)

---

## IR-слой: защита от Semantic Overgeneralization

Проблема: CrossEncoder присваивает высокий скор общим новостным статьям из-за лексического перекрытия, задвигая специализированные разоблачения вниз. Пример — факт «провалил экзамен по **математике**», статья «провалил **вступительные экзамены**»: слово «математика» отсутствует, но CrossEncoder даёт 3.97 против 0.93 у provereno.media.

Реализованные исправления в `search.py` + `pipeline.py`:

### boost_factcheck_scores (post-CrossEncoder)

Применяется сразу после `CrossEncoder.rerank()`. Добавляет `+2.0` к скору если:
- Домен — специализированный фактчекер (snopes, provereno, politifact, …)
- Текст содержит debunk-маркеры: «миф», «разоблачение», «debunk», «hoax», …
- URL содержит паттерн `/fact-check/`, `/debunk/`, …

Финальная пересортировка гарантирует вытеснение лайфстайл-новостей.

### validate_context_entities (entity matching filter)

Для каждого документа вычисляет покрытие значимых слов из проверяемого факта:

```
penalty = 1.0 − (кол-во покрытых слов / всего значимых слов)
```

Значимые слова: длина ≥ 4 символа, не стоп-слово. Матчинг по 5-символьному псевдо-стему (перекрывает русские падежные формы: «математик» → «матем»).

- Применяется как штраф к `cross_encoder_score` (вес 1.5)
- При `penalty > 0.35` документ фильтруется, если остаётся ≥ 3 альтернатив

### Myth-buster резервирование слотов (Direction 3)

В myth-buster запросах фактчекеры добавляются в пул первыми (до 3 слотов), не давая новостным сайтам вытеснить их на этапе первичного сбора.

---

## Обучение модели

### SFT (Supervised Fine-Tuning)

```bash
python train_meta.py
```

### GRPO (Group Relative Policy Optimization)

```bash
python train_grpo.py
```

GRPO модель использует формат `<reasoning>...</reasoning><answer>...</answer>` и активирует Chain-of-Thought рассуждения в финальном промпте.

Адаптеры сохраняются в `adapters/`. При запуске `pipeline.py` автоматически выбирается лучший: GRPO > SFT > base.

---

## Конфигурация

Все параметры — в `config.py`:

```python
# Модель
ModelConfig(
    base_model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    load_in_4bit=True,
)

# Пайплайн
PipelineConfig(
    enable_nli=True,
    enable_cross_encoder=True,
    enable_claim_decomposition=True,
    nli_model_name="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    verdict_max_new_tokens=1500,
)

# Поиск
SearchConfig(
    num_results=8,      # результатов на DDG-запрос
    hl="ru",            # язык
)
```

---

## Загружаемые модели

| Модель | Размер | Устройство | Назначение |
|--------|--------|------------|------------|
| Mistral-7B-Instruct-v0.3 (4-bit) | ~4 GB | GPU | Вердикт, ключевые слова, декомпозиция |
| mDeBERTa-v3-base-xnli | ~280 MB | CPU | NLI entailment/contradiction |
| multilingual-e5-base | ~278 MB | CPU | Семантическое ранжирование (bi-encoder) |
| ms-marco-MiniLM-L-6-v2 | ~22 MB | CPU | CrossEncoder re-ranking |
| NLLB-200-distilled-600M | ~600 MB | CPU | Перевод запросов RU→EN |
