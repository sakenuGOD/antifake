# Antifake

Система автоматической проверки русскоязычных утверждений. Принимает текст
claim'а, собирает свидетельства из открытых источников (DuckDuckGo, Wikipedia,
Wikidata), пропускает через NLI и LLM, возвращает вердикт с обоснованием
и списком источников.

**Вердикты:** `ПРАВДА` · `ЛОЖЬ` · `МАНИПУЛЯЦИЯ` · `ЧАСТИЧНО ПОДТВЕРЖДЕНО`
· `НЕ ПОДТВЕРЖДЕНО` · `СКАМ`.

---

## Точность на проверочных наборах

Система оценивается на **28 проверочных утверждениях** трёх типов,
покрывающих разные паттерны фальсификаций. Все три набора запускаются
детерминированно на одном адаптере (`fact_checker_lora_v2`); метрики
ниже — с финальной версии pipeline'а (HEAD ветки `main`).

| Проверочный набор | Что проверяет | Точность |
|---|---|---|
| **hard10** (10 claim'ов) | Канонические факты и их подмены — Пушкин/Онегин, Гагарин, Солнце-Земля, Эйфелева в Париже, Кислород 21%, ВМВ 1943, ДНК, Бутерин/Биткоин, Сидней/Австралия, Сахара | **9 / 10  (90 %)** |
| **OOD probe** (8 claim'ов) | Out-of-distribution — Война и мир/Достоевский, Ньютон/относительность, Канада/Торонто, Амазонка/Африка, Москва/Россия, Эверест, Берлинская стена 1979, скорость света. Ни один claim из этого набора не участвует в обучении и не пересекается с hard10. | **7 / 8  (87.5 %)** |
| **Manipulative** (10 claim'ов) | Мифы, конспирологии, псевдонаука — лунный заговор, миф «10 % мозга», викинги с рогами, прививки/аутизм, 5G/рак, Земля плоская, стена с Луны, COVID-биооружие, 97 % климатологов, рост Наполеона | **8 / 10  (80 %)** |
| **Итого** | | **24 / 28  (86 %)** |

### Что за этими цифрами

- **hard10 и OOD** — структурные факты (именные/локационные/датовые
  свопы). Главный механизм — сверка с Wikidata: если в утверждении
  «разработчик Биткоин = Виталик Бутерин», а Wikidata говорит
  «P178 = Сатоси Накамото», это детектируется автоматически без
  обращения к LLM-судье. На этом классе точность стабильно 85–90 %.
- **Manipulative** — мифы без явных структурных противоречий
  («Земля плоская», «викинги с рогами»). Здесь срабатывает отдельная
  myth-проба через `peft.disable_adapter()` к базовому Mistral 7B —
  он классифицирует утверждение как **МИФ / ФАКТ / НЕИЗВЕСТНО**.
  Точность ниже, потому что базовая модель не покрывает все известные
  мифы параметрически, а поиск не всегда возвращает debunk-источники.

### Известные ограничения (честно)

- **Поисковая вариативность.** DuckDuckGo и Wikipedia могут вернуть
  разный набор источников между запусками — в пределах одного прогона
  результат детерминирован, но межсессионный flip на пограничных
  claim'ах возможен (наблюдалось на ВМВ-1943, Эвересте, Кислороде).
- **LLM run-to-run flakiness.** Параметрическая проба Mistral 7B
  иногда возвращает `НЕИЗВЕСТНО` на claim'е, где в другом запуске
  ответила уверенно — артефакт 4-битной bitsandbytes-квантизации.
  Размер выборки (28 claim'ов) означает 1 flip ≈ ±3.5 % в метрике.
- **Myths без structural signal.** Для специфичных мифов, которые
  базовый Mistral не помнит (Викинги-рогатые, рост Наполеона),
  система возвращает `НЕ УВЕРЕНА` — честный defer вместо уверенно
  неправильного вердикта.

---

## Архитектура

```
Утверждение
    │
    ▼
[PARSE]        классификация claim'а, извлечение чисел/дат, detect scam
    │
    ▼
[DECOMPOSE]    rule-based split по союзам (quote-aware); Mistral для сложных
    │
    ▼
[SEARCH]       DDG (параллельно 3 frame) + Wikipedia entity lookup
    │          + verification queries + counter-search (debunk framing)
    │          + Wikipedia «common misconceptions» frame
    │          + quoted-number query (для claim'ов с числом+единицей)
    │          + rate-limiter 0.35с/host + 429-aware retry
    ▼
[RANK]         multilingual-e5-base (bi-encoder) + mmarco cross-encoder reranker
    │          + boost фактчекеров + TRUSTED_SOURCES фильтр
    ▼
[EVIDENCE]     4 параллельных сигнала:
    │          • Wikidata SPARQL — структурированная KG проверка
    │          •                   structural entity-mismatch для
    │          •                   single-value props (столица, страна,
    │          •                   материк, автор, основатель, разработчик)
    │          • NUM comparison — deterministic числовая сверка
    │          • NLI — mDeBERTa sentence-level +
    │                  cross-encoder tiebreaker + doc-level CE fallback +
    │                  subject-mention guard
    │          • LLM knowledge probe (параметрическая память Mistral)
    │          • LLM myth probe (через disable_adapter — base Mistral,
    │                  не SFT — классифицирует как МИФ/ФАКТ)
    ▼
[DECIDE]       priority-based tree по сигналам:
    │          TIER 1: WD hard-mismatch → ЛОЖЬ 90
    │          TIER 2: NUM ±1 (с LLM/myth consensus override)
    │          TIER 3: debunk-aware stance gate (myth detection)
    │          TIER 4: NLI gap zones (strong / moderate / ambiguous)
    │                  + subject-verification gates
    │                  + LLM coherence overrides
    │          TIER 5: LLM parametric / myth fallback
    ▼
[EXPLAIN]      LLM генерирует объяснение вердикта (verdict уже определён,
               модель только объясняет почему)
    ▼
Вердикт + credibility_score (0-100) + reasoning + sources
```

LLM как объясняющий слой, а не как судья — все вердикты рассчитываются
детерминированно из signal-сигналов. Это делает вердикты воспроизводимыми
и debug'ным.

---

## Стек

| Компонент | Модель |
|---|---|
| Base LLM | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` |
| SFT адаптер | `adapters/fact_checker_lora_v2` (custom QLoRA r=16) |
| Bi-encoder ranker | `intfloat/multilingual-e5-base` |
| Cross-encoder reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| NLI (основной) | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` |
| NLI (cross-encoder fallback) | `cross-encoder/nli-deberta-v3-base` |
| Translator (RU→EN fallback) | `facebook/nllb-200-distilled-600M` |
| NER + лемматизация | Natasha + pymorphy2 |
| Knowledge graph | Wikidata SPARQL endpoint |
| Поиск | DuckDuckGo (async) + MediaWiki API |

GPU: 4-bit Mistral работает на ~5 GB VRAM, NLI/reranker — на CPU,
embeddings — на CPU. Протестировано на RTX 5070 12 GB (Blackwell sm_120);
`xformers` отключён — несовместим с sm_120.

---

## Установка

```bash
git clone https://github.com/sakenuGOD/antifake.git
cd antifake
python -m venv venv
source venv/bin/activate          # Linux/Mac
# или:
venv\Scripts\activate             # Windows

pip install -r requirements.txt
```

### Переменные окружения

| Переменная | Обязательна | Назначение |
|---|---|---|
| `SERPAPI_API_KEY` | нет | Google Search fallback (без ключа — только DDG) |
| `HF_TOKEN` | нет | higher rate limit для скачивания весов с HuggingFace |
| `ANTIFAKE_DETERMINISTIC=1` | нет | strict-reproducibility режим (cuDNN deterministic) |

Сохраняются в `.env` в корне проекта.

---

## Использование

### Streamlit UI

```bash
streamlit run app.py
```

Интерфейс показывает 14 stage'ей pipeline'а в реальном времени: парсинг,
декомпозиция, ключевые слова, найденные источники, Wikidata факты, NUM
сравнения, NLI сигналы, debunk count, LLM probes, финальный вердикт,
объяснение.

### Python API

```python
from pipeline import FactCheckPipeline

pipeline = FactCheckPipeline(adapter_path="adapters/fact_checker_lora_v2")

result = pipeline.check("Роман «Война и мир» написал Фёдор Достоевский")

print(result["verdict"])              # "ЛОЖЬ"
print(result["credibility_score"])    # 10
print(result["reasoning"])            # текст обоснования
for src in result["sources"][:3]:
    print(src["link"], src["title"])
```

Сигнатура результата — см. `pipeline.py::check()`.

### CLI

```bash
python main.py "Москва — столица России"
```

---

## Структура проекта

```
antifake/
├── app.py                       # Streamlit entry point
├── main.py                      # CLI entry point
├── pipeline.py                  # Основной evidence-first pipeline
├── search.py                    # DDG/Wiki/Wikidata + rate-limiter + frames
├── nli_checker.py               # mDeBERTa + cross-encoder + doc-level CE
├── wikidata.py                  # SPARQL + structural entity-mismatch
├── counter_search.py            # Multi-frame debunk/verify queries
├── model.py                     # Mistral + LoRA loader
├── prompts.py                   # Knowledge + myth probe templates
├── claim_parser.py              # Числа/даты/локации + scam patterns
├── embeddings.py                # Semantic ranker + CE reranker
├── evidence_tiers.py            # T1-T3 authority weighting
├── nlp_russian.py               # Natasha/pymorphy2 helpers
├── source_credibility.py        # Domain trust boosting
├── cache.py                     # Disk-based search cache (24h TTL)
├── fact_cache.py                # Verified facts cache
├── config.py                    # Все константы и пороги
├── utils.py
│
├── tests/                       # Regression suites
│   ├── _path.py                 # sys.path shim
│   ├── test_hard10.py           # 10 canonical claims
│   ├── test_manipulative.py     # 10 conspiracy/myth claims
│   ├── test_ood_probe.py        # 8 out-of-distribution claims
│   └── test_*.py (архив)
│
├── eval/                        # Metrics on larger datasets
│   ├── evaluate.py
│   ├── evaluate_universal.py
│   └── quick_eval.py
│
├── scripts/                     # Training / data-gen / housekeeping
│   ├── train.py                 # SFT
│   ├── train_grpo.py            # Reward-based
│   ├── generate_*.py            # Обучающие данные
│   ├── merge_training_data.py
│   ├── audit_training_data.py
│   ├── download_dataset.py
│   ├── sanity_check.py
│   ├── summarize_night.py
│   └── run_training.sh, setup_and_train.sh, night_run.{sh,ps1}
│
├── data/                        # Датасеты + результаты + кэши
│   ├── train_*.jsonl            # Обучающие выборки
│   ├── hard10_results.json
│   ├── manipulative_results.json
│   ├── ood_probe_results.json
│   └── wikidata_cache.json
│
├── adapters/                    # LoRA / GRPO адаптеры (weights ignored in git)
├── logs/                        # Прогоны pipeline'а (gitignored)
└── requirements.txt
```

### Импорты в подпапках

Файлы в `tests/`, `eval/`, `scripts/` начинаются с
`import _path  # noqa: F401,E402`. Этот shim добавляет корень
проекта в `sys.path`, чтобы `from pipeline import …` работал
без установки проекта как пакета.

---

## Обучение

### SFT на собственных данных

```bash
python scripts/train.py \
    --dataset data/train_v2_combined.jsonl \
    --output adapters/fact_checker_lora_v3 \
    --epochs 4 \
    --learning-rate 2e-5 \
    --resume adapters/fact_checker_lora_v2   # опционально
```

### GRPO (reward-based)

```bash
python scripts/train_grpo.py \
    --load-adapter adapters/fact_checker_lora_v2 \
    --steps 200
```

### Генерация данных

```bash
python scripts/generate_russian_data.py --output data/train.jsonl
python scripts/generate_failure_patterns.py --output data/failure_patterns.jsonl
python scripts/merge_training_data.py \
    data/train.jsonl data/failure_patterns.jsonl \
    --output data/train_v2_combined.jsonl
```

---

## Тесты

```bash
# Маленькие regression-suites (~15-25 минут на suite)
python tests/test_hard10.py       --adapter adapters/fact_checker_lora_v2
python tests/test_manipulative.py --adapter adapters/fact_checker_lora_v2
python tests/test_ood_probe.py    --adapter adapters/fact_checker_lora_v2

# Отладка одного claim'а
python tests/debug_single.py "Bitcoin создал Виталик Бутерин"
```

Результаты пишутся в `data/*_results.json` + детальный лог в `logs/`.

---

## Конфигурация

Все пороги и настройки в `config.py`:

| Класс | Что настраивает |
|---|---|
| `ModelConfig` | Base model path, max_seq_length, load_in_4bit |
| `PipelineConfig` | NLI device, cross-encoder model, re-ranker on/off |
| `SearchConfig` | DDG timeout, num_results, SerpAPI |
| `DecisionThresholds` | strong_gap, moderate_gap, num_nli_override |

---

## Внешние API

| Сервис | Rate limit | Fallback |
|---|---|---|
| DuckDuckGo | нет формального, но 429 при burst | retry + async parallel |
| Wikipedia / Wikidata | ~30 req/s anonymous | 0.35s/host token bucket + 429 retry |
| SerpAPI | по плану ключа | DDG (основной канал) |

Rate-limiter реализован в `search.py::_wiki_rate_limit` — глобальный
token bucket per-host + tenacity retry с 2-30с exponential backoff.

---

## Лицензия

См. `LICENSE` (если присутствует).
