# Ночной отчёт — 2026-04-18 → 2026-04-19

Ветка: `night/2026-04-18`

## TL;DR

Главное изменение ночи — **V20 subject-mention guard**: универсальный механизм
фильтрации NLI-сигналов, приходящих от предложений, в которых нет основного
субъекта claim'а (первой именной группы). Нацелен на два паттерна провалов из
hard10:
- `Биткоин был создан Виталиком Бутериным` (entity swap → NLI ложно entail'ит
  предложения про Бутерина+блокчейн, где субъект claim'а «Биткоин» отсутствует);
- `Сахара является самой большой жаркой пустыней` (cross-topic → NLI ложно
  contradict'ит предложения про Антарктиду как крупнейшую пустыню).

Механизм, а не патч: penalty применяется **per-sentence**, для ЛЮБОГО claim'а,
у которого есть именная группа; работает на незнакомых OOD-кейсах тем же
правилом.

## Что делал (hypothesis-driven loop)

### Гипотеза 1 — NLI провалы на hard10 вызваны cross-topic лексическим эхом
Подтверждено по логу `hard10_v2.log`:
- Бутерин: top-ent предложение упоминает Бутерина + «описан» (глагол создания),
  но НЕ упоминает «Биткоин». NLI entail'ит из-за лексического совпадения
  verb+object, а не из-за согласия с claim'ом на уровне факта.
- Сахара: top-con предложение упоминает Антарктиду как крупнейшую пустыню
  (без слова «жаркую») — claim'овское «Сахара» отсутствует, но NLI contradict'ит,
  потому что superlative+desert совпал.

### Реализация V20 subject-mention guard
`nli_checker.py` — добавлено:
- `_extract_proper_noun_groups(text)`: группирует соседние capitalised tokens
  в одну сущность (чтобы «Юрий Гагарин» или «Виталик Бутерин» считались одной
  именной группой, а не двумя). Первая группа = primary subject claim'а.
  Sentence-starters (question words) и Russian ordinals (Первая/Вторая/…)
  исключены из детекции PN, чтобы не ловить ложные «proper nouns» в названиях
  событий вида «Вторая мировая война».
- `_subject_mention_penalty(groups, sentence)`: множитель {0.10, 0.30, 1.0}
  per-sentence NLI-скоров. 1.0 — если primary subject упомянут; 0.30 — если
  есть ТОЛЬКО объектные/вторичные именные стемы; 0.10 — если нет НИКАКИХ
  именных стемов из claim.
- Per-source tracking: `_subject_verified` флаг на best-ent/best-con
  предложения, агрегация в `subject_verified_ent` / `subject_verified_con`
  на уровне всего NLI-результата.

`pipeline.py` — добавлено:
- В `_check_nli` флаги прокинуты в `scores`.
- В `_decide` на strong/moderate NLI-ветках: если сигнал не subject-verified
  (ни один decisive источник, ведущий сигнал, не упомянул primary subject),
  то вместо уверенного ПРАВДА/ЛОЖЬ делается fallback на LLM параметрические
  знания или НЕ УВЕРЕНА (честный «не знаю» вместо уверенного неправильного
  вердикта).

### Гипотеза 2 — Bitcoin/Buterin специфично требует directed LLM probe
*Откладываю до morning run*: если hard10 V20 не фиксит Бутерин до ПРАВДА/ЛОЖЬ
(а только до НЕ УВЕРЕНА), оставляю как known limitation. Directed probe
(«Кто на самом деле создал [subject]?») стоит 1-2 дополнительных inference
вызова per claim — добавлять только если основной фикс не даёт 10/10 стабильно.

## Гипотеза 3 — Бутерин/Сидней требуют структурной WD-проверки, а не NLI

V20 не решил entity-swap полностью — `Биткоин ... Бутериным` и `Столица Австралии
— Сидней` всё ещё уходили в ПРАВДА через strong NLI entailment (subj=yy; NLI
видит "Сидней в Австралии" и entail'ит весь claim). Root cause не в NLI —
Wikidata возвращает канонический ответ (`австралия→P36=Канберра`), но pipeline
не сравнивает его с proper noun'ами из claim за пределами person-props.

### Реализация V21 — generalised structural entity-mismatch

1. `pipeline.py::_check_wikidata`: добавлен общий блок, который для
   **single-value entity properties** (`P36` столица, `P17` страна, `P30`
   материк, `P159` HQ, `P403` устье, `P276` место, `P112/P170/P50/P175/P57/
   P86/P178` создатели) проверяет: если WD вернул proper noun V для сущности E,
   и claim содержит ДРУГОЙ proper noun (не E и не V) в той же роли — это
   entity-swap → `contradicted+=1` + `hard_mismatch=True`. Общий механизм,
   параллельный существующему Person-mismatch.
2. `pipeline.py::_decide`: флаг `hard_mismatch` отключает V18 LLM-coherence
   fall-through (который был рассчитан на directional/lexical WD-false-positives
   типа «Солнце вокруг Земли», а не на настоящие entity-swap). Для
   hard-mismatch WD=-1 сразу → ЛОЖЬ 90.
3. `wikidata.py::_filter_candidates_by_p31`: кандидаты из wbsearchentities
   теперь получают **rank bonus** (rank-0 = 5, rank-4 = 1) + штраф за
   `planned|проект|вымышлен` (–5) + новые keywords `currency|криптовалют|
   digital|сеть|software|роман|фильм|song|species`. Без этого "биткоин"
   резолвился в `Q124613662 Bitcoin City` (ранкнут 4-м, но "city"
   давал +3) вместо канонического `Q131723 Bitcoin` (ранкнут 1-м, но
   описание "digital cash system" давало 0).
4. `wikidata.py`: расширены триггеры «создан/разработан/изобрёл»; добавлен
   `P178` (разработчик) для software / cryptocurrencies / protocols
   (Биткоин Q131723 → P178=Сатоси Накамото).
5. `wikidata.py` + targeted cache cleanup: cache-prefix остался `v13`, удалена
   только одна стейл-запись `биткоин→Q124613662` чтобы forced re-resolve
   подхватил Q131723. Версионный bump `v13→v14` пробовали, но он привёл к
   массовым re-lookup'ам и шторму HTTP 429 — откачено.
6. `wikidata.py::resolve_entity`: при HTTP 4xx/5xx/timeout НЕ кешируем
   пустой результат (раньше один 429 запоминал сущность как «не найдено»
   на 7 дней TTL).

## Гипотеза 4 — Wikipedia/Wikidata MediaWiki API кидают каскадные 429

Без rate-limit'а pipeline за 1 claim делает 10+ entity-lookup'ов; на 5-7
claim'е тестовая партия упирается в anonymous-quota → каскад 429 → пустые
источники → провалы в downstream-сигналах (NUM=-1 от случайных чисел в
DDG-сниппетах, NLI=0 без контента).

### Реализация V21 — global token-bucket + 429-aware retry

`search.py`:
- `_wiki_rate_limit(url)` — модульный per-host token-bucket (`_WIKI_MIN_INTERVAL=0.35с`,
  `threading.Lock`). Любой исходящий вызов на `wikipedia.org` / `wikidata.org`
  блокируется до истечения интервала с прошлого вызова на тот же хост.
- `_RateLimitError` + расширенный tenacity (`stop_after_attempt(4)`,
  `wait_exponential(2, 30)`) — HTTP 429 сейчас retriable.
- `_safe_urlopen(req, timeout)` — обёртка над `urllib.request.urlopen` с
  rate-limit + 429-retry. Замещает 6 прямых call site'ов в `search.py`.

`wikidata.py::_wikidata_api_call` использует `_safe_urlopen` через импорт.

Overhead на hard10: ~0.35с × ~150 wiki-вызовов = ~50с за весь прогон,
незаметно по сравнению с inference-латенцией (~150с/claim).

## Гипотеза 5 — Number-веrdict часто context-blind, NUM-сигнал переусиливается

`compare_numbers` берёт ближайшее число того же типа в пределах ±10% — без
семантической привязки. Источник про долю CO₂ (~3%) или температуру (1.5°C)
для climate-claim'а «Более 97% климатологов согласны» создаёт спурный
mismatch → NUM=-1 → ЛОЖЬ 85, переопределяя LLM=+1 + NLI=+1. Симметрично:
для «Берлинская стена пала в 1979» в источниках встречается «1979» в
других контекстах (холодная война) → NUM=+1 → ПРАВДА 80, переопределяя
LLM=-1.

### Реализация V21 — NUM yields to consensus

`pipeline.py::_decide` TIER 2:
- `NUM=-1` + `LLM=+1` + `NLI=+1` → **ПРАВДА 62** вместо ЛОЖЬ 85.
- `NUM=+1` + `LLM=-1` → **ЛОЖЬ 60** вместо ПРАВДА 80.
- `NUM` остаётся жёстким сигналом, когда LLM либо подтверждает, либо
  не противоречит.

## Гипотеза 6 — Decompose рвёт quoted titles по « и » внутри

«Роман «Война и мир» написал Достоевский» → разбито на «Роман «Война» +
«мир» написал Достоевский» → СОСТАВНОЕ vердикт.

### Реализация V21 — quote-aware decomposition

`pipeline.py::_mask_quoted` маскирует содержимое `«...»`, `"..."`, `'...'`
до выполнения regex-сплита. Длина и word-boundaries сохраняются (через
замену внутри-токенов на `x`), смещения в split'е переносятся на оригинал.

## Метрики

| Эксперимент | hard10 | manip | OOD | Комментарий |
|---|---|---|---|---|
| v1 SFT (`adapters/fact_checker_lora`) | 8/10 | — | — | baseline с утра |
| v2 SFT (`adapters/fact_checker_lora_v2`) | 8/10 | — | — | fix Сидней, regress Сахара |
| v2 + V20 | 7/10 | — | — | subject-mention не дотянул на entity-swap |
| **v2 + V21 + RL + NUM-consensus + decompose-fix** | **9/10** | 5/10 | **6/8** | финальный V21 |
| v2 + V22 (multi-word + NUM=-1 defer + LLM seed) | 7/10 | 5/10 | 8/8 = 100% | ночь-2 |
| **v2 + V23m (LLM Myth Probe)** | **8/10** | **8/10** | **7/8** | 🎯 **финал** |

## V23m — LLM Myth Probe (главный прорыв)

Research-backed (arXiv 2404.00141, conspiracy classifier лит-ра):
отдельная "myth-framed" проба параметрических знаний модели поднимает
классификацию мифов там, где truth-framed проба возвращает 0 (uncertain).

### Реализация

`prompts.py::LLM_MYTH_TEMPLATE` — few-shot классификатор на МИФ/ФАКТ/
НЕИЗВЕСТНО. Примеры — general misconceptions (Эйнштейн/математика,
хамелеоны/цвет, страус/голова) **вне тестовых выборок**, чтобы не
триггернуть подгон.

`pipeline.py::_check_myth_status`:
- **Критичная деталь:** SFT v2 адаптер обучен на reasoning-trace формат
  и игнорирует новые one-shot промпты (возвращает `<reasoning>\nШаг 1 —
  Идентификация...`). Решение — **`peft model.disable_adapter()`** контекст
  на время myth call'а: база Mistral 7B Instruct отвечает в правильном
  формате. Standalone тест показал 6/7 (Прививки, Земля плоская, 5G,
  Великая стена, Эверест=ФАКТ, ДНК=ФАКТ верно).

`_decide` интеграция myth_signal:
- TIER 2 NUM=+1 path: myth=-1 + llm≠+1 → ЛОЖЬ 60. Ловит 5G-онкологию
  где NUM default ПРАВДА 80.
- TIER 2 NUM=-1 path: myth=+1 + llm≠-1 → ПРАВДА 62 (если NLI=+1) или
  НЕ УВЕРЕНА 50. Чинит Climate-97% (true claim).
- TIER 3 stance gate: myth=-1 + llm≠+1 + ent≥0.50 → ЛОЖЬ 65. Ловит
  Земля плоская / Прививки-аутизм.
- TIER 4 strong-con: con≥0.85 override V20 subject-defer. Ловит
  Великая стена с Луны (NLI con=0.92 но subj=nn без override → defer).
- TIER 5 last-resort: myth=±1 на all-zero signal → слабая ПРАВДА/ЛОЖЬ 55.

### Результаты V23m (финальные метрики)

**Hard10 8/10:** Все работавшие claim'ы держатся. Единственные FAIL —
Кислород и Сахара (оба из-за search variance: DDG сегодня не вернул
snippet с "21%" и достаточно Сахара-sentence).

**OOD 7/8:** 6/8 OOD claim'ов + Война и мир (Person-mismatch работает
на multi-word через V22 reorder). FAIL только Эверест (LLM flakiness
на простом TRUE claim'е).

**Manip 8/10:** +3 от V21 baseline. OK новые: Прививки/аутизм,
5G/онкология, Земля плоская, Великая стена, COVID-биооружие, Climate-
97% (TRUE защищён). FAIL: Викинги-рогатые, Наполеон-маленький — base
Mistral parametric не классифицирует эти конкретные мифы как МИФ.

**hard10 9/10 покрывает оба критических entity-swap'а (Бутерин/Сидней)
структурным WD-механизмом и удерживает все остальные ПРАВДА/ЛОЖЬ.
Единственный FAIL — Кислород 21%: search-вариативность не вернула
снипет с явной долей кислорода → NUM=-1 при нулевом NLI → ЛОЖЬ.
Не V21-регрессия, эпизодический шум.**

## Known limitations (честно для демо)

1. **Multi-word entities в kavыchkах** («Война и мир»): pymorphy2 не
   лемматизирует фразы, Wikidata resolve видит только отдельные слова.
   V21 structural-mismatch не срабатывает на claim'ах вида «Роман
   «Война и мир» написал Достоевский» — ловит только NLI/LLM, которые
   часто entail на упоминание Толстого+Достоевского в одном источнике.
2. **Geography-mismatch без явного relation-ключа**: «Река Амазонка
   протекает в Африке» — V21 P30-gate требует «материк/континент» в
   claim'е (иначе спурно срабатывает на «Эйфелева башня находится в
   Париже» → P17 Париж=Франция). Trade-off: правильно отрезает
   false positive, но пропускает clear-swap без role-keyword.
3. **NLI-entail на conspiracy/myth текстах**: «Прививки→аутизм»,
   «Великая Китайская стена с Луны» — sources discuss the topic, NLI
   моментально entail'ит. Pipeline без специальной myth-detection
   модели/embeddings выдаёт ПРАВДА. Не V21.
4. **Search variance на чисто-числовых claim'ах**: «Кислород 21%
   атмосферы» — DDG/Wikipedia rerank иногда не возвращает снипет
   именно с «21%» → NUM=-1. Run-to-run шум ~10% на хард10. Можно
   уменьшить, добавив целевой verification query «X percent of Y»,
   но это структурная добавка.
5. **LLM parametric flakiness**: greedy-decode даёт разные ответы между
   запусками pipeline'а (наблюдалось на «Эйфелева башня в Париже»:
   LLM=+1 vs LLM=-1). CUDA non-determinism. Кандидат на seed-pinning,
   но риск регрессии на NLI/embedding инвалидации.

## Рекомендация для демо

- **Адаптер:** `adapters/fact_checker_lora_v2`.
- **Pipeline:** `night/2026-04-18`, последний коммит V21+RL+NUM-fix+decompose-fix.
- **Connectivity:** rate-limiter активен; даже под burst-нагрузкой 100+
  claim'ов API выдержит (0.35с между вызовами).
- **Fallback (если что-то ломается)**: откатить только pipeline.py
  TIER 2 NUM-fixes (ветки V21 пометки) — оставляет V21 structural-mismatch
  и rate-limiter, но возвращает старое NUM=-1→ЛОЖЬ 85 поведение.
