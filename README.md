# Antifake

Automated fact-checker for Russian-language claims. Takes a claim, gathers evidence from open sources (DuckDuckGo, Wikipedia, Wikidata), runs it through NLI + LLM, and returns a verdict with reasoning and source list.

**Verdicts** (returned verbatim in Russian): `ПРАВДА` (true) · `ЛОЖЬ` (false) · `НЕ УВЕРЕНА` (unsure) · `СКАМ` (scam).

---

## Accuracy

The system is evaluated on **28 probe claims**, split into three sets by falsification type. All runs are deterministic, on a single adapter (`fact_checker_lora_v2`).

<table>
<tr>
    <td width="33%" align="center">
        <h3>📘 Core set</h3>
        <sub>10 canonical claims</sub><br><br>
        <h2>9 / 10</h2>
        <b>90 %</b>
    </td>
    <td width="33%" align="center">
        <h3>🌐 Out-of-distribution</h3>
        <sub>8 claims not seen in training</sub><br><br>
        <h2>7 / 8</h2>
        <b>87.5 %</b>
    </td>
    <td width="33%" align="center">
        <h3>🕵️ Myths &amp; fakes</h3>
        <sub>10 conspiracies and misconceptions</sub><br><br>
        <h2>8 / 10</h2>
        <b>80 %</b>
    </td>
</tr>
</table>

<p align="center">
<b>Total: 24 of 28 · 86 %</b>
</p>

<br>

### 📘 Core set

Canonical facts and their substitutions — probes the main falsification patterns:

- **Name swaps.** Who created Bitcoin? Who wrote *Eugene Onegin*?
- **Location swaps.** What's the capital of Australia? Where is the Eiffel Tower?
- **Date swaps.** When did World War II end?
- **Directional facts.** Does the Sun orbit the Earth?
- **Numerical facts.** How much oxygen is in the atmosphere?

Main mechanism — Wikidata lookup. If the claim says "Bitcoin creator = Vitalik Buterin" and Wikidata says `P178 = Satoshi Nakamoto`, the mismatch is detected automatically without calling the LLM judge.

---

### 🌐 Out-of-distribution set

Claims **not seen in training or in the core set**. Measures whether the mechanisms generalise to fresh claims.

Covers:

- **Multi-word entities in quotes.** *«Роман „Война и мир“ написал Достоевский»* (claims Dostoevsky wrote *War and Peace*)
- **Geographic swaps.** Capital of Canada = Toronto
- **Scientific authorship.** Newton formulated the theory of relativity
- **Event dates.** Berlin Wall fell in 1979
- **Common truths.** Moscow is the capital of Russia; Everest is the tallest mountain

---

### 🕵️ Myths & fakes

Claims with no clear structural contradiction in the knowledge graph:

- Moon landing hoax
- "We only use 10% of our brain"
- Vikings wore horned helmets
- Vaccines cause autism
- 5G radiation causes cancer
- Flat Earth
- Great Wall of China visible from the Moon
- COVID-19 engineered as a bioweapon
- >97% of climatologists agree on anthropogenic warming
- Napoleon was short

A separate **myth probe** runs here via `peft.disable_adapter()` against the base Mistral 7B — it classifies the claim as **МИФ / ФАКТ / НЕИЗВЕСТНО** (myth / fact / unknown). Accuracy is lower because the base model doesn't parametrically know every named myth, and search doesn't always surface a debunk source for a given claim.

---

### ⚠️ Known limitations

**Search variability.** DuckDuckGo and Wikipedia can return different source sets across runs — within a single run the result is deterministic, but cross-session flips on borderline claims are possible.

**LLM run-to-run flakiness.** The Mistral 7B parametric probe sometimes returns `НЕИЗВЕСТНО` on a claim it answered confidently in another run — an artefact of 4-bit bitsandbytes quantisation. Sample size (28 claims) means 1 flip ≈ ±3.5 % in the metric.

**Myths without structural signal.** For niche myths the base Mistral doesn't know parametrically (horned Vikings, Napoleon's height), the system returns `НЕ УВЕРЕНА` — an honest defer rather than a confidently wrong verdict.

---

## Architecture

```
Claim
    │
    ▼
[PARSE]        claim classification, extract numbers/dates, detect scam
    │
    ▼
[DECOMPOSE]    rule-based split on conjunctions (quote-aware); Mistral for complex cases
    │
    ▼
[SEARCH]       DDG (3 parallel frames) + Wikipedia entity lookup
    │          + verification queries + counter-search (debunk framing)
    │          + Wikipedia "common misconceptions" frame
    │          + quoted-number query (for claims with number+unit)
    │          + rate-limiter 0.35s/host + 429-aware retry
    ▼
[RANK]         multilingual-e5-base (bi-encoder) + mmarco cross-encoder reranker
    │          + fact-checker boost + TRUSTED_SOURCES filter
    ▼
[EVIDENCE]     4 parallel signals:
    │          • Wikidata SPARQL — structured KG check
    │                              structural entity-mismatch for
    │                              single-value props (capital, country,
    │                              continent, author, founder, developer)
    │          • NUM comparison — deterministic numeric check
    │          • NLI — mDeBERTa sentence-level +
    │                  cross-encoder tiebreaker + doc-level CE fallback +
    │                  subject-mention guard
    │          • LLM knowledge probe (Mistral parametric memory)
    │          • LLM myth probe (via disable_adapter — base Mistral,
    │                            not SFT — classifies as МИФ/ФАКТ)
    ▼
[DECIDE]       priority-based tree over the signals:
    │          TIER 1: WD hard-mismatch → ЛОЖЬ 90
    │          TIER 2: NUM ±1 (with LLM/myth consensus override)
    │          TIER 3: debunk-aware stance gate (myth detection)
    │          TIER 4: NLI gap zones (strong / moderate / ambiguous)
    │                  + subject-verification gates
    │                  + LLM coherence overrides
    │          TIER 5: LLM parametric / myth fallback
    ▼
[EXPLAIN]      LLM generates the reasoning text (the verdict is already
               decided — the model only explains why)
    ▼
Verdict + credibility_score (0-100) + reasoning + sources
```

LLM is an explainer, not a judge — all verdicts are derived deterministically from signals. This keeps verdicts reproducible and debuggable.

---

## Stack

| Component | Model |
|---|---|
| Base LLM | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` |
| SFT adapter | `adapters/fact_checker_lora_v2` (custom QLoRA r=16) |
| Bi-encoder ranker | `intfloat/multilingual-e5-base` |
| Cross-encoder reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| NLI (primary) | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` |
| NLI (cross-encoder fallback) | `cross-encoder/nli-deberta-v3-base` |
| Translator (RU→EN fallback) | `facebook/nllb-200-distilled-600M` |
| NER + lemmatisation | Natasha + pymorphy2 |
| Knowledge graph | Wikidata SPARQL endpoint |
| Search | DuckDuckGo (async) + MediaWiki API |

GPU: 4-bit Mistral fits in ~5 GB VRAM, NLI/reranker run on CPU, embeddings on CPU. Tested on RTX 5070 12 GB (Blackwell sm_120); `xformers` is disabled — incompatible with sm_120.

---

## Install

```bash
git clone https://github.com/sakenuGOD/antifake.git
cd antifake
python -m venv venv
source venv/bin/activate          # Linux/Mac
# or:
venv\Scripts\activate             # Windows

pip install -r requirements.txt
```

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `SERPAPI_API_KEY` | no | Google Search fallback (without it — DDG only) |
| `HF_TOKEN` | no | higher HuggingFace rate limit when pulling weights |
| `ANTIFAKE_DETERMINISTIC=1` | no | strict-reproducibility mode (cuDNN deterministic) |

Stored in `.env` at the project root.

---

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

The UI shows 13 pipeline stages in real time: parsing, decomposition, keywords, found sources, Wikidata facts, NUM comparisons, NLI signals, debunk count, LLM probes, decision, aggregation, and explanation.

### Python API

```python
from pipeline import FactCheckPipeline

pipeline = FactCheckPipeline(adapter_path="adapters/fact_checker_lora_v2")

result = pipeline.check("Роман «Война и мир» написал Фёдор Достоевский")

print(result["verdict"])              # "ЛОЖЬ"
print(result["credibility_score"])    # 10
print(result["reasoning"])            # reasoning text
for src in result["sources"][:3]:
    print(src["link"], src["title"])
```

Full result signature — see `pipeline.py::check()`.

### CLI

```bash
python main.py "Москва — столица России"
```

---

## Project layout

```
antifake/
├── app.py                       # Streamlit entry point
├── main.py                      # CLI entry point
├── pipeline.py                  # Main evidence-first pipeline
├── search.py                    # DDG/Wiki/Wikidata + rate-limiter + frames
├── nli_checker.py               # mDeBERTa + cross-encoder + doc-level CE
├── wikidata.py                  # SPARQL + structural entity-mismatch
├── counter_search.py            # Multi-frame debunk/verify queries
├── model.py                     # Mistral + LoRA loader
├── prompts.py                   # Knowledge + myth probe templates
├── claim_parser.py              # Numbers/dates/locations + scam patterns
├── embeddings.py                # Semantic ranker + CE reranker
├── evidence_tiers.py            # T1-T3 authority weighting
├── nlp_russian.py               # Natasha/pymorphy2 helpers
├── source_credibility.py        # Domain trust boosting
├── cache.py                     # Disk-based search cache (24h TTL)
├── fact_cache.py                # Verified facts cache
├── config.py                    # All constants and thresholds
├── utils.py
│
├── tests/                       # Regression suites
│   ├── _path.py                 # sys.path shim
│   ├── test_hard10.py           # 10 canonical claims
│   ├── test_manipulative.py     # 10 conspiracy/myth claims
│   ├── test_ood_probe.py        # 8 out-of-distribution claims
│   └── test_*.py (archive)
│
├── eval/                        # Metrics on larger datasets
│   ├── evaluate.py
│   ├── evaluate_universal.py
│   └── quick_eval.py
│
├── scripts/                     # Training / data-gen / housekeeping
│   ├── train.py                 # SFT
│   ├── train_grpo.py            # Reward-based
│   ├── generate_*.py            # Training data
│   ├── merge_training_data.py
│   ├── audit_training_data.py
│   ├── download_dataset.py
│   ├── sanity_check.py
│   ├── summarize_night.py
│   └── run_training.sh, setup_and_train.sh, night_run.{sh,ps1}
│
├── data/                        # Datasets + results + caches
│   ├── train_*.jsonl            # Training sets
│   ├── hard10_results.json
│   ├── manipulative_results.json
│   ├── ood_probe_results.json
│   └── wikidata_cache.json
│
├── adapters/                    # LoRA / GRPO adapters (weights gitignored)
├── logs/                        # Pipeline runs (gitignored)
└── requirements.txt
```

### Imports inside subfolders

Files in `tests/`, `eval/`, `scripts/` start with `import _path  # noqa: F401,E402`. This shim adds the project root to `sys.path` so `from pipeline import …` works without installing the project as a package.

---

## Training

### SFT on your own data

```bash
python scripts/train.py \
    --dataset data/train_v2_combined.jsonl \
    --output adapters/fact_checker_lora_v3 \
    --epochs 4 \
    --learning-rate 2e-5 \
    --resume adapters/fact_checker_lora_v2   # optional
```

### GRPO (reward-based)

```bash
python scripts/train_grpo.py \
    --load-adapter adapters/fact_checker_lora_v2 \
    --steps 200
```

### Data generation

```bash
python scripts/generate_russian_data.py --output data/train.jsonl
python scripts/generate_failure_patterns.py --output data/failure_patterns.jsonl
python scripts/merge_training_data.py \
    data/train.jsonl data/failure_patterns.jsonl \
    --output data/train_v2_combined.jsonl
```

---

## Tests

```bash
# Small regression suites (~15-25 min per suite)
python tests/test_hard10.py       --adapter adapters/fact_checker_lora_v2
python tests/test_manipulative.py --adapter adapters/fact_checker_lora_v2
python tests/test_ood_probe.py    --adapter adapters/fact_checker_lora_v2

# Debug a single claim
python tests/debug_single.py "Bitcoin создал Виталик Бутерин"
```

Results land in `data/*_results.json` with a detailed log in `logs/`.

---

## Configuration

All thresholds and settings live in `config.py`:

| Class | What it controls |
|---|---|
| `ModelConfig` | base model path, max_seq_length, load_in_4bit |
| `PipelineConfig` | NLI device, cross-encoder model, re-ranker on/off |
| `SearchConfig` | DDG timeout, num_results, SerpAPI |
| `DecisionThresholds` | strong_gap, moderate_gap, num_nli_override |

---

## External APIs

| Service | Rate limit | Fallback |
|---|---|---|
| DuckDuckGo | no formal limit, 429 under burst | retry + async parallel |
| Wikipedia / Wikidata | ~30 req/s anonymous | 0.35s/host token bucket + 429 retry |
| SerpAPI | per-plan key | DDG (primary channel) |

Rate-limiter lives in `search.py::_wiki_rate_limit` — global per-host token bucket with tenacity retry (2-30s exponential backoff).

---

## License

See `LICENSE` (if present).
