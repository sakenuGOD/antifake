"""Antifake — Streamlit UI.

Переработанный фронт: live-прогресс по этапам, sticky status bar с
анимированным "думаю...", большой verdict-card, чистое обоснование,
rich-source карточки. Post-processing обоснования убирает LLM-артефакты
(ФАКТЫ:, УВЕРЕННОСТЬ:, ВЕРДИКТ:, ИСТОЧНИКИ: и т.п.).
"""

import os
import re
import sys
import time

# Windows console is cp1251 by default; Streamlit-invoked prints with
# Unicode glyphs crash under UnicodeEncodeError. Force UTF-8 early.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import streamlit as st

st.set_page_config(
    page_title="Antifake · Проверка достоверности",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Layout ── */
.block-container { padding-top: 1.5rem !important; max-width: 880px !important; }
#MainMenu, footer { visibility: hidden; }

/* ── Hero ── */
.hero {
    text-align: center; padding: 4px 0 24px 0;
}
.hero h1 {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0; letter-spacing: -0.03em;
}
.hero .subtitle { color: #94a3b8; font-size: 1rem; margin: 0; }

/* ── Input area ── */
.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 4px;
}
.stTextArea textarea {
    background: transparent !important;
    border: none !important;
    font-size: 1.05rem !important;
    padding: 14px !important;
}
.stTextArea label { display: none !important; }

.example-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 4px 0; }
.chip-hint { color: #64748b; font-size: 0.82rem; margin-right: 6px; align-self: center; }
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: transform 0.15s ease !important;
}
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    border: none !important;
    font-size: 1.05rem !important;
    padding: 12px 20px !important;
}

/* ── Thinking strip ── */
.thinking-strip {
    display: flex; align-items: center; gap: 14px;
    padding: 16px 20px;
    background: linear-gradient(90deg, rgba(59,130,246,0.08), rgba(139,92,246,0.08));
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px;
    margin: 18px 0;
    animation: shimmer 3s ease infinite;
}
@keyframes shimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.thinking-strip .dots::after {
    content: '';
    animation: dots 1.4s steps(4, end) infinite;
}
@keyframes dots { 0%{content:'';} 25%{content:'·';} 50%{content:'··';} 75%{content:'···';} 100%{content:'';} }
.thinking-strip .spinner {
    width: 18px; height: 18px; flex-shrink: 0;
    border: 2.5px solid rgba(139,92,246,0.2);
    border-top-color: #a78bfa;
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.thinking-strip .text { color: #e2e8f0; font-weight: 500; flex: 1; }
.thinking-strip .counter { color: #94a3b8; font-variant-numeric: tabular-nums; font-size: 0.85rem; }

/* ── Progress ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899) !important;
}

/* ── Verdict cards (big) ── */
.verdict {
    text-align: center; padding: 36px 20px 28px;
    border-radius: 20px;
    margin: 20px 0 24px;
    position: relative;
    animation: pop 0.4s ease-out;
    border: 2px solid;
}
@keyframes pop {
    0% { opacity: 0; transform: scale(0.92); }
    100% { opacity: 1; transform: scale(1); }
}
.verdict .icon { font-size: 3.4rem; display: block; margin-bottom: 6px; line-height: 1; }
.verdict .label { font-size: 2.1rem; font-weight: 800; letter-spacing: 0.06em; margin: 4px 0; }
.verdict .sub { color: #94a3b8; font-size: 0.95rem; }

.v-true  { border-color: #22c55e; background: radial-gradient(circle at 50% 0%, rgba(34,197,94,0.18), rgba(34,197,94,0.02) 70%); }
.v-true  .label { color: #4ade80; }
.v-false { border-color: #ef4444; background: radial-gradient(circle at 50% 0%, rgba(239,68,68,0.18), rgba(239,68,68,0.02) 70%); }
.v-false .label { color: #f87171; }
.v-scam  { border-color: #f97316; background: radial-gradient(circle at 50% 0%, rgba(249,115,22,0.22), rgba(249,115,22,0.02) 70%); }
.v-scam  .label { color: #fb923c; }
.v-unsure { border-color: #eab308; background: radial-gradient(circle at 50% 0%, rgba(234,179,8,0.16), rgba(234,179,8,0.02) 70%); }
.v-unsure .label { color: #facc15; }
.v-composite { border-color: #a855f7; background: radial-gradient(circle at 50% 0%, rgba(168,85,247,0.18), rgba(168,85,247,0.02) 70%); }
.v-composite .label { color: #c084fc; }

/* ── Metric pills ── */
.metric-row { display: flex; gap: 12px; margin: 0 0 20px 0; }
.metric-pill {
    flex: 1; padding: 14px 18px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
}
.metric-pill .label { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-pill .value { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
.metric-pill .suffix { color: #64748b; font-size: 0.9rem; font-weight: 400; margin-left: 3px; }

/* ── Reasoning ── */
.reasoning-card {
    padding: 20px 22px;
    background: rgba(255,255,255,0.02);
    border-left: 3px solid #8b5cf6;
    border-radius: 8px;
    margin: 14px 0 20px;
    color: #cbd5e1; line-height: 1.6;
}
.reasoning-card h4 { margin: 0 0 10px 0; color: #a78bfa; font-size: 0.85rem; letter-spacing: 0.12em; text-transform: uppercase; }

/* ── Sub-claims ── */
.sub-list { margin: 10px 0 18px; display: flex; flex-direction: column; gap: 6px; }
.sub-item {
    padding: 10px 14px;
    background: rgba(255,255,255,0.02);
    border-radius: 8px;
    border-left: 3px solid;
    font-size: 0.95rem;
}
.sub-true  { border-left-color: #22c55e; }
.sub-false { border-left-color: #ef4444; }
.sub-unsure { border-left-color: #eab308; }
.sub-item .tag { font-weight: 700; margin-right: 10px; }

/* ── Sources ── */
.sources-title { color: #94a3b8; font-size: 0.82rem; letter-spacing: 0.1em; text-transform: uppercase; margin: 18px 0 8px; }
.src-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; }
.src-card {
    padding: 10px 12px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    text-decoration: none !important;
    color: #cbd5e1 !important;
    transition: all 0.15s ease;
    display: block;
}
.src-card:hover { background: rgba(139,92,246,0.08); border-color: rgba(139,92,246,0.3); }
.src-card .domain { color: #60a5fa; font-size: 0.78rem; font-weight: 600; }
.src-card .title { display: block; font-size: 0.9rem; margin-top: 3px; line-height: 1.35;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Stage timeline ── */
.stage-item {
    padding: 8px 14px; margin: 2px 0;
    border-radius: 8px;
    display: flex; align-items: center; gap: 10px;
    font-size: 0.92rem; color: #cbd5e1;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
}
.stage-item.done { color: #64748b; }
.stage-item.done .st-icon { color: #22c55e; }
.stage-item.current {
    border-color: rgba(139,92,246,0.3);
    background: rgba(139,92,246,0.06);
    color: #e2e8f0;
}
.stage-item.current .st-icon {
    color: #a78bfa;
    animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.15); }
}
.stage-item .st-icon { flex-shrink: 0; width: 18px; text-align: center; }
.stage-item .st-data { color: #94a3b8; font-size: 0.82rem; margin-left: auto; font-variant-numeric: tabular-nums; }

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  Pipeline loader
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🧠 Загружаю модели (первый раз ~40с)...")
def load_pipeline():
    from pipeline import FactCheckPipeline
    from config import SearchConfig
    from model import find_best_adapter

    search_config = SearchConfig(api_key=os.environ.get("SERPAPI_API_KEY", ""))
    adapter_path = find_best_adapter()
    return FactCheckPipeline(adapter_path=adapter_path, search_config=search_config)


# ══════════════════════════════════════════════════════════════════
#  Reasoning cleanup
# ══════════════════════════════════════════════════════════════════
# Очищает explanation LLM от шаблонных маркеров и мусорных префиксов.
_LLM_ARTIFACTS = [
    # Шаблонные шапки
    (r'^\s*(?:ВЕРДИКТ|ОБОСНОВАНИЕ|ФАКТЫ|ИСТОЧНИКИ|УВЕРЕННОСТЬ|АНАЛИЗ)\s*:\s*', ''),
    # Повторы в середине
    (r'\s*\|\s*(?:ВЕРДИКТ|ОБОСНОВАНИЕ|ФАКТЫ|ИСТОЧНИКИ|УВЕРЕННОСТЬ|АНАЛИЗ)\s*:\s*', ' '),
    (r'\s*(?:ВЕРДИКТ|ОБОСНОВАНИЕ|ФАКТЫ|ИСТОЧНИКИ|УВЕРЕННОСТЬ|АНАЛИЗ)\s*:\s*', ' '),
    # "[Источник] сообщает:" — LLM often hallucinates these
    (r'\[Источник\]\s*сообщает:\s*', ''),
    (r'«\[Источник\][^»]*»', ''),
    # Проценты в начале "NN% НЕ ПОДТВЕРЖДЕНО" тип
    (r'^\s*\d+%?\s*(?:НЕ\s+ПОДТВЕРЖДЕНО|ПОДТВЕРЖДЕНО)[^\.]*\.?\s*', ''),
    # "Подтверждающие источники не найдены" — убираем если верхний верилдикт ПРАВДА
    # (ловит inconsistency)
    # Просто trim
    (r'\s{2,}', ' '),
]


def clean_reasoning(text: str) -> str:
    """Удаляет шаблонные маркеры и повторяющиеся префиксы."""
    if not text:
        return ""
    cleaned = text.strip()
    for pattern, repl in _LLM_ARTIFACTS:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE | re.MULTILINE)
    # Дополнительно: если есть несколько "ВЕРДИКТ:" — берём текст ПОСЛЕ первого
    # (LLM иногда дублирует блок)
    if cleaned.count('ОБОСНОВАНИЕ:') > 1 or cleaned.count('обоснование:') > 1:
        # split на обоснования, оставить самое длинное
        parts = re.split(r'(?:ОБОСНОВАНИЕ|обоснование)\s*:\s*', cleaned)
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            cleaned = max(parts, key=len)
    # Removing leftover separators at start / end
    cleaned = cleaned.strip(" |.,;—-")
    # Capitalise first letter
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


# ══════════════════════════════════════════════════════════════════
#  Example claims (chips)
# ══════════════════════════════════════════════════════════════════
_EXAMPLES = [
    "Александр Пушкин написал роман в стихах Евгений Онегин",
    "Столица Австралии — Сидней",
    "Биткоин был создан Виталиком Бутериным",
    "Викинги носили шлемы с рогами в боевых сражениях",
]


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    # ── Hero ──
    st.markdown("""
    <div class="hero">
        <h1>Antifake</h1>
        <p class="subtitle">Система автоматической проверки достоверности утверждений</p>
    </div>
    """, unsafe_allow_html=True)

    pipeline = load_pipeline()
    if pipeline is None:
        st.error("Не удалось загрузить пайплайн.")
        st.stop()

    # ── Examples first (must be rendered BEFORE text_area so we can
    # pre-populate its `value` from a separate state key without hitting
    # Streamlit's "cannot modify widget key after instantiation" error) ──
    if "prefill_claim" not in st.session_state:
        st.session_state["prefill_claim"] = ""

    st.markdown("#### Введите утверждение")

    st.markdown('<div class="chip-hint">Быстрые примеры:</div>', unsafe_allow_html=True)
    cols = st.columns(len(_EXAMPLES))
    for i, ex in enumerate(_EXAMPLES):
        if cols[i].button(
            ex[:34] + "…" if len(ex) > 34 else ex,
            key=f"ex_{i}",
            use_container_width=True,
        ):
            st.session_state["prefill_claim"] = ex
            st.rerun()

    claim = st.text_area(
        "Введите утверждение для проверки:",
        height=90,
        value=st.session_state.get("prefill_claim", ""),
        placeholder="Например: Менделеев изобрёл водку · Земля плоская · В Сахаре живут пингвины…",
        label_visibility="collapsed",
    )

    check_btn = st.button("🔍 Проверить", type="primary", use_container_width=True)

    if not check_btn:
        return
    if not claim.strip():
        st.warning("⚠️ Введите утверждение.")
        return

    st.markdown("---")

    # ── Result slot (top) ──
    result_slot = st.empty()

    # ── Thinking strip ──
    thinking_slot = st.empty()

    # ── Progress bar ──
    progress_slot = st.empty()

    # ── Timeline of stages ──
    timeline_slot = st.empty()

    # ── Expanded details below ──
    st.markdown(" ")
    details_expander = st.expander("🔬 Подробности анализа (по этапам)", expanded=False)

    # Stage metadata
    STAGES = [
        ("parse",         "📋", "Парсинг утверждения"),
        ("decompose",     "✂️", "Декомпозиция"),
        ("keywords",      "🔑", "Извлечение ключевых слов"),
        ("search",        "🔍", "Поиск источников"),
        ("wikidata",      "📊", "Wikidata — структурные факты"),
        ("numbers",       "🔢", "Числовая сверка"),
        ("nli",           "🧬", "NLI (подтверждение vs противоречие)"),
        ("debunk",        "🛑", "Debunk-источники"),
        ("llm_knowledge", "🤖", "Параметрическая память LLM"),
        ("myth",          "🧠", "Myth-probe (базовая Mistral)"),
        ("decide",        "⚖️", "Принятие решения"),
        ("aggregate",     "📦", "Агрегация (если составное)"),
        ("explain",       "💬", "Генерация обоснования"),
    ]
    stage_keys = [s[0] for s in STAGES]
    stage_labels = {s[0]: (s[1], s[2]) for s in STAGES}

    stages_data = {}
    stages_done = set()
    current_stage = {"name": None, "started": time.time()}
    start_time = time.time()

    # Short status snippets per stage (shown in thinking strip)
    _STATUS_LINES = {
        "parse":        "Разбираю утверждение на части",
        "decompose":    "Проверяю, составное ли",
        "keywords":     "Извлекаю именованные сущности",
        "search":       "Ищу источники в DDG и Wikipedia",
        "wikidata":     "Сверяю с Wikidata (SPARQL)",
        "numbers":      "Сравниваю числа с источниками",
        "nli":          "Анализирую подтверждения и противоречия",
        "debunk":       "Ищу опровержения и разоблачения",
        "llm_knowledge":"Консультируюсь с параметрической памятью",
        "myth":         "Проверяю: миф ли это",
        "decide":       "Принимаю финальное решение",
        "aggregate":    "Агрегирую под-вердикты",
        "explain_start":"Генерирую обоснование",
        "explain":      "Формирую ответ",
    }

    def render_thinking(stage_key: str):
        elapsed = int(time.time() - start_time)
        line = _STATUS_LINES.get(stage_key, stage_key)
        thinking_slot.markdown(f"""
        <div class="thinking-strip">
            <div class="spinner"></div>
            <div class="text">{line}<span class="dots"></span></div>
            <div class="counter">{elapsed}s</div>
        </div>
        """, unsafe_allow_html=True)

    def render_progress():
        frac = len(stages_done) / max(len(stage_keys), 1)
        progress_slot.progress(frac)

    def render_timeline():
        items = []
        # past stages as done, current as animating, future as dim
        for key in stage_keys:
            emoji, label = stage_labels[key]
            if key in stages_done:
                # Show short summary from data
                d = stages_data.get(key, {})
                summary = _stage_summary(key, d)
                items.append(f'<div class="stage-item done"><span class="st-icon">✓</span><span>{emoji} {label}</span><span class="st-data">{summary}</span></div>')
            elif key == current_stage["name"]:
                items.append(f'<div class="stage-item current"><span class="st-icon">●</span><span>{emoji} {label}</span><span class="st-data">...</span></div>')
            else:
                items.append(f'<div class="stage-item" style="opacity:0.4"><span class="st-icon">○</span><span>{emoji} {label}</span></div>')
        timeline_slot.markdown("\n".join(items), unsafe_allow_html=True)

    def _stage_summary(key: str, data: dict) -> str:
        """Короткая сводка для timeline после завершения стадии."""
        if key == "parse":
            parts = []
            if data.get("numbers"): parts.append(f"чисел {len(data['numbers'])}")
            if data.get("dates"): parts.append(f"дат {len(data['dates'])}")
            if data.get("is_scam"): parts.append("СКАМ")
            return " · ".join(parts) or data.get("type", "")
        if key == "decompose":
            subs = data.get("sub_claims", [])
            return f"{len(subs)} частей" if data.get("is_composite") else "единичное"
        if key == "keywords":
            return f"{len(data.get('keywords', []))} шт."
        if key == "search":
            return f"{data.get('num_sources', 0)} источн."
        if key == "wikidata":
            sig = data.get("signal", 0)
            return "+1 подтверждение" if sig > 0 else ("−1 противоречие" if sig < 0 else "нет фактов")
        if key == "numbers":
            sig = data.get("signal", 0)
            if sig == 0: return "—"
            return "+1 совпали" if sig > 0 else "−1 не совпали"
        if key == "nli":
            ent = data.get("ent", 0)
            con = data.get("con", 0)
            return f"ent {ent:.2f} · con {con:.2f}"
        if key == "debunk":
            c = data.get("count", 0)
            return f"{c} источн." if c else "—"
        if key == "llm_knowledge":
            sig = data.get("signal", 0)
            return {+1: "правда", -1: "ложь", 0: "не знает"}[sig]
        if key == "myth":
            sig = data.get("signal", 0)
            return {+1: "факт", -1: "МИФ", 0: "не классиф."}[sig]
        if key == "decide":
            return f"{data.get('verdict', '?')} · {data.get('confidence', 0)}%"
        if key == "aggregate":
            return data.get("verdict", "—")
        if key == "explain":
            return "готово"
        return ""

    def on_progress(stage: str, data: dict):
        # Mark previous stage as done (if this is a new stage)
        if current_stage["name"] and current_stage["name"] != stage:
            stages_done.add(current_stage["name"])
        # Special stages like explain_start just update status, не являются самостоятельными
        if stage == "explain_start":
            render_thinking("explain_start")
            return
        if stage == "cache":
            # Hit cache — show directly
            thinking_slot.empty()
            progress_slot.empty()
            timeline_slot.empty()
            return

        stages_data[stage] = data
        current_stage["name"] = stage
        current_stage["started"] = time.time()

        render_thinking(stage)
        render_progress()
        render_timeline()

    # ── Запуск пайплайна ──
    result = pipeline.check(claim.strip(), progress_callback=on_progress)

    # Финальный shutdown прогресса
    thinking_slot.empty()
    progress_slot.empty()
    # Mark all remaining stages as done for the final timeline render
    for key in stage_keys:
        if key in stages_data and key not in stages_done:
            stages_done.add(key)
    current_stage["name"] = None
    render_timeline()

    # ══════════════════════════════════════════════════════════
    #  Result display
    # ══════════════════════════════════════════════════════════
    verdict = result["verdict"].upper().strip()
    score = result["credibility_score"]
    confidence = result["confidence"]
    total_time = result.get("total_time", 0)
    is_composite = result.get("_composite", False) or verdict == "СОСТАВНОЕ"

    if verdict == "СКАМ":
        vc, vlabel, icon, sub = "scam", "СКАМ", "⚠️", "Признаки мошенничества"
    elif is_composite:
        vc, vlabel, icon, sub = "composite", "СОСТАВНОЕ", "🧩", "Одни факты верны, другие — нет"
    elif verdict in ("ЛОЖЬ", "FALSE", "ФЕЙК"):
        vc, vlabel, icon, sub = "false", "ЛОЖЬ", "❌", "Утверждение опровергнуто"
    elif verdict in ("ПРАВДА", "TRUE"):
        vc, vlabel, icon, sub = "true", "ПРАВДА", "✅", "Утверждение подтверждено"
    elif verdict == "НЕ УВЕРЕНА" or verdict == "НЕ ПОДТВЕРЖДЕНО":
        vc, vlabel, icon, sub = "unsure", "НЕ УВЕРЕНА", "❓", "Данных недостаточно"
    else:
        vc, vlabel, icon, sub = "unsure", verdict, "❓", ""

    with result_slot.container():
        # Verdict card
        st.markdown(f"""
        <div class="verdict v-{vc}">
            <span class="icon">{icon}</span>
            <div class="label">{vlabel}</div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

        # Metric pills
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-pill">
                <div class="label">Достоверность</div>
                <div class="value">{score}<span class="suffix">/100</span></div>
            </div>
            <div class="metric-pill">
                <div class="label">Уверенность</div>
                <div class="value">{confidence}<span class="suffix">%</span></div>
            </div>
            <div class="metric-pill">
                <div class="label">Время</div>
                <div class="value">{total_time:.0f}<span class="suffix">сек</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Sub-claims (composite)
        sub_verdicts = result.get("sub_verdicts", [])
        if sub_verdicts:
            st.markdown('<div class="sub-list">', unsafe_allow_html=True)
            for sv in sub_verdicts:
                s = sv.get("status", "?").upper()
                cls = "sub-true" if s == "ПРАВДА" else ("sub-false" if s == "ЛОЖЬ" else "sub-unsure")
                tag_icon = "✓" if s == "ПРАВДА" else ("✗" if s == "ЛОЖЬ" else "?")
                st.markdown(
                    f'<div class="sub-item {cls}"><span class="tag">{tag_icon} {s}</span>{sv["claim"]}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Reasoning — cleaned
        reasoning = clean_reasoning(result.get("reasoning", ""))
        if reasoning:
            st.markdown(f"""
            <div class="reasoning-card">
                <h4>Обоснование</h4>
                {reasoning}
            </div>
            """, unsafe_allow_html=True)

        # Sources (rich cards)
        sources = result.get("sources", [])
        if sources:
            st.markdown('<div class="sources-title">Источники</div>', unsafe_allow_html=True)
            cards = []
            for s in sources[:12]:
                link = s.get("link", "")
                title = s.get("title", "") or s.get("source", "Источник")
                # Домен из url
                domain = ""
                if link:
                    m = re.match(r'https?://([^/]+)/?', link)
                    if m:
                        domain = m.group(1).replace("www.", "")
                title_short = title[:70] + "…" if len(title) > 70 else title
                if link:
                    cards.append(f'<a href="{link}" target="_blank" class="src-card"><span class="domain">{domain or "источник"}</span><span class="title">{title_short}</span></a>')
                else:
                    cards.append(f'<div class="src-card"><span class="domain">{domain or "—"}</span><span class="title">{title_short}</span></div>')
            st.markdown('<div class="src-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  Details expander — по этапам (nerd view)
    # ══════════════════════════════════════════════════════════
    with details_expander:
        # Compact per-stage display
        for key in stage_keys:
            if key not in stages_data:
                continue
            emoji, label = stage_labels[key]
            data = stages_data[key]
            st.markdown(f"**{emoji} {label}** — {_stage_summary(key, data)}")

            if key == "keywords":
                kws = data.get("keywords", [])
                if kws:
                    st.code(", ".join(kws), language=None)
            elif key == "search":
                srcs = data.get("sources", [])
                if srcs:
                    lines = []
                    for s in srcs[:8]:
                        link = s.get("link", "")
                        name = s.get("source") or s.get("title", "?")
                        lines.append(f"- [{name[:50]}]({link})" if link else f"- {name[:50]}")
                    st.markdown("\n".join(lines))
            elif key == "wikidata":
                facts = data.get("facts", [])
                for f in facts[:6]:
                    prop = f.get("property", "?")
                    vals = ", ".join(f.get("values", []))
                    match = f.get("match")
                    mark = "✅" if match is True else ("❌" if match is False else "—")
                    st.markdown(f"{mark} **{prop}**: {vals}")
            elif key == "nli":
                ent = data.get("ent", 0); con = data.get("con", 0); gap = data.get("gap", 0)
                c1, c2, c3 = st.columns(3)
                c1.metric("Entailment", f"{ent:.3f}")
                c2.metric("Contradiction", f"{con:.3f}")
                c3.metric("Gap", f"{gap:+.3f}")
            elif key == "numbers":
                for c in data.get("comparisons", [])[:4]:
                    mark = "✅" if c["match"] else "❌"
                    st.markdown(f"{mark} claim `{c['claim']}` ↔ source `{c['source']}`")
            elif key == "decide":
                signals = []
                if data.get("scam"): signals.append("TIER 0: scam")
                if data.get("wd") != 0: signals.append(f"TIER 1: WD={'+1' if data['wd'] > 0 else '−1'}")
                if data.get("num") != 0: signals.append(f"TIER 2: NUM={'+1' if data['num'] > 0 else '−1'}")
                if data.get("debunk", 0) > 0: signals.append(f"TIER 3: debunk×{data['debunk']}")
                if data.get("nli") != 0: signals.append(f"TIER 4: NLI={'+1' if data['nli'] > 0 else '−1'}")
                if data.get("llm") != 0: signals.append(f"TIER 5: LLM={'+1' if data['llm'] > 0 else '−1'}")
                if signals:
                    st.markdown("Путь: " + " → ".join(signals))
            st.markdown("")  # spacer


if __name__ == "__main__":
    main()
