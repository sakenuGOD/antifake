"""Antifake — Streamlit UI.

Переработанный фронт с акцентом на динамику: большая thinking-панель с
живой JS-анимацией (спиннер + тикер + анимированный emoji по этапу),
анимированная progress-bar с %, пошаговый timeline со строгой
анимацией current-stage, verdict reveal с bounce + typewriter для
обоснования, staggered fade-in для metric pills и source cards.
Обоснование генерируется детерминированно в pipeline._explain (без LLM).
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
from streamlit.components.v1 import html as _html

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
.block-container { padding-top: 1.5rem !important; max-width: 900px !important; }
#MainMenu, footer { visibility: hidden; }

/* ── Hero ── */
.hero { text-align: center; padding: 4px 0 24px 0; }
.hero h1 {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0; letter-spacing: -0.03em;
    animation: shimmerGrad 6s ease-in-out infinite;
    background-size: 200% 200%;
}
@keyframes shimmerGrad {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.hero .subtitle { color: #94a3b8; font-size: 1rem; margin: 0; }

/* ── Input ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    font-size: 1.05rem !important;
    padding: 14px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextArea textarea:focus {
    border-color: rgba(139,92,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
}
.stTextArea label { display: none !important; }

.chip-hint { color: #64748b; font-size: 0.82rem; margin: 8px 0 4px; }
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: transform 0.15s ease, box-shadow 0.2s ease !important;
}
.stButton > button:hover { transform: translateY(-2px); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    border: none !important;
    font-size: 1.1rem !important;
    padding: 14px 22px !important;
    box-shadow: 0 4px 14px rgba(139,92,246,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 22px rgba(139,92,246,0.45) !important;
    transform: translateY(-2px);
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899, #f472b6) !important;
    background-size: 300% 100% !important;
    animation: flowGrad 2.5s linear infinite !important;
    border-radius: 10px !important;
}
@keyframes flowGrad { 0%{background-position:0% 50%} 100%{background-position:300% 50%} }
.stProgress { margin: 8px 0 14px 0 !important; }

/* ── Verdict cards (big) with bounce-in ── */
.verdict {
    text-align: center; padding: 40px 20px 32px;
    border-radius: 22px;
    margin: 24px 0 24px;
    position: relative;
    border: 2px solid;
    animation: bounceIn 0.7s cubic-bezier(0.34, 1.56, 0.64, 1);
    overflow: hidden;
}
@keyframes bounceIn {
    0% { opacity: 0; transform: scale(0.7) translateY(20px); }
    60% { opacity: 1; transform: scale(1.04) translateY(-4px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}
.verdict::after {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(255,255,255,0.06), transparent 60%);
    pointer-events: none;
}
.verdict .icon {
    font-size: 3.8rem; display: block; margin-bottom: 8px; line-height: 1;
    animation: iconPop 0.9s cubic-bezier(0.34, 1.56, 0.64, 1) 0.1s both;
}
@keyframes iconPop {
    0% { transform: scale(0); opacity: 0; }
    60% { transform: scale(1.3); opacity: 1; }
    100% { transform: scale(1); opacity: 1; }
}
.verdict .label {
    font-size: 2.4rem; font-weight: 800; letter-spacing: 0.08em; margin: 6px 0;
    animation: fadeSlide 0.5s ease-out 0.35s both;
}
.verdict .sub {
    color: #94a3b8; font-size: 0.95rem;
    animation: fadeSlide 0.5s ease-out 0.5s both;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

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

/* ── Metric pills (staggered fade-in) ── */
.metric-row { display: flex; gap: 12px; margin: 0 0 20px 0; }
.metric-pill {
    flex: 1; padding: 16px 20px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    transition: transform 0.2s ease, border-color 0.2s ease;
    animation: fadeSlide 0.5s ease-out both;
}
.metric-pill:nth-child(1) { animation-delay: 0.6s; }
.metric-pill:nth-child(2) { animation-delay: 0.75s; }
.metric-pill:nth-child(3) { animation-delay: 0.9s; }
.metric-pill:hover { transform: translateY(-2px); border-color: rgba(139,92,246,0.3); }
.metric-pill .label { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; }
.metric-pill .value { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; margin-top: 6px; font-variant-numeric: tabular-nums; }
.metric-pill .suffix { color: #64748b; font-size: 0.9rem; font-weight: 400; margin-left: 3px; }

/* ── Reasoning (typewriter-style reveal) ── */
.reasoning-card {
    padding: 20px 22px;
    background: rgba(255,255,255,0.02);
    border-left: 3px solid #8b5cf6;
    border-radius: 10px;
    margin: 14px 0 20px;
    color: #cbd5e1; line-height: 1.65;
    animation: fadeSlide 0.6s ease-out 1s both;
}
.reasoning-card h4 {
    margin: 0 0 10px 0; color: #a78bfa;
    font-size: 0.82rem; letter-spacing: 0.12em; text-transform: uppercase;
}
.reasoning-card p { margin: 6px 0; }

/* ── Sub-claims ── */
.sub-list { display: flex; flex-direction: column; gap: 8px; margin: 14px 0 18px; }
.sub-item {
    padding: 10px 14px;
    background: rgba(255,255,255,0.02);
    border-radius: 10px;
    border-left: 3px solid;
    font-size: 0.95rem;
    animation: fadeSlide 0.5s ease-out both;
}
.sub-item:nth-child(1) { animation-delay: 1.1s; }
.sub-item:nth-child(2) { animation-delay: 1.25s; }
.sub-item:nth-child(3) { animation-delay: 1.4s; }
.sub-item:nth-child(4) { animation-delay: 1.55s; }
.sub-true  { border-left-color: #22c55e; }
.sub-false { border-left-color: #ef4444; }
.sub-unsure { border-left-color: #eab308; }
.sub-item .tag { font-weight: 700; margin-right: 10px; }

/* ── Sources (cards with slide-in) ── */
.sources-title {
    color: #94a3b8; font-size: 0.82rem; letter-spacing: 0.1em;
    text-transform: uppercase; margin: 22px 0 8px;
    animation: fadeSlide 0.5s ease-out 1.4s both;
}
.src-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 8px; }
.src-card {
    padding: 11px 13px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    text-decoration: none !important; color: #cbd5e1 !important;
    transition: all 0.18s ease;
    display: block;
    animation: fadeSlide 0.4s ease-out both;
}
.src-card:nth-child(1) { animation-delay: 1.5s; }
.src-card:nth-child(2) { animation-delay: 1.6s; }
.src-card:nth-child(3) { animation-delay: 1.7s; }
.src-card:nth-child(4) { animation-delay: 1.8s; }
.src-card:nth-child(5) { animation-delay: 1.85s; }
.src-card:nth-child(n+6) { animation-delay: 1.9s; }
.src-card:hover {
    background: rgba(139,92,246,0.09);
    border-color: rgba(139,92,246,0.3);
    transform: translateY(-1px);
}
.src-card .domain { color: #60a5fa; font-size: 0.78rem; font-weight: 600; }
.src-card .title {
    display: block; font-size: 0.9rem; margin-top: 3px; line-height: 1.35;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

/* ── Stage timeline ── */
.stage-item {
    padding: 10px 14px; margin: 3px 0;
    border-radius: 10px;
    display: flex; align-items: center; gap: 12px;
    font-size: 0.95rem; color: #cbd5e1;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    transition: all 0.3s ease;
}
.stage-item.done {
    color: #64748b;
    background: rgba(34,197,94,0.04);
    border-color: rgba(34,197,94,0.12);
}
.stage-item.done .st-icon { color: #22c55e; }
.stage-item.current {
    border-color: rgba(139,92,246,0.45);
    background: linear-gradient(90deg, rgba(139,92,246,0.1), rgba(59,130,246,0.04));
    color: #f1f5f9;
    box-shadow: 0 0 20px rgba(139,92,246,0.2);
    transform: translateX(4px);
}
.stage-item.current .st-icon {
    color: #a78bfa;
    animation: pulse 1s ease-in-out infinite;
}
.stage-item.current .emoji { animation: wiggle 1.4s ease-in-out infinite; display: inline-block; }
@keyframes pulse {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.25); }
}
@keyframes wiggle {
    0%, 100% { transform: rotate(0deg); }
    25% { transform: rotate(-10deg) scale(1.08); }
    75% { transform: rotate(10deg) scale(1.08); }
}
.stage-item .st-icon { flex-shrink: 0; width: 18px; text-align: center; font-weight: 700; }
.stage-item .st-data {
    color: #94a3b8; font-size: 0.82rem; margin-left: auto;
    font-variant-numeric: tabular-nums;
}
.stage-item.done .st-data { color: #22c55e; }

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
#  Cleanup (safety net — deterministic explain should already be clean)
# ══════════════════════════════════════════════════════════════════
_LLM_ARTIFACTS = [
    (r'^\s*(?:ВЕРДИКТ|ОБОСНОВАНИЕ|ФАКТЫ|ИСТОЧНИКИ|УВЕРЕННОСТЬ|АНАЛИЗ)\s*:\s*', ''),
    (r'\s*\|\s*(?:ВЕРДИКТ|ОБОСНОВАНИЕ|ФАКТЫ|ИСТОЧНИКИ|УВЕРЕННОСТЬ|АНАЛИЗ)\s*:\s*', ' '),
    (r'\s*(?:ВЕРДИКТ|ОБОСНОВАНИЕ|ФАКТЫ|ИСТОЧНИКИ|УВЕРЕННОСТЬ|АНАЛИЗ)\s*:\s*', ' '),
    (r'\[Источник\]\s*сообщает:\s*', ''),
    (r'«\[Источник\][^»]*»', ''),
    (r'^\s*\d+%?\s*(?:НЕ\s+ПОДТВЕРЖДЕНО|ПОДТВЕРЖДЕНО)[^\.]*\.?\s*', ''),
    (r'\s{2,}', ' '),
]


def clean_reasoning(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    for pattern, repl in _LLM_ARTIFACTS:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE | re.MULTILINE)
    if cleaned.count('ОБОСНОВАНИЕ:') > 1 or cleaned.count('обоснование:') > 1:
        parts = re.split(r'(?:ОБОСНОВАНИЕ|обоснование)\s*:\s*', cleaned)
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            cleaned = max(parts, key=len)
    cleaned = cleaned.strip(" |.,;—-")
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

    # ── Examples (rendered BEFORE text_area so prefill works safely) ──
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

    # ── Slots ──
    result_slot = st.empty()
    thinking_slot = st.empty()
    progress_slot = st.empty()
    timeline_slot = st.empty()

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
        ("nli",           "🧬", "NLI анализ (за/против)"),
        ("debunk",        "🛑", "Поиск опровержений"),
        ("llm_knowledge", "🤖", "Параметрическая память LLM"),
        ("myth",          "🧠", "Myth-probe (базовая Mistral)"),
        ("decide",        "⚖️", "Принятие решения"),
        ("aggregate",     "📦", "Агрегация (если составное)"),
        ("explain",       "💬", "Генерация обоснования"),
    ]
    stage_keys = [s[0] for s in STAGES]
    stage_labels = {s[0]: (s[1], s[2]) for s in STAGES}

    # Thinking-strip human text per stage
    _STATUS_LINES = {
        "parse":         "Разбираю структуру утверждения",
        "decompose":     "Проверяю на составные факты",
        "keywords":      "Извлекаю именованные сущности",
        "search":        "Ищу источники в интернете",
        "wikidata":      "Сверяю факты через Wikidata",
        "numbers":       "Сравниваю числа в утверждении и источниках",
        "nli":           "Анализирую подтверждения и противоречия",
        "debunk":        "Ищу опровержения и разоблачения",
        "llm_knowledge": "Консультируюсь с параметрической памятью",
        "myth":          "Проверяю: миф ли это",
        "decide":        "Принимаю финальное решение",
        "aggregate":     "Агрегирую под-вердикты",
        "explain":       "Формирую обоснование",
        "explain_start": "Формирую обоснование",
    }

    stages_data = {}
    stages_done = set()
    stage_started_at = {}
    current_stage = {"name": None}
    start_time = time.time()

    def _stage_summary(key: str, data: dict) -> str:
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
            return "✓ подтверждение" if sig > 0 else ("✗ противоречие" if sig < 0 else "нет фактов")
        if key == "numbers":
            sig = data.get("signal", 0)
            if sig == 0: return "—"
            return "✓ совпали" if sig > 0 else "✗ не совпали"
        if key == "nli":
            ent = data.get("ent", 0); con = data.get("con", 0)
            return f"ent {ent:.2f} · con {con:.2f}"
        if key == "debunk":
            c = data.get("count", 0)
            return f"{c} источн." if c else "—"
        if key == "llm_knowledge":
            return {+1: "правда", -1: "ложь", 0: "не знает"}[data.get("signal", 0)]
        if key == "myth":
            return {+1: "факт", -1: "МИФ", 0: "не классиф."}[data.get("signal", 0)]
        if key == "decide":
            return f"{data.get('verdict', '?')} · {data.get('confidence', 0)}%"
        if key == "aggregate":
            return data.get("verdict", "—")
        if key == "explain":
            return "готово"
        return ""

    def render_thinking(stage_key: str):
        emoji, short = stage_labels.get(stage_key, ("…", stage_key))
        line = _STATUS_LINES.get(stage_key, stage_key)
        elapsed_start = int(time.time() - start_time)
        step_num = len(stages_done) + 1
        step_total = len(stage_keys)
        with thinking_slot:
            _html(f"""
            <style>
                body {{ margin: 0; padding: 0; background: transparent;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        color: #e2e8f0; }}
                .ts {{
                    display: flex; align-items: center; gap: 18px;
                    padding: 20px 24px;
                    background: linear-gradient(110deg,
                        rgba(59,130,246,0.10) 0%,
                        rgba(139,92,246,0.14) 50%,
                        rgba(236,72,153,0.10) 100%);
                    background-size: 200% 200%;
                    animation: bgShift 4s ease infinite;
                    border: 1px solid rgba(139,92,246,0.28);
                    border-radius: 14px;
                    box-shadow: 0 4px 24px rgba(139,92,246,0.14);
                    position: relative;
                    overflow: hidden;
                }}
                @keyframes bgShift {{
                    0%,100% {{ background-position: 0% 50%; }}
                    50% {{ background-position: 100% 50%; }}
                }}
                .ts::before {{
                    content: ''; position: absolute; inset: 0;
                    background: radial-gradient(circle at var(--mx, 50%) var(--my, 50%),
                        rgba(139,92,246,0.18), transparent 45%);
                    pointer-events: none;
                    opacity: 0.5;
                }}
                .emoji-big {{
                    font-size: 2.2rem; flex-shrink: 0;
                    animation: bounceEmoji 1.6s ease-in-out infinite;
                    filter: drop-shadow(0 0 8px rgba(139,92,246,0.4));
                }}
                @keyframes bounceEmoji {{
                    0%,100% {{ transform: translateY(0) scale(1) rotate(0deg); }}
                    25% {{ transform: translateY(-4px) scale(1.1) rotate(-8deg); }}
                    50% {{ transform: translateY(-2px) scale(1.15) rotate(0deg); }}
                    75% {{ transform: translateY(-4px) scale(1.1) rotate(8deg); }}
                }}
                .sp {{
                    width: 22px; height: 22px; flex-shrink: 0;
                    border: 2.5px solid rgba(139,92,246,0.22);
                    border-top-color: #a78bfa;
                    border-right-color: #60a5fa;
                    border-radius: 50%;
                    animation: sp 0.8s linear infinite;
                }}
                @keyframes sp {{ to {{ transform: rotate(360deg); }} }}
                .text-wrap {{ flex: 1; display: flex; flex-direction: column; gap: 2px; }}
                .tx {{ color: #f1f5f9; font-weight: 600; font-size: 1.08rem; }}
                .dt {{ display: inline-block; min-width: 20px; color: #a78bfa; }}
                .sub-tx {{ color: #94a3b8; font-size: 0.78rem; }}
                .cr {{
                    display: flex; flex-direction: column; align-items: flex-end;
                    gap: 2px; flex-shrink: 0;
                }}
                .cr-timer {{
                    color: #e2e8f0; font-variant-numeric: tabular-nums;
                    font-size: 1.25rem; font-weight: 700;
                    font-family: 'SF Mono', Menlo, monospace;
                }}
                .cr-step {{ color: #94a3b8; font-size: 0.76rem; letter-spacing: 0.08em; }}
            </style>
            <div class="ts" id="strip">
                <div class="emoji-big">{emoji}</div>
                <div class="sp"></div>
                <div class="text-wrap">
                    <div class="tx">{line}<span class="dt" id="dots">···</span></div>
                    <div class="sub-tx">{short}</div>
                </div>
                <div class="cr">
                    <div class="cr-timer"><span id="timer">{elapsed_start}</span>s</div>
                    <div class="cr-step">ЭТАП {step_num} из {step_total}</div>
                </div>
            </div>
            <script>
                (function() {{
                    var elapsed = {elapsed_start};
                    var t = document.getElementById('timer');
                    var d = document.getElementById('dots');
                    var strip = document.getElementById('strip');
                    var seq = ['', '·', '··', '···'];
                    var i = 0;
                    setInterval(function() {{
                        elapsed++; if (t) t.textContent = elapsed;
                    }}, 1000);
                    setInterval(function() {{
                        i = (i + 1) % seq.length;
                        if (d) d.textContent = seq[i];
                    }}, 320);
                    if (strip) {{
                        strip.addEventListener('mousemove', function(e) {{
                            var r = strip.getBoundingClientRect();
                            strip.style.setProperty('--mx', ((e.clientX - r.left)/r.width*100)+'%');
                            strip.style.setProperty('--my', ((e.clientY - r.top)/r.height*100)+'%');
                        }});
                    }}
                }})();
            </script>
            """, height=92)

    def render_progress():
        frac = len(stages_done) / max(len(stage_keys), 1)
        pct = int(frac * 100)
        progress_slot.progress(frac, text=f"Анализ: {pct}%")

    def render_timeline():
        items = []
        for key in stage_keys:
            emoji, label = stage_labels[key]
            if key in stages_done:
                d = stages_data.get(key, {})
                summary = _stage_summary(key, d)
                duration = stage_started_at.get(key + "_dur", "")
                right = f"{summary}" + (f" · {duration}s" if duration else "")
                items.append(
                    f'<div class="stage-item done">'
                    f'<span class="st-icon">✓</span>'
                    f'<span><span class="emoji">{emoji}</span> {label}</span>'
                    f'<span class="st-data">{right}</span>'
                    f'</div>'
                )
            elif key == current_stage["name"]:
                items.append(
                    f'<div class="stage-item current">'
                    f'<span class="st-icon">●</span>'
                    f'<span><span class="emoji">{emoji}</span> {label}</span>'
                    f'<span class="st-data">работаю…</span>'
                    f'</div>'
                )
            else:
                items.append(
                    f'<div class="stage-item" style="opacity:0.35">'
                    f'<span class="st-icon">○</span>'
                    f'<span><span class="emoji">{emoji}</span> {label}</span>'
                    f'</div>'
                )
        timeline_slot.markdown("\n".join(items), unsafe_allow_html=True)

    def on_progress(stage: str, data: dict):
        prev = current_stage["name"]
        if prev and prev != stage and prev != "explain_start":
            stages_done.add(prev)
            # record duration
            start_of_prev = stage_started_at.get(prev, time.time())
            stage_started_at[prev + "_dur"] = int(time.time() - start_of_prev)

        if stage == "explain_start":
            render_thinking("explain_start")
            return
        if stage == "cache":
            thinking_slot.empty()
            progress_slot.empty()
            timeline_slot.empty()
            return

        stages_data[stage] = data
        current_stage["name"] = stage
        stage_started_at[stage] = time.time()

        render_thinking(stage)
        render_progress()
        render_timeline()

    # ── Run pipeline ──
    result = pipeline.check(claim.strip(), progress_callback=on_progress)

    # Finalise UI
    if current_stage["name"]:
        stages_done.add(current_stage["name"])
        start_of_last = stage_started_at.get(current_stage["name"], time.time())
        stage_started_at[current_stage["name"] + "_dur"] = int(time.time() - start_of_last)
    for key in stage_keys:
        if key in stages_data and key not in stages_done:
            stages_done.add(key)
    current_stage["name"] = None
    thinking_slot.empty()
    progress_slot.empty()
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
    elif verdict in ("НЕ УВЕРЕНА", "НЕ ПОДТВЕРЖДЕНО"):
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

        # Sub-claims
        sub_verdicts = result.get("sub_verdicts", [])
        if sub_verdicts:
            sub_items = []
            for sv in sub_verdicts:
                s = sv.get("status", "?").upper()
                cls = "sub-true" if s == "ПРАВДА" else ("sub-false" if s == "ЛОЖЬ" else "sub-unsure")
                tag_icon = "✓" if s == "ПРАВДА" else ("✗" if s == "ЛОЖЬ" else "?")
                sub_items.append(
                    f'<div class="sub-item {cls}"><span class="tag">{tag_icon} {s}</span>{sv["claim"]}</div>'
                )
            st.markdown('<div class="sub-list">' + "".join(sub_items) + '</div>', unsafe_allow_html=True)

        # Reasoning
        reasoning = clean_reasoning(result.get("reasoning", ""))
        if reasoning:
            reasoning_html = reasoning.replace("\n", "<br>")
            st.markdown(f"""
            <div class="reasoning-card">
                <h4>Обоснование</h4>
                {reasoning_html}
            </div>
            """, unsafe_allow_html=True)

        # Sources
        sources = result.get("sources", [])
        if sources:
            st.markdown('<div class="sources-title">Источники</div>', unsafe_allow_html=True)
            cards = []
            for s in sources[:12]:
                link = s.get("link", "")
                title = s.get("title", "") or s.get("source", "Источник")
                domain = ""
                if link:
                    m = re.match(r'https?://([^/]+)/?', link)
                    if m: domain = m.group(1).replace("www.", "")
                title_short = title[:70] + "…" if len(title) > 70 else title
                if link:
                    cards.append(f'<a href="{link}" target="_blank" class="src-card"><span class="domain">{domain or "источник"}</span><span class="title">{title_short}</span></a>')
                else:
                    cards.append(f'<div class="src-card"><span class="domain">{domain or "—"}</span><span class="title">{title_short}</span></div>')
            st.markdown('<div class="src-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)

    # ── Details expander ──
    with details_expander:
        for key in stage_keys:
            if key not in stages_data:
                continue
            emoji, label = stage_labels[key]
            data = stages_data[key]
            duration = stage_started_at.get(key + "_dur", "")
            dur_suffix = f"  ·  {duration}с" if duration != "" else ""
            st.markdown(f"**{emoji} {label}** — {_stage_summary(key, data)}{dur_suffix}")

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
            st.markdown("")


if __name__ == "__main__":
    main()
