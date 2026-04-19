"""
Система определения достоверности новостей на основе нейросети.
Веб-интерфейс с прозрачным отображением работы пайплайна.
"""

import os
import sys
import time

# Windows console is cp1251 by default; Streamlit-invoked prints with
# Unicode glyphs (✓, arrows, emoji) crash under UnicodeEncodeError.
# Force UTF-8 stdout/stderr early, before any module does its first print.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import streamlit as st

st.set_page_config(
    page_title="Проверка достоверности новостей",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Минимальный CSS ──
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; max-width: 1000px; }

.verdict-big {
    text-align: center; padding: 20px; margin: 16px 0;
    border-radius: 12px; border: 2px solid;
}
.verdict-big .label { font-size: 1.5rem; font-weight: 700; letter-spacing: 2px; }
.vb-true  { border-color: #22c55e; background: rgba(34,197,94,0.06); }
.vb-true .label { color: #22c55e; }
.vb-false { border-color: #ef4444; background: rgba(239,68,68,0.06); }
.vb-false .label { color: #ef4444; }
.vb-scam { border-color: #f97316; background: rgba(249,115,22,0.08); }
.vb-scam .label { color: #f97316; }
.vb-unsure { border-color: #eab308; background: rgba(234,179,8,0.06); }
.vb-unsure .label { color: #eab308; }
.vb-composite { border-color: #a855f7; background: rgba(168,85,247,0.06); }
.vb-composite .label { color: #a855f7; }

.signal-pos { color: #22c55e; font-weight: 600; }
.signal-neg { color: #ef4444; font-weight: 600; }
.signal-zero { color: #6b7280; }

.src-chip {
    display: inline-block; padding: 4px 12px; margin: 3px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; font-size: 12px; color: #93c5fd;
    text-decoration: none;
}
.src-chip:hover { background: rgba(255,255,255,0.08); color: #bfdbfe; }
</style>
""", unsafe_allow_html=True)


# ── Загрузка пайплайна ──
@st.cache_resource
def load_pipeline():
    from pipeline import FactCheckPipeline
    from config import SearchConfig
    from model import find_best_adapter

    search_config = SearchConfig(api_key=os.environ.get("SERPAPI_API_KEY", ""))
    adapter_path = find_best_adapter()
    return FactCheckPipeline(adapter_path=adapter_path, search_config=search_config)


def _signal_str(val: int) -> str:
    if val > 0:
        return '<span class="signal-pos">+1 подтверждает</span>'
    elif val < 0:
        return '<span class="signal-neg">−1 противоречит</span>'
    return '<span class="signal-zero">0 нет данных</span>'


# ── Sidebar ──
with st.sidebar:
    st.markdown("### Архитектура пайплайна")
    st.markdown("""
1. **Парсинг** — тип, числа, даты, скам
2. **Декомпозиция** — разбивка составных
3. **Ключевые слова** — NER + pymorphy2
4. **Поиск** — Wikipedia + DuckDuckGo
5. **Wikidata** — SPARQL (20+ свойств)
6. **NLI** — RuBERT + mDeBERTa
7. **Числовой анализ** — сравнение чисел
8. **Решение** — Gap-based приоритеты
9. **Агрегация** — для составных
10. **Объяснение** — LLM генерация
    """)
    st.markdown("---")
    st.markdown("##### Технологии")
    st.markdown(
        "Mistral 7B · QLoRA 4-bit · GRPO · "
        "RuBERT NLI · mDeBERTa · Cross-Encoder · "
        "Wikidata · Natasha NER · pymorphy2 · "
        "LangChain · DuckDuckGo · Wikipedia"
    )
    st.markdown("---")
    st.caption("GPU: RTX 5070 12GB · NLI: CPU")
    st.caption("Конференция «Инженеры будущего» 2026")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    st.markdown("## Система определения достоверности новостей")
    st.caption("Нейросетевой анализ с верификацией через открытые источники")

    with st.spinner("Загрузка моделей..."):
        pipeline = load_pipeline()

    if pipeline is None:
        st.error("Не удалось загрузить пайплайн.")
        st.stop()

    # ── Ввод ──
    claim = st.text_area(
        "Введите утверждение для проверки:",
        height=80,
        placeholder="Например: Менделеев изобрёл водку",
    )

    check_btn = st.button("Проверить", type="primary", use_container_width=True)

    if not (check_btn and claim.strip()):
        if check_btn:
            st.warning("Введите текст.")
        return

    st.markdown("---")

    # ── Placeholder для результата СВЕРХУ ──
    result_slot = st.empty()

    # ── Ход анализа НИЖЕ ──
    st.markdown("### Ход анализа")
    log = st.container()
    stages_data = {}

    def on_progress(stage: str, data: dict):
        stages_data[stage] = data

        with log:
            if stage == "cache":
                st.info(f"Результат из кэша: **{data.get('verdict')}** ({data.get('credibility_score')}%)")

            elif stage == "parse":
                with st.expander("📋 Этап 1: Парсинг", expanded=True):
                    ctype = data.get("type", "?")
                    nums = data.get("numbers", [])
                    dates = data.get("dates", [])
                    scam = data.get("is_scam", False)
                    st.markdown(f"**Тип:** {ctype}")
                    if nums:
                        st.markdown(f"**Числа:** {', '.join(str(n.get('original', n.get('value', ''))) for n in nums)}")
                    if dates:
                        st.markdown(f"**Даты:** {', '.join(str(d) for d in dates)}")
                    if scam:
                        patterns = data.get("scam_patterns", [])
                        st.error(f"Скам-паттерны: {', '.join(patterns) if patterns else 'да'}")
                    elif not nums and not dates:
                        st.markdown("Числа/даты: нет · Скам: нет")

            elif stage == "decompose":
                subs = data.get("sub_claims", [])
                comp = data.get("is_composite", False)
                title = f"✂️ Этап 2: Декомпозиция — {'составное (' + str(len(subs)) + ' частей)' if comp else 'единичное'}"
                with st.expander(title, expanded=comp):
                    if comp:
                        for i, sc in enumerate(subs, 1):
                            st.markdown(f"{i}. {sc}")
                    else:
                        st.markdown("Декомпозиция не требуется.")

            elif stage == "keywords":
                kws = data.get("keywords", [])
                with st.expander(f"🔑 Этап 3: Ключевые слова — {len(kws)} шт.", expanded=True):
                    st.code(", ".join(kws), language=None)

            elif stage == "search":
                n = data.get("num_sources", 0)
                srcs = data.get("sources", [])
                with st.expander(f"🔍 Этап 4: Поиск — найдено {n} источников", expanded=True):
                    if srcs:
                        chips = ""
                        for s in srcs:
                            link = s.get("link", "")
                            name = s.get("source", "") or s.get("title", "?")
                            name = name[:40] + "…" if len(name) > 40 else name
                            if link:
                                chips += f'<a href="{link}" target="_blank" class="src-chip">{name}</a> '
                            else:
                                chips += f'<span class="src-chip">{name}</span> '
                        st.markdown(chips, unsafe_allow_html=True)
                    else:
                        st.markdown("Источники не найдены.")

            elif stage == "wikidata":
                sig = data.get("signal", 0)
                facts = data.get("facts", [])
                found = data.get("found", False)
                sc = data.get("claim", "")
                label = f"📊 Wikidata — сигнал: {'+1' if sig > 0 else '−1' if sig < 0 else '0'}"
                if sc and stages_data.get("decompose", {}).get("is_composite"):
                    label += f" [{sc[:40]}]"
                with st.expander(label, expanded=(sig != 0)):
                    if found and facts:
                        for f in facts:
                            prop = f.get("property", "?")
                            vals = ", ".join(f.get("values", []))
                            match = f.get("match")
                            if match is True:
                                st.markdown(f"✅ **{prop}**: {vals}")
                            elif match is False:
                                st.markdown(f"❌ **{prop}**: {vals}")
                            else:
                                st.markdown(f"— **{prop}**: {vals}")
                    else:
                        st.markdown("Факты не найдены в Wikidata.")
                    st.markdown(f"Сигнал: {_signal_str(sig)}", unsafe_allow_html=True)

            elif stage == "nli":
                sig = data.get("signal", 0)
                ent = data.get("ent", 0)
                con = data.get("con", 0)
                gap = data.get("gap", 0)
                sc = data.get("claim", "")
                label = f"🧬 NLI — ent={ent:.2f} con={con:.2f} gap={gap:+.2f}"
                if sc and stages_data.get("decompose", {}).get("is_composite"):
                    label += f" [{sc[:40]}]"
                with st.expander(label, expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Entailment", f"{ent:.3f}")
                    c2.metric("Contradiction", f"{con:.3f}")
                    c3.metric("Gap (ent−con)", f"{gap:+.3f}")
                    if abs(gap) < 0.12:
                        st.markdown("Зона: **неоднозначная** (|gap| < 0.12)")
                    elif abs(gap) < 0.30:
                        st.markdown(f"Зона: **умеренная** ({'подтверждение' if gap > 0 else 'противоречие'})")
                    else:
                        st.markdown(f"Зона: **сильная** ({'подтверждение' if gap > 0 else 'противоречие'})")
                    st.markdown(f"Сигнал: {_signal_str(sig)}", unsafe_allow_html=True)

            elif stage == "numbers":
                sig = data.get("signal", 0)
                claim_nums = data.get("claim_numbers", [])
                comparisons = data.get("comparisons", [])
                if claim_nums or sig != 0:
                    with st.expander(f"🔢 Числа — сигнал: {'+1' if sig > 0 else '−1' if sig < 0 else '0'}", expanded=(sig != 0)):
                        if claim_nums:
                            st.markdown(f"Числа в утверждении: **{', '.join(claim_nums)}**")
                        if comparisons:
                            for c in comparisons:
                                mark = "✅" if c["match"] else "❌"
                                st.markdown(f"{mark} утверждение: {c['claim']} → источник: {c['source']}")
                        st.markdown(f"Сигнал: {_signal_str(sig)}", unsafe_allow_html=True)

            elif stage == "debunk":
                count = data.get("count", 0)
                if count > 0:
                    with st.expander(f"🔎 Debunk-источники: {count}", expanded=True):
                        st.markdown(f"Найдено **{count}** источников, разоблачающих утверждение.")

            elif stage == "llm_knowledge":
                sig = data.get("signal", 0)
                if sig != 0:
                    with st.expander(f"🤖 LLM-знания — сигнал: {'+1' if sig > 0 else '−1'}", expanded=True):
                        if sig > 0:
                            st.markdown("Модель считает утверждение **правдивым**.")
                        else:
                            st.markdown("Модель считает утверждение **ложным**.")

            elif stage == "decide":
                v = data.get("verdict", "?")
                conf = data.get("confidence", 0)
                sc = data.get("claim", "")
                label = f"⚖️ Решение: {v} ({conf}%)"
                if sc and stages_data.get("decompose", {}).get("is_composite"):
                    label += f" [{sc[:40]}]"
                with st.expander(label, expanded=True):
                    signals = []
                    if data.get("scam"):
                        signals.append("TIER 0: скам")
                    if data.get("wd") != 0:
                        signals.append(f"TIER 1: Wikidata={'+1' if data['wd'] > 0 else '−1'}")
                    if data.get("num") != 0:
                        signals.append(f"TIER 2: числа={'+1' if data['num'] > 0 else '−1'}")
                    if data.get("debunk", 0) > 0:
                        signals.append(f"TIER 3: debunk×{data['debunk']}")
                    if data.get("nli") != 0:
                        signals.append(f"TIER 4: NLI={'+1' if data['nli'] > 0 else '−1'}")
                    if data.get("llm") != 0:
                        signals.append(f"TIER 5: LLM={'+1' if data['llm'] > 0 else '−1'}")
                    if not signals:
                        signals.append("TIER 6: нет сигналов")
                    st.markdown("**Путь решения:** " + " → ".join(signals))
                    st.markdown(f"**Вердикт: {v}** (уверенность {conf}%)")

            elif stage == "aggregate":
                if data.get("is_composite"):
                    v = data.get("verdict", "?")
                    subs = data.get("sub_verdicts", [])
                    with st.expander(f"📦 Агрегация: {v}", expanded=True):
                        for sr in subs:
                            icon = "✅" if sr["verdict"] == "ПРАВДА" else "❌" if sr["verdict"] == "ЛОЖЬ" else "⚠️"
                            st.markdown(f"{icon} **{sr['verdict']}** — {sr['claim']}")
                        st.markdown(f"**Итого: {v}**")

            elif stage == "explain_start":
                st.markdown("💬 *Генерация обоснования...*")

    # ── Запуск пайплайна ──
    result = pipeline.check(claim.strip(), progress_callback=on_progress)

    # ══════════════════════════════════════════════════════════
    #  РЕЗУЛЬТАТ — вставляется в placeholder СВЕРХУ
    # ══════════════════════════════════════════════════════════
    verdict = result["verdict"].upper().strip()
    score = result["credibility_score"]
    confidence = result["confidence"]
    is_composite = result.get("_composite", False) or verdict == "СОСТАВНОЕ"

    if verdict == "СКАМ":
        vc, vlabel = "scam", "⚠️ СКАМ / МОШЕННИЧЕСТВО"
    elif is_composite:
        vc, vlabel = "composite", "СОСТАВНОЕ"
    elif verdict in ("ЛОЖЬ", "FALSE", "ФЕЙК"):
        vc, vlabel = "false", "ЛОЖЬ"
    elif verdict in ("ПРАВДА", "TRUE"):
        vc, vlabel = "true", "ПРАВДА"
    else:
        vc, vlabel = "unsure", "НЕ УВЕРЕНА"

    with result_slot.container():
        # Вердикт
        st.markdown(f"""
        <div class="verdict-big vb-{vc}">
            <div class="label">{vlabel}</div>
        </div>
        """, unsafe_allow_html=True)

        # Метрики
        c1, c2, c3 = st.columns(3)
        c1.metric("Достоверность", f"{score}%")
        c2.metric("Уверенность ИИ", f"{confidence}%")
        c3.metric("Время анализа", f"{result['total_time']:.1f} сек")

        # Подвердикты (для составных)
        sub_verdicts = result.get("sub_verdicts", [])
        if sub_verdicts:
            for sv in sub_verdicts:
                s = sv.get("status", "?").upper()
                icon = "✅" if s == "ПРАВДА" else "❌" if s == "ЛОЖЬ" else "⚠️"
                st.markdown(f"{icon} **{s}** — {sv['claim']}")

        # Обоснование
        reasoning = result.get("reasoning", "").strip()
        if reasoning:
            st.markdown(f"**Обоснование:** {reasoning}")

        # Источники
        sources = result.get("sources", [])
        if sources:
            chips = ""
            for s in sources[:10]:
                link = s.get("link", "")
                name = s.get("source", "") or s.get("title", "?")
                name = name[:45] + "…" if len(name) > 45 else name
                if link:
                    chips += f'<a href="{link}" target="_blank" class="src-chip">🌐 {name}</a> '
                else:
                    chips += f'<span class="src-chip">📄 {name}</span> '
            st.markdown(chips, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
