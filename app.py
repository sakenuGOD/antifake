"""
Streamlit веб-интерфейс для Fact-Checker.

Дизайн соответствует Рисунку 4 документации (Раздел 2.5 / 3.4).
"""

import os
import streamlit as st

st.set_page_config(
    page_title="Fact-Checker — Проверка достоверности",
    page_icon="🔍",
    layout="wide",
)

# --- Тёмная тема (CSS) ---
st.markdown("""
<style>
    .main-header { font-size: 28px; font-weight: bold; margin-bottom: 5px; }
    .sub-header { color: #888; font-size: 14px; margin-bottom: 20px; }
    .status-bar {
        background: #1a3a1a; border: 1px solid #2d5a2d; border-radius: 8px;
        padding: 12px 16px; margin: 15px 0; color: #4caf50; font-size: 14px;
    }
    .metric-container {
        display: flex; justify-content: space-between; align-items: center;
        margin: 15px 0; gap: 20px;
    }
    .metric-box { text-align: center; flex: 1; }
    .metric-label { color: #888; font-size: 12px; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: bold; }
    .metric-delta { font-size: 12px; }
    .verdict-badge {
        display: inline-block; padding: 6px 18px; border-radius: 6px;
        font-size: 14px; font-weight: bold; margin: 5px 0;
    }
    .verdict-false { background: #dc3545; color: #fff; }
    .verdict-true { background: #28a745; color: #fff; }
    .verdict-partial { background: #ffc107; color: #333; }
    .reasoning-box {
        background: #2a1a1a; border: 1px solid #5a2d2d; border-radius: 8px;
        padding: 16px; margin: 15px 0; line-height: 1.6;
    }
    .reasoning-title {
        color: #ff6b6b; font-weight: bold; font-size: 14px;
        margin-bottom: 10px; text-transform: uppercase;
    }
    .critique-box {
        background: #1a2a1a; border: 1px solid #2d5a2d; border-radius: 8px;
        padding: 16px; margin: 15px 0; line-height: 1.6;
    }
    .critique-title {
        color: #4caf50; font-weight: bold; font-size: 14px;
        margin-bottom: 10px; text-transform: uppercase;
    }
    .source-item {
        padding: 8px 0; border-bottom: 1px solid #333;
    }
    .source-link { color: #4caf50; text-decoration: none; }
    .tech-params { color: #888; font-size: 13px; line-height: 1.8; }

    /* Карточки фактов */
    .fact-card {
        border-radius: 8px; padding: 14px 16px; margin: 10px 0;
        border-left: 5px solid; background: #1e1e2e;
    }
    .fact-card-true  { border-left-color: #28a745; background: #0d2617; }
    .fact-card-false { border-left-color: #dc3545; background: #2a0d0d; }
    .fact-card-unknown { border-left-color: #ffc107; background: #2a2100; }
    .fact-card-header {
        display: flex; justify-content: space-between;
        align-items: flex-start; gap: 12px;
    }
    .fact-claim-text { font-size: 15px; line-height: 1.5; flex: 1; }
    .fact-status-badge {
        padding: 4px 12px; border-radius: 4px; font-size: 12px;
        font-weight: bold; white-space: nowrap;
    }
    .fact-citation {
        color: #bbb; font-size: 13px; margin-top: 10px;
        border-top: 1px solid #333; padding-top: 8px; font-style: italic;
    }
    .fact-source { color: #777; font-size: 12px; margin-top: 3px; }

    /* Чипы источников */
    .sources-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
    .source-chip {
        display: inline-flex; align-items: center; gap: 5px;
        background: #1e2a1e; border: 1px solid #2d5a2d;
        border-radius: 20px; padding: 6px 14px;
        font-size: 13px; color: #4caf50; text-decoration: none;
        transition: background 0.15s;
    }
    .source-chip:hover { background: #263d26; color: #6fcf70; }
    .source-chip-plain {
        display: inline-flex; align-items: center; gap: 5px;
        background: #222; border: 1px solid #444;
        border-radius: 20px; padding: 6px 14px;
        font-size: 13px; color: #888;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Загрузка пайплайна один раз при старте приложения."""
    from pipeline import FactCheckPipeline
    from config import SearchConfig
    from model import find_best_adapter

    api_key = os.environ.get("SERPAPI_API_KEY", "")
    search_config = SearchConfig(api_key=api_key)

    # Автовыбор: GRPO > SFT > base
    adapter_path = find_best_adapter()

    return FactCheckPipeline(
        adapter_path=adapter_path,
        search_config=search_config,
    )


def main():
    st.markdown('<div class="main-header">Система определения достоверности новостей</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Mistral 7B + RAG-Pipeline | LangChain | SerpAPI + DuckDuckGo</div>', unsafe_allow_html=True)

    # Проверка API ключа (не обязательно — есть DuckDuckGo fallback)
    api_key = os.environ.get("SERPAPI_API_KEY", "")
    if not api_key:
        st.warning(
            "SERPAPI_API_KEY не установлен — поиск через DuckDuckGo. "
            "Для лучших результатов установите: `export SERPAPI_API_KEY='ваш_ключ'`"
        )

    # Загрузка пайплайна
    with st.spinner("Загрузка модели Mistral 7B + семантический ранкер..."):
        pipeline = load_pipeline()

    if pipeline is None:
        st.error("Не удалось загрузить пайплайн.")
        st.stop()

    # --- Область ввода (Раздел 2.5, п.1) ---
    st.markdown("#### Исходный текст для анализа")
    claim = st.text_area(
        label="Исходный текст",
        height=80,
        placeholder="Например: ЦБ РФ экстренно поднял ключевую ставку до 25% сегодня утром",
        label_visibility="collapsed",
    )

    check_btn = st.button("Проверить достоверность", type="primary", use_container_width=True)

    # --- Результаты ---
    if check_btn and claim.strip():

        # Панель индикации статуса (Раздел 2.5, п.2)
        with st.status("Выполняется анализ...", expanded=True) as status:
            st.write("Этап 1: Извлечение ключевых слов (Агент-Исследователь)...")
            st.write("Этап 2: Поиск новостей через SerpAPI (Модуль извлечения)...")
            st.write("Этап 3: Ранжирование источников (Cosine Similarity)...")
            st.write("Этап 4: Оценка достоверности (Mistral 7B QLoRA)...")

            result = pipeline.check(claim.strip())

            status.update(
                label="Анализ завершен на основе актуальных данных поисковой выдачи",
                state="complete",
            )

        # --- Область результатов (Раздел 2.5, п.3) ---
        score = result["credibility_score"]
        confidence = result["confidence"]
        verdict = result["verdict"]
        latency = result["total_time"]

        # Определение цвета и статуса
        v_upper = verdict.upper().strip()
        if v_upper in ("ЛОЖЬ", "FALSE", "ФЕЙК"):
            score_color = "#dc3545"
            delta_prefix = "↓"
            verdict_class = "verdict-false"
            verdict_label = "ЛОЖЬ / ФЕЙК"
        elif "МАНИПУЛЯЦИЯ" in v_upper or "ПОЛУПРАВДА" in v_upper:
            score_color = "#fd7e14"
            delta_prefix = "⚠"
            verdict_class = "verdict-partial"
            verdict_label = "МАНИПУЛЯЦИЯ / ПОЛУПРАВДА"
        elif v_upper in ("ПРАВДА", "TRUE"):
            score_color = "#28a745"
            delta_prefix = "↑"
            verdict_class = "verdict-true"
            verdict_label = "ПРАВДА / TRUE"
        else:
            score_color = "#ffc107"
            delta_prefix = "~"
            verdict_class = "verdict-partial"
            verdict_label = "НЕ ПОДТВЕРЖДЕНО / UNVERIFIED"

        # Метрики: Достоверность | Статус | Уверенность ИИ
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Достоверность</div>
                <div class="metric-value">{score}%</div>
                <div class="metric-delta" style="color: {score_color};">{delta_prefix} {100 - score}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Статус:</div>
                <div><span class="verdict-badge {verdict_class}">{verdict_label}</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Уверенность ИИ:</div>
                <div class="metric-value">{confidence}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Карточки фактов (аудиторский шаблон — составные утверждения)
        sub_verdicts = result.get("sub_verdicts", [])
        if sub_verdicts:
            st.markdown("#### Анализ по пунктам:")
            for sv in sub_verdicts:
                status = sv.get("status", "НЕТ ДАННЫХ").upper()
                if status == "ПРАВДА":
                    card_class = "fact-card-true"
                    badge_bg = "#28a745"
                    badge_color = "#fff"
                    icon = "✅"
                elif status == "ЛОЖЬ":
                    card_class = "fact-card-false"
                    badge_bg = "#dc3545"
                    badge_color = "#fff"
                    icon = "❌"
                else:
                    card_class = "fact-card-unknown"
                    badge_bg = "#ffc107"
                    badge_color = "#333"
                    icon = "⚠️"

                citation_html = ""
                if sv.get("citation"):
                    citation_html = (
                        f'<div class="fact-citation">«{sv["citation"]}»</div>'
                    )
                    if sv.get("source"):
                        citation_html += (
                            f'<div class="fact-source">— {sv["source"]}</div>'
                        )

                st.markdown(f"""
                <div class="fact-card {card_class}">
                    <div class="fact-card-header">
                        <div class="fact-claim-text">{sv['claim']}</div>
                        <span class="fact-status-badge"
                              style="background:{badge_bg};color:{badge_color};">
                            {icon} {status}
                        </span>
                    </div>
                    {citation_html}
                </div>
                """, unsafe_allow_html=True)

        # Экспертное обоснование — только если не пустое
        reasoning = result.get("reasoning", "").strip()
        if reasoning:
            st.markdown(f"""
            <div class="reasoning-box">
                <div class="reasoning-title">Экспертное обоснование:</div>
                <div>{reasoning}</div>
            </div>
            """, unsafe_allow_html=True)

        # Chain-of-Thought (если GRPO модель)
        cot = result.get("chain_of_thought", "")
        if cot:
            with st.expander("Цепочка рассуждений (Chain-of-Thought)", expanded=False):
                st.markdown(cot.replace("\n", "  \n"))

        # Self-critique (если есть)
        critique_errors = result.get("self_critique_errors", "")
        if critique_errors and critique_errors.lower() != "нет":
            st.markdown(f"""
            <div class="critique-box">
                <div class="critique-title">Самопроверка (Self-Critique):</div>
                <div>{critique_errors}</div>
            </div>
            """, unsafe_allow_html=True)
        elif critique_errors:
            st.markdown(f"""
            <div class="critique-box">
                <div class="critique-title">Самопроверка (Self-Critique):</div>
                <div>Ошибок не обнаружено — вердикт подтверждён.</div>
            </div>
            """, unsafe_allow_html=True)

        # Первоисточники — чипы
        st.markdown("#### Первоисточники:")
        if result["sources"]:
            chips_html = '<div class="sources-chips">'
            for src in result["sources"]:
                link = src.get("link", "")
                source_name = src.get("source", "") or src.get("title", "Источник")
                # Обрезаем слишком длинные названия
                display_name = source_name[:40] + "…" if len(source_name) > 40 else source_name
                if link:
                    chips_html += (
                        f'<a href="{link}" target="_blank" class="source-chip">'
                        f'🌐 {display_name}</a>'
                    )
                else:
                    chips_html += (
                        f'<span class="source-chip-plain">📄 {display_name}</span>'
                    )
            chips_html += '</div>'
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.info("Подтверждающие источники не найдены.")

        # Технические параметры анализа (как на Рисунке 4)
        with st.expander("Технические параметры анализа", expanded=False):
            st.markdown(f"""
            <div class="tech-params">
            • <b>Модель:</b> Mistral-7B-Instruct (v0.3)<br>
            • <b>Метод оптимизации:</b> QLoRA 4-bit NF4<br>
            • <b>Инфраструктура:</b> LangChain Agentic RAG Pipeline<br>
            • <b>Задержка (Latency):</b> {latency} сек<br>
            • <b>Источники:</b> SerpAPI Google Index + DuckDuckGo (fallback)<br>
            • <b>Ключевые слова:</b> {', '.join(result['keywords'])}
            </div>
            """, unsafe_allow_html=True)

            st.text_area("Сырой ответ модели", result["raw_verdict"], height=150, disabled=True)
            if result.get("self_critique"):
                st.text_area("Сырой self-critique", result["self_critique"], height=100, disabled=True)

    elif check_btn:
        st.warning("Введите текст для проверки.")


if __name__ == "__main__":
    main()
