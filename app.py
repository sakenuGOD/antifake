"""Streamlit веб-интерфейс для Fact-Checker."""

import os
import streamlit as st

st.set_page_config(
    page_title="Fact-Checker — Проверка достоверности",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    """Загрузка пайплайна один раз при старте приложения."""
    from pipeline import FactCheckPipeline
    from config import SearchConfig

    api_key = os.environ.get("SERPAPI_API_KEY", "")
    if not api_key:
        return None

    search_config = SearchConfig(api_key=api_key)
    adapter_path = "adapters/fact_checker_lora"
    adapter_path = adapter_path if os.path.exists(adapter_path) else None

    return FactCheckPipeline(
        adapter_path=adapter_path,
        search_config=search_config,
    )


def render_score_bar(score: int) -> None:
    """Визуальный индикатор достоверности."""
    if score >= 70:
        color = "#28a745"
        label = "Высокая достоверность"
    elif score >= 40:
        color = "#ffc107"
        label = "Средняя достоверность"
    else:
        color = "#dc3545"
        label = "Низкая достоверность"

    st.markdown(
        f"""
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-weight: bold; font-size: 18px;">{score}/100</span>
                <span style="color: {color}; font-weight: bold;">{label}</span>
            </div>
            <div style="background: #e0e0e0; border-radius: 10px; height: 24px; overflow: hidden;">
                <div style="background: {color}; width: {score}%; height: 100%; border-radius: 10px;
                            transition: width 0.5s ease;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_verdict_badge(verdict: str) -> None:
    """Бейдж вердикта."""
    colors = {
        "ПРАВДА": ("#28a745", "#fff"),
        "ЛОЖЬ": ("#dc3545", "#fff"),
        "ЧАСТИЧНО": ("#ffc107", "#333"),
    }
    bg, fg = colors.get(verdict.upper(), ("#6c757d", "#fff"))
    st.markdown(
        f"""
        <div style="display: inline-block; padding: 8px 24px; border-radius: 8px;
                    background: {bg}; color: {fg}; font-size: 22px; font-weight: bold;
                    margin: 10px 0;">
            {verdict}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.title("Fact-Checker: Проверка достоверности новостей")
    st.markdown("Система проверки утверждений на основе нейросети Mistral 7B + RAG-поиск")

    # Проверка API ключа
    api_key = os.environ.get("SERPAPI_API_KEY", "")
    if not api_key:
        st.error(
            "Переменная окружения `SERPAPI_API_KEY` не установлена. "
            "Установите её перед запуском: `export SERPAPI_API_KEY='ваш_ключ'`"
        )
        st.stop()

    # Загрузка пайплайна
    with st.spinner("Загрузка модели..."):
        pipeline = load_pipeline()

    if pipeline is None:
        st.error("Не удалось загрузить пайплайн.")
        st.stop()

    # --- Ввод утверждения ---
    st.markdown("---")
    claim = st.text_area(
        "Введите утверждение для проверки:",
        height=100,
        placeholder="Например: Россия запустила новую космическую станцию в 2024 году",
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        check_btn = st.button("Проверить", type="primary", use_container_width=True)
    with col_info:
        st.caption("Система извлечёт ключевые слова, найдёт новости и оценит достоверность.")

    # --- Результаты ---
    if check_btn and claim.strip():
        with st.status("Анализ утверждения...", expanded=True) as status:
            st.write("Извлечение ключевых слов...")
            st.write("Поиск новостей через SerpAPI...")
            st.write("Оценка достоверности моделью...")

            result = pipeline.check(claim.strip())

            status.update(label="Анализ завершён!", state="complete")

        st.markdown("---")

        # Основные метрики
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Достоверность", f"{result['credibility_score']}/100")
        with col2:
            st.metric("Уверенность модели", f"{result['confidence']}%")
        with col3:
            st.metric("Время анализа", f"{result['total_time']} сек.")

        # Шкала достоверности
        render_score_bar(result["credibility_score"])

        # Вердикт
        st.subheader("Вердикт")
        render_verdict_badge(result["verdict"])

        # Обоснование
        st.subheader("Обоснование")
        st.write(result["reasoning"])

        # Источники
        st.subheader("Найденные источники")
        if result["sources"]:
            for i, src in enumerate(result["sources"], 1):
                with st.expander(f"{i}. {src['title']}", expanded=False):
                    st.write(f"**Источник:** {src['source']}")
                    if src.get("date"):
                        st.write(f"**Дата:** {src['date']}")
                    if src.get("link"):
                        st.markdown(f"[Открыть статью]({src['link']})")
        else:
            st.info("Источники не найдены.")

        # Технические параметры (сворачиваемый блок)
        with st.expander("Технические параметры", expanded=False):
            st.write(f"**Ключевые слова:** {', '.join(result['keywords'])}")
            st.write(f"**Источники (из вердикта):** {result['sources_text']}")
            st.text_area("Сырой ответ модели", result["raw_verdict"], height=200, disabled=True)
            st.text_area(
                "Результаты поиска (форматированные)",
                result["search_results_formatted"],
                height=200,
                disabled=True,
            )

    elif check_btn:
        st.warning("Введите утверждение для проверки.")


if __name__ == "__main__":
    main()
