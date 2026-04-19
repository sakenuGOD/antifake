"""
V15 UI Test — прогоняет все типы утверждений и показывает результат
через новый UI: СОСТАВНОЕ, НЕ УВЕРЕНА + объяснение, чипы сигналов.

Запуск:
    streamlit run test_ui_v15.py
"""
import _path  # noqa: F401,E402 — inject project root into sys.path

import os
import time
import streamlit as st

st.set_page_config(
    page_title="V15 UI Test — Все типы вердиктов",
    page_icon="🧪",
    layout="wide",
)

# ── CSS (полная копия из app.py + дополнения для теста) ──
st.markdown("""
<style>
    .main-header { font-size: 28px; font-weight: bold; margin-bottom: 5px; }
    .sub-header  { color: #888; font-size: 14px; margin-bottom: 20px; }

    .metric-box   { text-align: center; flex: 1; }
    .metric-label { color: #888; font-size: 12px; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: bold; }
    .metric-delta { font-size: 12px; }

    .verdict-badge {
        display: inline-block; padding: 6px 18px; border-radius: 6px;
        font-size: 14px; font-weight: bold; margin: 5px 0;
    }
    .verdict-false   { background: #dc3545; color: #fff; }
    .verdict-true    { background: #28a745; color: #fff; }
    .verdict-partial { background: #ffc107; color: #333; }

    .reasoning-box {
        background: #2a1a1a; border: 1px solid #5a2d2d; border-radius: 8px;
        padding: 16px; margin: 15px 0; line-height: 1.6;
    }
    .reasoning-title {
        color: #ff6b6b; font-weight: bold; font-size: 14px;
        margin-bottom: 10px; text-transform: uppercase;
    }

    .fact-card {
        border-radius: 8px; padding: 14px 16px; margin: 10px 0;
        border-left: 5px solid; background: #1e1e2e;
    }
    .fact-card-true    { border-left-color: #28a745; background: #0d2617; }
    .fact-card-false   { border-left-color: #dc3545; background: #2a0d0d; }
    .fact-card-unknown { border-left-color: #ffc107; background: #2a2100; }
    .fact-card-header  {
        display: flex; justify-content: space-between;
        align-items: flex-start; gap: 12px;
    }
    .fact-claim-text   { font-size: 15px; line-height: 1.5; flex: 1; }
    .fact-status-badge {
        padding: 4px 12px; border-radius: 4px; font-size: 12px;
        font-weight: bold; white-space: nowrap;
    }
    .fact-citation {
        color: #bbb; font-size: 13px; margin-top: 10px;
        border-top: 1px solid #333; padding-top: 8px; font-style: italic;
    }
    .fact-source { color: #777; font-size: 12px; margin-top: 3px; }

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

    /* scorecard */
    .sc-pass { color: #28a745; font-weight: bold; }
    .sc-fail { color: #dc3545; font-weight: bold; }
    .sc-skip { color: #888; }

    /* signal chip inline */
    .sig-chip {
        display: inline-flex; align-items: center;
        border-radius: 20px; padding: 4px 12px;
        font-size: 13px; margin: 2px;
    }
    .sig-green  { background:#0d2617; border:1px solid #28a745; color:#4caf50; }
    .sig-red    { background:#2a0d0d; border:1px solid #dc3545; color:#dc3545; }
    .sig-gray   { background:#222;    border:1px solid #444;    color:#888;    }
</style>
""", unsafe_allow_html=True)


# ── Тестовые утверждения (покрывают все ветви V15) ──
TEST_CLAIMS = [
    # ── ПРАВДА ──
    {
        "claim": "Токио является столицей Японии",
        "expected": "ПРАВДА",
        "category": "Простой факт",
        "icon": "🟢",
        "checks": {
            "verdict": "ПРАВДА",
            "score_min": 60,
        },
    },
    {
        "claim": "Земля вращается вокруг Солнца",
        "expected": "ПРАВДА",
        "category": "Простой факт",
        "icon": "🟢",
        "checks": {
            "verdict": "ПРАВДА",
            "score_min": 60,
        },
    },
    # ── ЛОЖЬ (простой) ──
    {
        "claim": "Австралия является частью Европы",
        "expected": "ЛОЖЬ",
        "category": "Простой ложный факт",
        "icon": "🔴",
        "checks": {
            "verdict": "ЛОЖЬ",
            "score_max": 34,
        },
    },
    # ── ЛОЖЬ (числа) ──
    {
        "claim": "Расстояние от Земли до Солнца составляет 15 миллионов километров",
        "expected": "ЛОЖЬ",
        "category": "Числовое несоответствие",
        "icon": "🔴",
        "checks": {
            "verdict": "ЛОЖЬ",
        },
    },
    # ── ЛОЖЬ (даты) ──
    {
        "claim": "Первая мировая война началась в 1939 году",
        "expected": "ЛОЖЬ",
        "category": "Подмена даты",
        "icon": "🔴",
        "checks": {
            "verdict": "ЛОЖЬ",
        },
    },
    # ── ЛОЖЬ (миф) ──
    {
        "claim": "Молния никогда не бьёт в одно и то же место дважды",
        "expected": "ЛОЖЬ",
        "category": "Миф / заблуждение",
        "icon": "🔴",
        "checks": {
            "verdict": "ЛОЖЬ",
        },
    },
    {
        "claim": "Золотая рыбка помнит события только 3 секунды",
        "expected": "ЛОЖЬ",
        "category": "Миф / заблуждение",
        "icon": "🔴",
        "checks": {
            "verdict": "ЛОЖЬ",
        },
    },
    {
        "claim": "Человек проглатывает в среднем 8 пауков в год во сне",
        "expected": "ЛОЖЬ",
        "category": "Миф / заблуждение",
        "icon": "🔴",
        "checks": {
            "verdict": "ЛОЖЬ",
        },
    },
    {
        "claim": "Сахар вызывает гиперактивность у детей",
        "expected": "ЛОЖЬ",
        "category": "Миф / заблуждение",
        "icon": "🔴",
        "checks": {
            "verdict": "ЛОЖЬ",
        },
    },
    # ── СОСТАВНОЕ ──
    {
        "claim": "Альберт Эйнштейн родился в Германии и получил Нобелевскую премию по математике",
        "expected": "СОСТАВНОЕ",
        "category": "Составное утверждение",
        "icon": "🟠",
        "checks": {
            "verdict": "СОСТАВНОЕ",
            "has_sub_verdicts": True,
            "expected_subs": ["ПРАВДА", "ЛОЖЬ"],
        },
    },
    {
        "claim": "Земля имеет один естественный спутник и расстояние до Луны составляет около 38 тысяч километров",
        "expected": "СОСТАВНОЕ",
        "category": "Составное утверждение",
        "icon": "🟠",
        "checks": {
            "verdict": "СОСТАВНОЕ",
            "has_sub_verdicts": True,
            "expected_subs": ["ПРАВДА", "ЛОЖЬ"],
        },
    },
    {
        "claim": "Дмитрий Менделеев изобрёл периодическую таблицу и открыл водку",
        "expected": "СОСТАВНОЕ",
        "category": "Составное утверждение",
        "icon": "🟠",
        "checks": {
            "verdict": "СОСТАВНОЕ",
            "has_sub_verdicts": True,
            "expected_subs": ["ПРАВДА", "ЛОЖЬ"],
        },
    },
    # ── НЕ УВЕРЕНА ──
    {
        "claim": "Мобильные телефоны вызывают рак мозга",
        "expected": "НЕ УВЕРЕНА",
        "category": "Дискуссионный вопрос",
        "icon": "🟡",
        "checks": {
            "verdict": "НЕ УВЕРЕНА",
            "has_explanation": True,
        },
    },
    # ── Подмена персоны ──
    {
        "claim": "Александр Пушкин написал роман Война и мир",
        "expected": "ЛОЖЬ",
        "category": "Подмена персоны",
        "icon": "🔴",
        "checks": {
            "verdict": "ЛОЖЬ",
        },
    },
    # ── Числовой факт (правда) ──
    {
        "claim": "Температура кипения воды составляет 100 градусов Цельсия",
        "expected": "ПРАВДА",
        "category": "Числовой факт",
        "icon": "🟢",
        "checks": {
            "verdict": "ПРАВДА",
        },
    },
]

NORM = {
    "СОСТАВНОЕ": "СОСТАВНОЕ", "СОСТАВНОЕ УТВЕРЖДЕНИЕ": "СОСТАВНОЕ",
    "НЕ УВЕРЕНА": "НЕ УВЕРЕНА",
    "ЛОЖЬ": "ЛОЖЬ", "FALSE": "ЛОЖЬ", "ФЕЙК": "ЛОЖЬ",
    "ПРАВДА": "ПРАВДА", "TRUE": "ПРАВДА",
    "МАНИПУЛЯЦИЯ": "СОСТАВНОЕ", "ПОЛУПРАВДА": "СОСТАВНОЕ",
    "НЕ ПОДТВЕРЖДЕНО": "НЕ УВЕРЕНА",
    "ЧАСТИЧНО ПОДТВЕРЖДЕНО": "СОСТАВНОЕ",
}


def norm_v(v: str) -> str:
    v = v.strip().upper()
    for k, val in NORM.items():
        if k in v:
            return val
    return "НЕ УВЕРЕНА"


# ── Рендер одного результата (идентичен app.py) ──

def render_result(result: dict):
    """Полный рендер результата — точная копия app.py"""
    score = result["credibility_score"]
    confidence = result["confidence"]
    verdict = result["verdict"]
    latency = result.get("total_time", 0)

    v_upper = verdict.upper().strip()
    is_composite = result.get("_composite", False) or v_upper == "СОСТАВНОЕ"

    if is_composite:
        score_color, delta_prefix = "#fd7e14", "~"
        verdict_class, verdict_label = "verdict-partial", "СОСТАВНОЕ УТВЕРЖДЕНИЕ"
    elif v_upper in ("ЛОЖЬ", "FALSE", "ФЕЙК"):
        score_color, delta_prefix = "#dc3545", "↓"
        verdict_class, verdict_label = "verdict-false", "ЛОЖЬ / ФЕЙК"
    elif v_upper in ("ПРАВДА", "TRUE"):
        score_color, delta_prefix = "#28a745", "↑"
        verdict_class, verdict_label = "verdict-true", "ПРАВДА / TRUE"
    else:
        score_color, delta_prefix = "#ffc107", "~"
        verdict_class, verdict_label = "verdict-partial", "НЕ УВЕРЕНА"

    # ── Метрики ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">Достоверность</div>
            <div class="metric-value">{score}%</div>
            <div class="metric-delta" style="color:{score_color};">{delta_prefix} {100-score}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">Статус:</div>
            <div><span class="verdict-badge {verdict_class}">{verdict_label}</span></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">Уверенность ИИ:</div>
            <div class="metric-value">{confidence}%</div>
        </div>""", unsafe_allow_html=True)

    # ── НЕ УВЕРЕНА — объяснение ──
    if v_upper == "НЕ УВЕРЕНА" and result.get("_explanation"):
        st.info(f"Почему нет данных: {result['_explanation']}")

    # ── Карточки под-вердиктов ──
    sub_verdicts = result.get("sub_verdicts", [])
    if sub_verdicts:
        if is_composite:
            st.markdown("### Результат по пунктам:")
        else:
            st.markdown("#### Анализ по пунктам:")

        for sv in sub_verdicts:
            s = sv.get("status", "НЕ УВЕРЕНА").upper()
            if s == "ПРАВДА":
                cc, bg, fc, ic = "fact-card-true",  "#28a745", "#fff", "✅"
            elif s == "ЛОЖЬ":
                cc, bg, fc, ic = "fact-card-false", "#dc3545", "#fff", "❌"
            else:
                cc, bg, fc, ic = "fact-card-unknown","#ffc107", "#333", "⚠️"

            cite = ""
            if sv.get("citation"):
                cite = f'<div class="fact-citation">«{sv["citation"]}»</div>'
                if sv.get("source"):
                    cite += f'<div class="fact-source">— {sv["source"]}</div>'

            st.markdown(f"""<div class="fact-card {cc}">
                <div class="fact-card-header">
                    <div class="fact-claim-text">{sv['claim']}</div>
                    <span class="fact-status-badge" style="background:{bg};color:{fc};">
                        {ic} {s}
                    </span>
                </div>{cite}
            </div>""", unsafe_allow_html=True)

    # ── Чипы сигналов ──
    _ef = result.get("_ensemble_features", {})
    if _ef:
        chips = []
        wd_c = _ef.get("wd_confirmed", 0)
        wd_x = _ef.get("wd_contradicted", 0)
        if wd_c > 0 and wd_x == 0:
            chips.append('<span class="sig-chip sig-green">Wikidata: подтверждает</span>')
        elif wd_x > 0:
            chips.append('<span class="sig-chip sig-red">Wikidata: противоречит</span>')

        ne = _ef.get("nli_ent", 0); nc = _ef.get("nli_con", 0)
        if ne >= 0.5 and nc < 0.3:
            chips.append('<span class="sig-chip sig-green">NLI: подтверждает</span>')
        elif nc >= 0.5 and ne < 0.3:
            chips.append('<span class="sig-chip sig-red">NLI: противоречит</span>')
        elif ne > 0 or nc > 0:
            chips.append('<span class="sig-chip sig-gray">NLI: нейтрально</span>')

        nm = _ef.get("nums_match", 0); nx = _ef.get("nums_mismatch", 0)
        if nm > 0 and nx == 0:
            chips.append('<span class="sig-chip sig-green">Числа: совпадают</span>')
        elif nx > 0:
            chips.append('<span class="sig-chip sig-red">Числа: расхождение</span>')

        ns = _ef.get("num_sources", 0)
        if ns > 0:
            chips.append(f'<span class="sig-chip sig-gray">Источники: {ns}</span>')

        if chips:
            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin:10px 0;">{"".join(chips)}</div>',
                unsafe_allow_html=True,
            )

    # ── Обоснование ──
    reasoning = result.get("reasoning", "").strip()
    if reasoning:
        st.markdown(f"""<div class="reasoning-box">
            <div class="reasoning-title">Экспертное обоснование:</div>
            <div>{reasoning}</div>
        </div>""", unsafe_allow_html=True)

    # ── CoT ──
    cot = result.get("chain_of_thought", "")
    if cot:
        with st.expander("Цепочка рассуждений (Chain-of-Thought)", expanded=False):
            st.markdown(cot.replace("\n", "  \n"))

    # ── Источники ──
    sources = result.get("sources", [])
    if sources:
        st.markdown("#### Первоисточники:")
        ch = '<div class="sources-chips">'
        for src in sources:
            link = src.get("link", "")
            name = (src.get("source") or src.get("title", "Источник"))[:40]
            if link:
                ch += f'<a href="{link}" target="_blank" class="source-chip">🌐 {name}</a>'
            else:
                ch += f'<span class="source-chip-plain">📄 {name}</span>'
        ch += '</div>'
        st.markdown(ch, unsafe_allow_html=True)

    # ── Технические параметры ──
    with st.expander("Технические параметры", expanded=False):
        st.markdown(f"""<div style="color:#888;font-size:13px;line-height:1.8;">
            Latency: {latency:.1f}s &nbsp;|&nbsp;
            Keywords: {', '.join(result.get('keywords', []))}
        </div>""", unsafe_allow_html=True)
        st.text_area("Raw verdict", result.get("raw_verdict", ""), height=120, disabled=True)


# ── Проверки ──

def evaluate(result: dict, tc: dict) -> tuple:
    """Возвращает (passed: bool, details: list[str])."""
    checks = tc.get("checks", {})
    fails = []

    predicted = norm_v(result.get("verdict", ""))

    # verdict
    exp = checks.get("verdict")
    if exp and predicted != exp:
        fails.append(f"Вердикт: ожидали **{exp}**, получили **{predicted}**")

    # score range
    smin = checks.get("score_min")
    if smin is not None and result["credibility_score"] < smin:
        fails.append(f"Score {result['credibility_score']} < {smin}")
    smax = checks.get("score_max")
    if smax is not None and result["credibility_score"] > smax:
        fails.append(f"Score {result['credibility_score']} > {smax}")

    # sub-verdicts presence
    if checks.get("has_sub_verdicts"):
        svs = result.get("sub_verdicts", [])
        if len(svs) < 2:
            fails.append(f"Sub-verdicts: {len(svs)} (нужно >= 2)")

    # expected sub-verdicts content
    es = checks.get("expected_subs")
    if es:
        actual = sorted(sv.get("status", "").upper() for sv in result.get("sub_verdicts", []))[:len(es)]
        if sorted(es) != actual:
            fails.append(f"Sub-статусы: ожидали {sorted(es)}, получили {actual}")

    # explanation
    if checks.get("has_explanation"):
        expl = result.get("_explanation", "")
        if not expl or len(expl) < 10:
            fails.append(f"Объяснение отсутствует или слишком короткое")

    # no old verdicts in raw output
    raw = str(result)
    if "НЕ ПОДТВЕРЖДЕНО" in raw:
        fails.append("Найден старый вердикт **НЕ ПОДТВЕРЖДЕНО** в выводе")
    if "МАНИПУЛЯЦИЯ" in raw:
        fails.append("Найден старый вердикт **МАНИПУЛЯЦИЯ** в выводе")

    return len(fails) == 0, fails


# ── Загрузка пайплайна ──

@st.cache_resource
def load_pipeline():
    from pipeline import FactCheckPipeline
    from config import SearchConfig
    from model import find_best_adapter
    api_key = os.environ.get("SERPAPI_API_KEY", "")
    return FactCheckPipeline(
        adapter_path=find_best_adapter(),
        search_config=SearchConfig(api_key=api_key),
    )


# ── MAIN ──

def main():
    st.markdown('<div class="main-header">🧪 V15 UI Test — Все типы вердиктов</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        '3 вердикта (ПРАВДА / ЛОЖЬ / НЕ УВЕРЕНА) + СОСТАВНОЕ | '
        '15 утверждений | Полный UI-рендер'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Загрузка модели Mistral 7B..."):
        pipeline = load_pipeline()
    if pipeline is None:
        st.error("Не удалось загрузить пайплайн.")
        st.stop()

    st.success("Пайплайн загружен. Нажмите кнопку для запуска.")

    # ── Scorecard placeholder ──
    scorecard_ph = st.empty()

    run_btn = st.button("Запустить все 15 утверждений", type="primary", use_container_width=True)
    if not run_btn:
        # Show claims preview
        st.markdown("#### Утверждения для проверки:")
        for i, tc in enumerate(TEST_CLAIMS, 1):
            st.markdown(
                f"**{i}.** {tc['icon']} `{tc['expected']:<12s}` "
                f"*({tc['category']})* — {tc['claim']}"
            )
        st.stop()

    # ── Прогон ──
    results_data = []       # (tc, result, passed, fails, elapsed)
    total_pass = 0
    total_fail = 0
    total_time = 0.0

    progress = st.progress(0, text="Проверяем утверждения...")

    for i, tc in enumerate(TEST_CLAIMS):
        progress.progress(
            (i) / len(TEST_CLAIMS),
            text=f"[{i+1}/{len(TEST_CLAIMS)}] {tc['claim'][:60]}...",
        )

        t0 = time.time()
        try:
            result = pipeline.check(tc["claim"])
        except Exception as e:
            result = {
                "claim": tc["claim"],
                "credibility_score": 0,
                "verdict": "ERROR",
                "confidence": 0,
                "reasoning": str(e),
                "sources": [],
                "keywords": [],
                "raw_verdict": "",
                "total_time": 0,
                "sub_verdicts": [],
            }
        elapsed = time.time() - t0
        total_time += elapsed

        passed, fails = evaluate(result, tc)
        if passed:
            total_pass += 1
        else:
            total_fail += 1

        results_data.append((tc, result, passed, fails, elapsed))

    progress.progress(1.0, text="Готово!")

    # ── Scorecard ──
    n = len(TEST_CLAIMS)
    pct = total_pass / n if n else 0
    bar_color = "#28a745" if pct >= 0.8 else ("#ffc107" if pct >= 0.5 else "#dc3545")

    scorecard_ph.markdown(f"""
    <div style="background:#111;border:2px solid {bar_color};border-radius:12px;
                padding:20px 30px;margin:15px 0;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:36px;font-weight:bold;color:{bar_color};">
                    {total_pass}/{n}
                </span>
                <span style="color:#888;font-size:16px;margin-left:10px;">
                    ({pct:.0%}) — {'ALL PASS' if total_fail == 0 else f'{total_fail} FAILED'}
                </span>
            </div>
            <div style="color:#888;font-size:14px;text-align:right;">
                Время: {total_time:.0f}s (avg {total_time/n:.1f}s)<br>
                ПРАВДА: {sum(1 for t,_,_,_,_ in results_data if t['expected']=='ПРАВДА')} |
                ЛОЖЬ: {sum(1 for t,_,_,_,_ in results_data if t['expected']=='ЛОЖЬ')} |
                СОСТАВНОЕ: {sum(1 for t,_,_,_,_ in results_data if t['expected']=='СОСТАВНОЕ')} |
                НЕ УВЕРЕНА: {sum(1 for t,_,_,_,_ in results_data if t['expected']=='НЕ УВЕРЕНА')}
            </div>
        </div>
        <div style="background:#333;border-radius:6px;height:8px;margin-top:12px;">
            <div style="background:{bar_color};border-radius:6px;height:8px;width:{pct*100:.0f}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Сводная таблица ──
    st.markdown("---")
    st.markdown("### Сводная таблица")
    header = f"| # | {'Утверждение':<50s} | Ожидали | Получили | Score | Время | Статус |"
    sep    = f"|---|{'---'*1}|---------|----------|-------|-------|--------|"
    rows = [header, sep]
    for idx, (tc, result, passed, fails, elapsed) in enumerate(results_data, 1):
        predicted = norm_v(result.get("verdict", ""))
        sc = result.get("credibility_score", 0)
        ok_str = "PASS" if passed else "FAIL"
        clm = tc["claim"][:48]
        rows.append(
            f"| {idx} | {clm} | {tc['expected']} | {predicted} | {sc} | {elapsed:.1f}s | {ok_str} |"
        )
    st.markdown("\n".join(rows))

    # ── Подробные карточки ──
    st.markdown("---")
    st.markdown("### Подробные результаты")

    for idx, (tc, result, passed, fails, elapsed) in enumerate(results_data, 1):
        predicted = norm_v(result.get("verdict", ""))
        status_icon = "✅" if passed else "❌"
        sc = result.get("credibility_score", 0)

        with st.expander(
            f"{status_icon} [{idx}/{n}] {tc['icon']} {tc['claim'][:70]}  "
            f"→  {predicted} ({sc}%)  |  {elapsed:.1f}s",
            expanded=not passed,     # автоматически раскрываем провалы
        ):
            # Мета
            mc1, mc2, mc3 = st.columns(3)
            mc1.markdown(f"**Категория:** {tc['category']}")
            mc2.markdown(f"**Ожидали:** `{tc['expected']}`")
            mc3.markdown(f"**Получили:** `{predicted}`")

            if fails:
                for f in fails:
                    st.error(f)

            st.markdown("---")

            # Полный рендер как в app.py
            render_result(result)


if __name__ == "__main__":
    main()
