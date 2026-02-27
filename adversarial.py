"""Адвокат Дьявола — Adversarial Agent Debate.

Режим двустороннего поиска: одна ветка ищет подтверждения, другая — опровержения.
Судья выносит вердикт на основе доводов обеих сторон.
Убивает Confirmation Bias.

Используется опционально для утверждений со смешанными NLI-сигналами.
"""

import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


class DebateState(TypedDict):
    claim: str
    defender_evidence: Annotated[list, operator.add]
    prosecutor_evidence: Annotated[list, operator.add]
    judge_verdict: str


def _format_evidence(evidence: list) -> str:
    """Форматирует список улик для судьи."""
    parts = []
    for i, e in enumerate(evidence[:5], 1):
        src = e.get("source", "неизвестен")
        snippet = e.get("snippet", "")[:200]
        parts.append(f"  {i}. [{src}] {snippet}")
    return "\n".join(parts) if parts else "  Доказательств не найдено."


class AdversarialDebate:
    """Фасад для запуска дебатов защитник/прокурор/судья."""

    def __init__(self, searcher, generate_fn=None):
        """
        Args:
            searcher: FactCheckSearcher instance
            generate_fn: callable(prompt: str) -> str (LLM для судьи)
        """
        self.searcher = searcher
        self.generate_fn = generate_fn
        self._graph = None

    def _build_graph(self):
        """Строит LangGraph только если библиотека доступна."""
        if not LANGGRAPH_AVAILABLE:
            return None

        def defender_search(state: DebateState) -> dict:
            claim = state["claim"]
            queries = [
                f"{claim} подтверждение доказательство факт",
                f"{claim} confirmed true evidence",
            ]
            evidence = []
            for q in queries:
                results = self.searcher._search_ddg(q, max_results=3)
                evidence.extend(results)
            return {"defender_evidence": evidence}

        def prosecutor_search(state: DebateState) -> dict:
            claim = state["claim"]
            queries = [
                f"{claim} опровержение миф фейк разоблачение",
                f"{claim} debunked false fake",
            ]
            evidence = []
            for q in queries:
                results = self.searcher._search_ddg(q, max_results=3)
                evidence.extend(results)
            return {"prosecutor_evidence": evidence}

        def judge_verdict(state: DebateState) -> dict:
            prompt = f"""[INST]Ты — беспристрастный судья. Прочитай доводы обеих сторон и вынеси вердикт.

УТВЕРЖДЕНИЕ: {state['claim']}

АРГУМЕНТЫ ЗАЩИТЫ (подтверждения):
{_format_evidence(state['defender_evidence'])}

АРГУМЕНТЫ ОБВИНЕНИЯ (опровержения):
{_format_evidence(state['prosecutor_evidence'])}

Вынеси вердикт: ПРАВДА / ЛОЖЬ / НЕ ПОДТВЕРЖДЕНО.
Объясни в 1-2 предложениях, почему одна сторона убедительнее.[/INST]"""

            if self.generate_fn:
                verdict = self.generate_fn(prompt)
            else:
                # Fallback: простое сравнение количества доказательств
                d_count = len(state.get("defender_evidence", []))
                p_count = len(state.get("prosecutor_evidence", []))
                if d_count > p_count * 1.2:
                    verdict = "ПРАВДА — больше подтверждающих доказательств"
                elif p_count > d_count * 1.2:
                    verdict = "ЛОЖЬ — больше опровергающих доказательств"
                else:
                    verdict = "НЕ ПОДТВЕРЖДЕНО — доводы сторон сопоставимы"
            return {"judge_verdict": verdict}

        builder = StateGraph(DebateState)
        builder.add_node("defender", defender_search)
        builder.add_node("prosecutor", prosecutor_search)
        builder.add_node("judge", judge_verdict)
        builder.add_edge(START, "defender")
        builder.add_edge(START, "prosecutor")
        builder.add_edge("defender", "judge")
        builder.add_edge("prosecutor", "judge")
        builder.add_edge("judge", END)
        return builder.compile()

    def debate(self, claim: str) -> Dict[str, Any]:
        """Запускает дебаты и возвращает результат.

        Returns:
            {"judge_verdict": str, "defender_count": int, "prosecutor_count": int}
        """
        if LANGGRAPH_AVAILABLE:
            if self._graph is None:
                self._graph = self._build_graph()
            if self._graph:
                result = self._graph.invoke({"claim": claim})
                return {
                    "judge_verdict": result.get("judge_verdict", "НЕ ПОДТВЕРЖДЕНО"),
                    "defender_count": len(result.get("defender_evidence", [])),
                    "prosecutor_count": len(result.get("prosecutor_evidence", [])),
                }

        # Fallback без LangGraph — последовательный поиск
        defender_evidence = []
        prosecutor_evidence = []
        for q in [f"{claim} подтверждение факт", f"{claim} confirmed true"]:
            defender_evidence.extend(self.searcher._search_ddg(q, max_results=3))
        for q in [f"{claim} опровержение фейк", f"{claim} debunked false"]:
            prosecutor_evidence.extend(self.searcher._search_ddg(q, max_results=3))

        d_count = len(defender_evidence)
        p_count = len(prosecutor_evidence)
        if d_count > p_count * 1.2:
            verdict = "ПРАВДА"
        elif p_count > d_count * 1.2:
            verdict = "ЛОЖЬ"
        else:
            verdict = "НЕ ПОДТВЕРЖДЕНО"

        return {
            "judge_verdict": verdict,
            "defender_count": d_count,
            "prosecutor_count": p_count,
        }
