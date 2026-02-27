"""Кэш вердиктов с семантическим хэшем и TTL.

Позволяет мгновенно возвращать результат для повторных проверок.
При отсутствии Redis — graceful деградация (кэш отключён).
"""

import hashlib
import json
from typing import Optional, Dict, Any


class FactCache:
    """Кэш вердиктов с семантическим хэшем и TTL."""

    def __init__(self, host="localhost", port=6379, ttl_hours=24):
        self.enabled = False
        self.ttl = ttl_hours * 3600
        try:
            import redis
            self.r = redis.Redis(host=host, port=port, decode_responses=True)
            self.r.ping()
            self.enabled = True
            print("[FactCache] Redis подключён — кэш активен")
        except Exception:
            self.r = None
            print("[FactCache] Redis не доступен — кэш отключён")

    def _hash_claim(self, claim: str) -> str:
        """Семантический хэш: нормализация + SHA256."""
        normalized = " ".join(sorted(
            w.lower() for w in claim.split()
            if len(w) > 2
        ))
        return f"fact:{hashlib.sha256(normalized.encode()).hexdigest()[:32]}"

    def get(self, claim: str) -> Optional[Dict[str, Any]]:
        """Получить кэшированный вердикт."""
        if not self.enabled:
            return None
        try:
            key = self._hash_claim(claim)
            data = self.r.get(key)
            if data:
                result = json.loads(data)
                result["_cached"] = True
                return result
        except Exception:
            pass
        return None

    def set(self, claim: str, verdict: Dict[str, Any]):
        """Сохранить вердикт с TTL."""
        if not self.enabled:
            return
        try:
            key = self._hash_claim(claim)
            cache_data = {
                "verdict": verdict.get("verdict"),
                "credibility_score": verdict.get("credibility_score"),
                "reasoning": verdict.get("reasoning"),
                "sources": verdict.get("sources", [])[:5],
            }
            self.r.set(key, json.dumps(cache_data, ensure_ascii=False), ex=self.ttl)
        except Exception:
            pass
