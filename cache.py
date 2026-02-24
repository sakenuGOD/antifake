"""Файловый кэш для результатов поиска.

Экономит API-квоту SerpAPI и ускоряет повторные запросы.
TTL по умолчанию: 1 час (новости быстро устаревают).
"""

import hashlib
import json
import os
import time
from typing import Optional, List, Dict


class SearchCache:
    """Простой файловый кэш для результатов поиска."""

    def __init__(self, cache_dir: str = ".cache/searches", ttl: int = 86400):  # 24h TTL
        self.cache_dir = cache_dir
        self.ttl = ttl
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[List[Dict]]:
        """Получить кэшированные результаты. None если протухли или нет."""
        path = os.path.join(self.cache_dir, f"{self._key(query)}.json")
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

        if time.time() - data.get("timestamp", 0) > self.ttl:
            try:
                os.remove(path)
            except OSError:
                pass
            return None

        return data.get("results", [])

    def set(self, query: str, results: List[Dict]):
        """Сохранить результаты в кэш."""
        path = os.path.join(self.cache_dir, f"{self._key(query)}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"timestamp": time.time(), "query": query, "results": results},
                    f,
                    ensure_ascii=False,
                )
        except IOError:
            pass

    def clear(self):
        """Очистка всего кэша."""
        if os.path.exists(self.cache_dir):
            for fname in os.listdir(self.cache_dir):
                try:
                    os.remove(os.path.join(self.cache_dir, fname))
                except OSError:
                    pass
