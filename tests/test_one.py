"""Single-claim diagnostic to isolate crash."""
import _path  # noqa: F401,E402 — inject project root into sys.path
import os, sys

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import traceback

try:
    from pipeline import FactCheckPipeline
    project_root = os.path.dirname(os.path.abspath(__file__))
    sft_path = os.path.join(project_root, "adapters", "fact_checker_lora")
    pipe = FactCheckPipeline(adapter_path=sft_path)
    print("\n=== Pipeline loaded ===\n", flush=True)
    result = pipe.check("Александр Пушкин написал роман в стихах Евгений Онегин")
    print("\n=== RESULT ===", flush=True)
    print(f"verdict: {result.get('verdict')}", flush=True)
    print(f"score:   {result.get('credibility_score')}", flush=True)
    print(f"reasoning: {(result.get('reasoning') or '')[:300]}", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(2)
