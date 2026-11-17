"""Framework-owned experiment defaults and append-only request accounting."""
import json
import os
import threading
from datetime import datetime, timezone

META_MODEL = "anthropic/claude-sonnet-4-5-20250929"
TASK_MODEL = "openai/o4-mini"
MAX_GENERATION = 30
RANDOM_SEED = 42
_lock = threading.Lock()


def record_usage(record):
    path = os.environ.get("RSI_USAGE_PATH")
    if not path:
        return
    record = dict(record, generation=int(os.environ.get("RSI_GENERATION", "0")),
                  category=os.environ.get("RSI_CATEGORY", "unspecified"),
                  timestamp=datetime.now(timezone.utc).isoformat())
    with _lock:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_usage(path):
    result = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                key = entry["category"]
                totals = result.setdefault(key, dict(requests=0, input_tokens=0,
                    output_tokens=0, missing_usage=0, failed_requests=0))
                totals["requests"] += 1
                totals["failed_requests"] += entry.get("status") == "error"
                totals["missing_usage"] += entry.get("input_tokens") is None or entry.get("output_tokens") is None
                for field in ("input_tokens", "output_tokens"):
                    totals[field] += entry.get(field) or 0
    return result
