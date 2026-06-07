from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

from openai import OpenAI
from api_model_proxy import APIModelProxy


class PersistantLoggingProxy(APIModelProxy):
    """Subclass of :class:`~api_model_proxy.proxy.APIModelProxy` that logs
    every request and response to daily files on disk.

    Logs are stored in *log_dir* as ``{YYYY-MM-DD}.jsonl`` (JSON Lines)
    or ``{YYYY-MM-DD}.yaml`` (multi-document YAML).  Each entry is
    flushed after every write so data is durable even if the process
    crashes.

    Usage::

        proxy = PersistantLoggingProxy(client)                  # -> logs/YYYY-MM-DD.jsonl
        proxy = PersistantLoggingProxy(client, format="yaml")   # -> logs/YYYY-MM-DD.yaml
        proxy = PersistantLoggingProxy(client, log_dir="data")  # -> data/YYYY-MM-DD.jsonl

    Args:
        openai_client: An ``openai.OpenAI`` instance.
        log_dir: Directory to store log files (default ``"logs"``).
        format: ``"json"`` or ``"yaml"`` (default ``"json"``).
    """

    def __init__(
        self,
        openai_client: OpenAI,
        log_dir: str = "logs",
        format: str = "json",
    ) -> None:
        super().__init__(openai_client)
        if format not in ("json", "yaml"):
            raise ValueError(f"format must be 'json' or 'yaml', got {format!r}")
        self._format = format
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def _preprocess_request(self, request: dict) -> dict:
        self._counter += 1
        self._write_log("request", request)
        return request

    def _postprocess_response(self, response: dict) -> dict:
        self._write_log("response", response)
        return response

    def _write_log(self, direction: str, body: dict) -> None:
        now = datetime.utcnow()
        date_str = now.strftime("%Y-%m-%d")
        ext = "jsonl" if self._format == "json" else "yaml"
        path = self._log_dir / f"{date_str}.{ext}"

        entry = {
            "timestamp": now.isoformat() + "Z",
            "counter": self._counter,
            "direction": direction,
            "body": body,
        }

        if self._format == "json":
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
        else:
            import yaml

            with open(path, "a") as f:
                f.write("---\n")
                yaml.dump(entry, f, default_flow_style=False)
                f.flush()
