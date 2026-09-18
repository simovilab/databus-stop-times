"""Publish TripUpdates: file, HTTP endpoint, MQTT.

One interface, `Sink.publish(feed_bytes, meta)`, so Phase 3 can add more
(Kafka, Redis, ...) without touching the pipeline's `publish` stage, which
only knows it has "a sink with a `.publish` method".
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

__all__ = ["Sink", "FileSink", "HttpSink", "MqttSink", "sink_from_uri"]


class Sink(Protocol):
    """What every publish target implements."""

    def publish(self, feed: bytes, meta: dict[str, Any]) -> None: ...


class FileSink:
    """Writes the feed to a file, atomically (temp file + `os.replace`)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def publish(self, feed: bytes, meta: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(feed)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class HttpSink:
    """POSTs the feed to an HTTP endpoint."""

    def __init__(self, url: str, *, headers: dict[str, str] | None = None, timeout_s: float = 5.0) -> None:
        self.url = url
        self.headers = dict(headers or {})
        self.timeout_s = timeout_s

    def publish(self, feed: bytes, meta: dict[str, Any]) -> None:
        import httpx

        content_type = "application/json" if meta.get("format") == "json" else "application/x-protobuf"
        headers = {"Content-Type": content_type, **self.headers}
        response = httpx.post(self.url, content=feed, headers=headers, timeout=self.timeout_s)
        response.raise_for_status()


class MqttSink:
    """Publishes the feed to an MQTT topic.

    `paho-mqtt` is optional; the import is deferred to `publish()` so
    `import stoptimes.serve.sinks` never requires it, and its round-trip
    test is skipped when it isn't installed.
    """

    def __init__(self, broker_url: str, topic: str, *, qos: int = 0) -> None:
        parsed = urlparse(broker_url)
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 1883
        self.topic = topic
        self.qos = qos

    def publish(self, feed: bytes, meta: dict[str, Any]) -> None:
        try:
            import paho.mqtt.publish as mqtt_publish
        except ImportError as exc:  # pragma: no cover - exercised only when paho-mqtt is absent
            raise ImportError("MQTT sink needs the optional 'paho-mqtt' package: pip install paho-mqtt") from exc

        mqtt_publish.single(self.topic, payload=feed, qos=self.qos, hostname=self.host, port=self.port)


def sink_from_uri(uri: str) -> Sink:
    """Build a `Sink` from a CLI `--sink` value: `file:PATH`, `http(s)://...`, `mqtt://host/topic`."""
    if uri.startswith("file:"):
        return FileSink(uri[len("file:") :])
    if uri.startswith("http://") or uri.startswith("https://"):
        return HttpSink(uri)
    if uri.startswith("mqtt://"):
        parsed = urlparse(uri)
        topic = parsed.path.lstrip("/") or "stoptimes/tripupdates"
        broker = f"mqtt://{parsed.netloc}"
        return MqttSink(broker, topic)
    raise ValueError(f"unrecognized --sink {uri!r}: expected file:PATH, http(s)://..., or mqtt://host/topic")
