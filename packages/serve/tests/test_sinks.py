"""Sinks: file (atomic), HTTP POST, MQTT (skipped without paho-mqtt), and
`sink_from_uri` dispatch."""

from __future__ import annotations

import importlib.util

import pytest
from stoptimes.serve.sinks import FileSink, HttpSink, MqttSink, sink_from_uri


def test_file_sink_writes_bytes_atomically(tmp_path):
    target = tmp_path / "sub" / "feed.json"
    sink = FileSink(target)
    sink.publish(b'{"hello": "world"}', meta={"format": "json"})
    assert target.read_bytes() == b'{"hello": "world"}'
    # no leftover temp files
    assert list(target.parent.glob(".*")) == []


def test_file_sink_overwrites_on_republish(tmp_path):
    target = tmp_path / "feed.pb"
    sink = FileSink(target)
    sink.publish(b"first", meta={})
    sink.publish(b"second", meta={})
    assert target.read_bytes() == b"second"


def test_http_sink_posts_with_content_type(monkeypatch):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

    def fake_post(url, content=None, headers=None, timeout=None):
        calls["url"] = url
        calls["content"] = content
        calls["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)

    sink = HttpSink("https://example.test/tripupdates")
    sink.publish(b"payload-bytes", meta={"format": "pb"})

    assert calls["url"] == "https://example.test/tripupdates"
    assert calls["content"] == b"payload-bytes"
    assert calls["headers"]["Content-Type"] == "application/x-protobuf"


def test_sink_from_uri_dispatch(tmp_path):
    file_sink = sink_from_uri(f"file:{tmp_path / 'out.json'}")
    assert isinstance(file_sink, FileSink)

    http_sink = sink_from_uri("http://example.test/feed")
    assert isinstance(http_sink, HttpSink)

    mqtt_sink = sink_from_uri("mqtt://broker.local:1884/stoptimes/tu")
    assert isinstance(mqtt_sink, MqttSink)
    assert mqtt_sink.host == "broker.local"
    assert mqtt_sink.port == 1884
    assert mqtt_sink.topic == "stoptimes/tu"


def test_sink_from_uri_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        sink_from_uri("ftp://nope")


@pytest.mark.skipif(importlib.util.find_spec("paho") is None, reason="paho-mqtt not installed")
def test_mqtt_sink_publishes(monkeypatch):
    calls = {}

    def fake_single(topic, payload=None, qos=0, hostname=None, port=None):
        calls.update(topic=topic, payload=payload, qos=qos, hostname=hostname, port=port)

    monkeypatch.setattr("paho.mqtt.publish.single", fake_single)

    sink = MqttSink("mqtt://localhost:1883", "stoptimes/tu")
    sink.publish(b"feed-bytes", meta={})
    assert calls["topic"] == "stoptimes/tu"
    assert calls["payload"] == b"feed-bytes"


def test_mqtt_sink_missing_dependency_raises_clear_error(monkeypatch):
    if importlib.util.find_spec("paho") is not None:
        pytest.skip("paho-mqtt is installed; this test covers the absent-dependency path")
    sink = MqttSink("mqtt://localhost:1883", "stoptimes/tu")
    with pytest.raises(ImportError, match="paho-mqtt"):
        sink.publish(b"feed-bytes", meta={})
