"""Network volumes through a fake RunPod REST API, and a job whose volume
bucket is read and written at the worker's mount."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from ico.cli import main
from ico.remote import LocalExecutor, RunPodApi, RunPodApiError, Volume, result, submit
from tests.apriori.ico.remote.test_jobs import PACKAGE, SOURCES, op

KEY = "rest-key"


class FakeRest:
    """/networkvolumes and /endpoints/<id> of the REST API."""

    def __init__(self) -> None:
        self.volumes: dict[str, dict[str, Any]] = {}
        self.patches: list[tuple[str, dict[str, Any]]] = []
        self.fail_patch = False
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                fake.serve(self, "GET")

            def do_POST(self) -> None:
                fake.serve(self, "POST")

            def do_PATCH(self) -> None:
                fake.serve(self, "PATCH")

            def do_DELETE(self) -> None:
                fake.serve(self, "DELETE")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def serve(self, request: BaseHTTPRequestHandler, method: str) -> None:
        if request.headers["authorization"] != f"Bearer {KEY}":
            return self.reply(request, 401, {"error": "unauthorized"})
        length = int(request.headers.get("content-length") or 0)
        body: dict[str, Any] = json.loads(request.rfile.read(length)) if length else {}
        path = request.path.removeprefix("/v1")
        if path == "/networkvolumes" and method == "GET":
            return self.reply(request, 200, list(self.volumes.values()))
        if path == "/networkvolumes" and method == "POST":
            volume = {"id": f"vol{len(self.volumes) + 1}", **body}
            self.volumes[volume["id"]] = volume
            return self.reply(request, 200, volume)
        if path.startswith("/networkvolumes/") and method == "DELETE":
            if self.volumes.pop(path.rsplit("/", 1)[1], None) is None:
                return self.reply(request, 404, {"error": "not found"})
            return self.reply(request, 204, None)
        if path.startswith("/endpoints/") and method == "PATCH":
            if self.fail_patch:
                return self.reply(request, 400, {"error": "bad datacenter"})
            self.patches.append((path.rsplit("/", 1)[1], body))
            return self.reply(request, 200, {"id": path.rsplit("/", 1)[1]})
        self.reply(request, 404, {"error": "no route"})

    def reply(self, request: BaseHTTPRequestHandler, code: int, body: object) -> None:
        data = json.dumps(body).encode() if body is not None else b""
        request.send_response(code)
        request.send_header("content-length", str(len(data)))
        request.end_headers()
        request.wfile.write(data)


@pytest.fixture
def rest(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeRest]:
    fake = FakeRest()
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setenv("ICO_RUNPOD_REST", fake.url)
    yield fake
    fake.server.shutdown()


def test_an_order_brings_a_volume_up_and_down(rest: FakeRest) -> None:
    api = RunPodApi(base_url=rest.url)
    volume = api.volume_up("ep1", 150, "EU-RO-1", "order-7")

    assert volume == Volume("vol1", "order-7", 150, "EU-RO-1")
    assert rest.patches == [
        ("ep1", {"networkVolumeId": "vol1", "dataCenterIds": ["EU-RO-1"]})
    ]
    assert api.volumes() == [volume]
    assert volume.uri("runs/r1/in") == "s3://vol1/runs/r1/in"
    assert volume.s3_env() == {
        "AWS_ENDPOINT_URL_S3": "https://s3api-eu-ro-1.runpod.io", "AWS_REGION": "EU-RO-1",
    }  # fmt: skip

    api.volume_down("ep1", volume.id)
    assert rest.patches[-1] == ("ep1", {"networkVolumeIds": []})
    assert api.volumes() == []


def test_a_failed_attach_leaves_no_volume(rest: FakeRest) -> None:
    rest.fail_patch = True
    with pytest.raises(RunPodApiError, match="PATCH /endpoints/ep1: HTTP 400"):
        RunPodApi(base_url=rest.url).volume_up("ep1", 10, "XX-1", "x")
    assert rest.volumes == {}
    with pytest.raises(RunPodApiError, match="HTTP 401"):
        RunPodApi(api_key="wrong", base_url=rest.url).volumes()


def test_the_command_line(rest: FakeRest, capsys: pytest.CaptureFixture[str]) -> None:
    up = [
        "runpod",
        "volume",
        "up",
        "--endpoint",
        "ep1",
        "--size",
        "150",
        "--datacenter",
        "EU-RO-1",
    ]
    assert main(up) == 0
    out = capsys.readouterr().out
    assert "volume vol1 (150 GB, EU-RO-1) attached to ep1" in out
    assert "storage: s3://vol1/  S3 API: https://s3api-eu-ro-1.runpod.io" in out
    assert main(["runpod", "volume", "list"]) == 0
    assert "vol1  ico-assets  150 GB  EU-RO-1" in capsys.readouterr().out
    assert (
        main(["runpod", "volume", "down", "--endpoint", "ep1", "--volume", "vol1"]) == 0
    )
    assert (
        main(["runpod", "volume", "down", "--endpoint", "ep1", "--volume", "vol1"]) == 1
    )
    assert "HTTP 404" in capsys.readouterr().err


def test_a_worker_reads_and_writes_its_volume_at_the_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount = tmp_path / "runpod-volume"  # what the worker sees of volume "vol1"
    (mount / "runs" / "r1" / "in").mkdir(parents=True)
    (mount / "runs" / "r1" / "in" / "a.txt").write_text("hello")
    monkeypatch.setenv("ICO_VOLUME_ROOT", str(mount))
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    spec = op("jobs.upper_file@1", out_dir={"$param": "out_dir"})
    handle = submit(
        spec, "s3://vol1/runs/r1/in/a.txt", executor=LocalExecutor(tmp_path / "jobs"),
        params={"out_dir": "$JOB/out"}, packages=[PACKAGE], sources=SOURCES,
        out="s3://vol1/runs/r1/out", volume="vol1",
    )  # fmt: skip
    outputs = cast(list[str], result(handle)["outputs"])

    assert outputs == ["s3://vol1/runs/r1/out/a.txt"]
    assert (mount / "runs" / "r1" / "out" / "a.txt").read_text() == "HELLO"
