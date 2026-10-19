"""s3:// storage: files and directories with moto (no network), and a job
staged through a local moto server."""

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from moto import mock_aws
from moto.server import ThreadedMotoServer

from ico.remote import LocalExecutor, result, submit
from ico.remote.s3 import S3Storage
from tests.apriori.ico.remote.test_jobs import PACKAGE, SOURCES, op

BUCKET = "ico-test"
boto3: Any = importlib.import_module("boto3")  # untyped


@pytest.fixture
def aws(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def test_files_and_directories(aws: None, tmp_path: Path) -> None:
    with mock_aws():
        client: Any = boto3.client("s3")
        client.create_bucket(Bucket=BUCKET)
        storage = S3Storage(client)
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "dir" / "sub").mkdir(parents=True)
        (tmp_path / "dir" / "x.txt").write_text("x")
        (tmp_path / "dir" / "sub" / "y.txt").write_text("y")

        storage.put(tmp_path / "a.txt", f"s3://{BUCKET}/in/a.txt")
        storage.put(tmp_path / "dir", f"s3://{BUCKET}/in/dir")
        storage.get(f"s3://{BUCKET}/in/a.txt", tmp_path / "got" / "a.txt")
        storage.get(f"s3://{BUCKET}/in/dir", tmp_path / "got" / "dir")
        storage.get(f"s3://{BUCKET}/in/dir/", tmp_path / "got" / "slash")

    assert (tmp_path / "got" / "a.txt").read_text() == "a"
    for copy in ("dir", "slash"):
        assert (tmp_path / "got" / copy / "x.txt").read_text() == "x"
        assert (tmp_path / "got" / copy / "sub" / "y.txt").read_text() == "y"


@pytest.fixture
def moto_server(aws: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", endpoint)  # the runner inherits it
    try:
        yield endpoint
    finally:
        server.stop()


def test_a_job_stages_through_s3(moto_server: str, tmp_path: Path) -> None:
    client: Any = boto3.client("s3", endpoint_url=moto_server)
    client.create_bucket(Bucket=BUCKET)
    client.put_object(Bucket=BUCKET, Key="in/a.txt", Body=b"hello")

    handle = submit(
        op("jobs.upper_file@1", out_dir={"$param": "out_dir"}),
        f"s3://{BUCKET}/in/a.txt",
        executor=LocalExecutor(tmp_path),
        params={"out_dir": "$JOB/out"},
        packages=[PACKAGE],
        sources=SOURCES,
        out=f"s3://{BUCKET}/jobs/1/",
    )
    outputs = result(handle)["outputs"]

    assert outputs == [f"s3://{BUCKET}/jobs/1/a.txt"]
    body = client.get_object(Bucket=BUCKET, Key="jobs/1/a.txt")["Body"].read()
    assert body == b"HELLO"
