"""RunPod's REST API (``rest.runpod.io/v1``) for what jobs need around an
endpoint: network volumes, created for an order and deleted after it.

A volume holds the assets (weights) the workers read and the inputs and
outputs of runs; its S3 API (``https://s3api-<datacenter>.runpod.io``,
bucket = the volume id) is how files get there from outside, while a
worker sees the same files under ``/runpod-volume``. Attaching a volume
binds the endpoint's workers to the volume's datacenter. The key comes
from ``RUNPOD_API_KEY``; standard library only.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import cast

from ico.remote.job import JsonObject
from ico.spec.model import JsonValue

REST_API = "https://rest.runpod.io/v1"
USER_AGENT = "ico-remote"
"""RunPod's REST API (behind Cloudflare) refuses the default
``Python-urllib`` user agent: HTTP 403, error code 1010."""
RETRIES = 5


class RunPodApiError(RuntimeError):
    """A REST call failed (the message has the method, path and status)."""


@dataclass(frozen=True)
class Volume:
    """A network volume."""

    id: str
    name: str
    size: int
    """GB."""
    datacenter: str

    @property
    def s3_endpoint(self) -> str:
        """The S3 API of the volume's datacenter."""
        return f"https://s3api-{self.datacenter.lower()}.runpod.io"

    def uri(self, path: str = "") -> str:
        """``s3://<id>/<path>``: files on the volume through its S3 API
        (``/runpod-volume/<path>`` on a worker)."""
        return f"s3://{self.id}/{path.lstrip('/')}"

    def s3_env(self) -> dict[str, str]:
        """Environment for boto3 to reach the volume (the keys are RunPod
        S3 API keys in ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``)."""
        return {"AWS_ENDPOINT_URL_S3": self.s3_endpoint, "AWS_REGION": self.datacenter}

    @classmethod
    def from_dict(cls, data: JsonObject) -> Volume:
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or ""),
            size=int(cast(int, data.get("size") or 0)),
            datacenter=str(data.get("dataCenterId") or ""),
        )


@dataclass(frozen=True)
class RunPodApi:
    """The REST API with a key (``RUNPOD_API_KEY`` by default)."""

    api_key: str | None = None
    base_url: str = REST_API

    def volumes(self) -> list[Volume]:
        found = cast(list[JsonValue], self.call("GET", "/networkvolumes"))
        return [Volume.from_dict(cast(JsonObject, v)) for v in found]

    def create_volume(self, name: str, size: int, datacenter: str) -> Volume:
        body: JsonObject = {"name": name, "size": size, "dataCenterId": datacenter}
        return Volume.from_dict(
            cast(JsonObject, self.call("POST", "/networkvolumes", body))
        )

    def delete_volume(self, volume_id: str) -> None:
        self.call("DELETE", f"/networkvolumes/{volume_id}")

    def endpoint(self, endpoint_id: str) -> JsonObject:
        return cast(JsonObject, self.call("GET", f"/endpoints/{endpoint_id}"))

    def attach(self, endpoint_id: str, volume: Volume) -> None:
        """Attach a volume to an endpoint; its workers then run in the
        volume's datacenter (a rolling release of the endpoint)."""
        body: JsonObject = {
            "networkVolumeId": volume.id,
            "dataCenterIds": [volume.datacenter],
        }
        self.call("PATCH", f"/endpoints/{endpoint_id}", body)

    def detach(self, endpoint_id: str) -> None:
        """Detach the endpoint's volumes (before deleting them)."""
        body: JsonObject = {"networkVolumeIds": []}
        self.call("PATCH", f"/endpoints/{endpoint_id}", body)

    def volume_up(
        self, endpoint_id: str, size: int, datacenter: str, name: str
    ) -> Volume:
        """Create a volume and attach it to the endpoint (an order starts)."""
        volume = self.create_volume(name, size, datacenter)
        try:
            self.attach(endpoint_id, volume)
        except RunPodApiError:
            self.delete_volume(volume.id)  # never leave a paid volume behind
            raise
        return volume

    def volume_down(self, endpoint_id: str, volume_id: str) -> None:
        """Detach the volume from the endpoint and delete it with its files
        (an order ends)."""
        self.detach(endpoint_id)
        self.delete_volume(volume_id)

    def call(self, method: str, path: str, body: JsonObject | None = None) -> JsonValue:
        key = self.api_key or os.environ.get("RUNPOD_API_KEY", "")
        url = f"{self.base_url.rstrip('/')}{path}"
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(RETRIES):
            request = urllib.request.Request(url, data=data, method=method)
            request.add_header("authorization", f"Bearer {key}")
            request.add_header("content-type", "application/json")
            request.add_header("user-agent", USER_AGENT)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    text = response.read()
                    return cast(JsonValue, json.loads(text)) if text.strip() else None
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == RETRIES - 1:
                    detail = e.read().decode(errors="replace")[:500]
                    raise RunPodApiError(
                        f"RunPod {method} {path}: HTTP {e.code} {detail}"
                    ) from None
            except urllib.error.URLError as e:
                raise RunPodApiError(f"RunPod {method} {path}: {e.reason}") from None
            time.sleep(min(2**attempt, 30) + random.uniform(0, 1))
        raise AssertionError("unreachable")
