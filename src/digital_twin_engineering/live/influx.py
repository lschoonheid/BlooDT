"""Writing the twin's outputs to InfluxDB 2 (line protocol over HTTP, no client
library), or to memory for tests and in-process runs."""

import math
import urllib.parse
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Point:
    measurement: str
    tags: dict[str, str]
    fields: dict[str, float | bool]
    t: float  # epoch s


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")


def line(p: Point) -> str | None:
    """The point in line protocol (ns precision), None if it has no finite field."""
    fields = []
    for k, v in p.fields.items():
        if isinstance(v, bool):
            fields.append(f"{_escape(k)}={'true' if v else 'false'}")
        elif v is not None and math.isfinite(v):
            fields.append(f"{_escape(k)}={float(v)!r}")
    if not fields:
        return None
    tags = "".join(f",{_escape(k)}={_escape(str(v))}" for k, v in sorted(p.tags.items()))
    return f"{_escape(p.measurement)}{tags} {','.join(fields)} {round(p.t * 1e9)}"


@dataclass
class MemorySink:
    points: list[Point] = field(default_factory=list)

    def write(self, points: list[Point]) -> None:
        self.points.extend(points)

    def flush(self) -> None:
        pass


class InfluxSink:
    """Buffers points and posts them in batches of `batch` lines."""

    def __init__(self, url: str, token: str, org: str, bucket: str, batch: int = 5000):
        query = urllib.parse.urlencode({"org": org, "bucket": bucket, "precision": "ns"})
        self.endpoint = f"{url.rstrip('/')}/api/v2/write?{query}"
        self.headers = {"Authorization": f"Token {token}", "Content-Type": "text/plain; charset=utf-8"}
        self.batch = batch
        self.lines: list[str] = []

    def write(self, points: list[Point]) -> None:
        self.lines.extend(s for s in map(line, points) if s)
        if len(self.lines) >= self.batch:
            self.flush()

    def flush(self) -> None:
        if not self.lines:
            return
        body = "\n".join(self.lines).encode()
        self.lines = []
        request = urllib.request.Request(self.endpoint, body, self.headers, method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
