"""Minimal Prometheus metrics (text exposition format), no dependency."""
from __future__ import annotations

import threading
from typing import Dict, Iterable, List, Sequence, Tuple

Labels = Tuple[Tuple[str, str], ...]


def _fmt_labels(labels: Labels) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"' for k, v in labels)
    return "{" + inner + "}"


class Counter:
    kind = "counter"

    def __init__(self, name: str, help: str) -> None:
        self.name, self.help = name, help
        self.values: Dict[Labels, float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self.values[key] = self.values.get(key, 0.0) + amount

    def samples(self) -> Iterable[Tuple[str, Labels, float]]:
        with self._lock:
            return [(self.name, k, v) for k, v in self.values.items()]


class Gauge(Counter):
    kind = "gauge"

    def set(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self.values[key] = float(value)


class Histogram:
    kind = "histogram"

    def __init__(self, name: str, help: str, buckets: Sequence[float]) -> None:
        self.name, self.help = name, help
        self.buckets = sorted(buckets)
        self.data: Dict[Labels, List[float]] = {}  # per label set: bucket counts + [sum, count]
        self._lock = threading.Lock()

    def observe(self, x: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            d = self.data.setdefault(key, [0.0] * (len(self.buckets) + 2))
            for i, b in enumerate(self.buckets):
                if x <= b:
                    d[i] += 1
            d[-2] += x
            d[-1] += 1

    def samples(self):
        out = []
        with self._lock:
            for key, d in self.data.items():
                for i, b in enumerate(self.buckets):
                    out.append((self.name + "_bucket", key + (("le", repr(float(b))),), d[i]))
                out.append((self.name + "_bucket", key + (("le", "+Inf"),), d[-1]))
                out.append((self.name + "_sum", key, d[-2]))
                out.append((self.name + "_count", key, d[-1]))
        return out


class Registry:
    def __init__(self) -> None:
        self.metrics: List = []
        self.collectors: List = []  # callables returning [(metric_name, kind, help, labels, value)]

    def counter(self, name: str, help: str) -> Counter:
        m = Counter(name, help)
        self.metrics.append(m)
        return m

    def gauge(self, name: str, help: str) -> Gauge:
        m = Gauge(name, help)
        self.metrics.append(m)
        return m

    def histogram(self, name: str, help: str, buckets: Sequence[float]) -> Histogram:
        m = Histogram(name, help, buckets)
        self.metrics.append(m)
        return m

    def render(self) -> str:
        lines: List[str] = []
        for m in self.metrics:
            lines.append(f"# HELP {m.name} {m.help}")
            lines.append(f"# TYPE {m.name} {m.kind}")
            for name, labels, v in m.samples():
                lines.append(f"{name}{_fmt_labels(labels)} {v:g}")
        families: Dict[str, Tuple[str, str, List[str]]] = {}
        for collect in self.collectors:
            for name, kind, help, labels, v in collect():
                fam = families.setdefault(name, (kind, help, []))
                fam[2].append(f"{name}{_fmt_labels(tuple(sorted(labels.items())))} {v:g}")
        for name, (kind, help, samples) in families.items():
            lines.append(f"# HELP {name} {help}")
            lines.append(f"# TYPE {name} {kind}")
            lines.extend(samples)
        return "\n".join(lines) + "\n"
