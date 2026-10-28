"""Export the trained weights to one compact file the browser can load.

Every matrix is quantised to 8-bit integers with one float scale per row
("per-row absmax"): w ≈ q · scale, q ∈ [-127, 127]. That divides the size by 4
compared with float32 while keeping the error tiny, because each row gets its
own scale. Vectors (biases, LayerNorm) stay float32: they are small and
sensitive.

File layout (little-endian):
  b"GLNC" | u32 header length | header JSON (utf-8) | zero padding to 4 bytes | tensor data
The header lists every tensor's name, shape, dtype and byte offsets.

Pure NumPy on purpose: the same code is unit-tested without PyTorch.
"""

from __future__ import annotations

import json
import struct

import numpy as np

MAGIC = b"GLNC"


def quantize_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = w.astype(np.float32)
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def export(state: dict[str, np.ndarray], config: dict, path: str, meta: dict | None = None) -> dict:
    """`state`: parameter name → NumPy array (the tied lm_head is skipped)."""
    entries, blobs, offset = [], [], 0

    def add(b: bytes) -> int:
        nonlocal offset
        start = offset
        blobs.append(b)
        offset += len(b)
        pad = (-offset) % 4
        if pad:
            blobs.append(b"\0" * pad)
            offset += pad
        return start

    for name in sorted(state):
        if name == "lm_head.weight":
            continue
        arr = np.ascontiguousarray(state[name])
        if arr.ndim == 2:
            q, s = quantize_rows(arr)
            entries.append(
                {"name": name, "shape": list(arr.shape), "dtype": "int8", "offset": add(q.tobytes()), "scales": add(s.astype("<f4").tobytes())}
            )
        else:
            entries.append({"name": name, "shape": list(arr.shape), "dtype": "f32", "offset": add(arr.astype("<f4").tobytes())})

    header = json.dumps({"format": "glance-v1", "config": config, "meta": meta or {}, "tensors": entries}).encode("utf-8")
    pad = (-(8 + len(header))) % 4
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header) + pad))
        f.write(header + b" " * pad)  # spaces keep the JSON valid
        for b in blobs:
            f.write(b)
    return {"bytes": 8 + len(header) + pad + offset, "tensors": len(entries)}


def load(path: str) -> tuple[dict, dict[str, np.ndarray]]:
    """Reads an exported file back (dequantised), for tests and the NumPy reference."""
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:4] == MAGIC, "not a Glance weights file"
    (hlen,) = struct.unpack("<I", raw[4:8])
    header = json.loads(raw[8 : 8 + hlen].decode("utf-8"))
    base = 8 + hlen
    out = {}
    for t in header["tensors"]:
        n = int(np.prod(t["shape"]))
        if t["dtype"] == "int8":
            q = np.frombuffer(raw, dtype=np.int8, count=n, offset=base + t["offset"]).reshape(t["shape"])
            s = np.frombuffer(raw, dtype="<f4", count=t["shape"][0], offset=base + t["scales"])
            out[t["name"]] = q.astype(np.float32) * s[:, None]
        else:
            out[t["name"]] = np.frombuffer(raw, dtype="<f4", count=n, offset=base + t["offset"]).reshape(t["shape"]).copy()
    return header, out
