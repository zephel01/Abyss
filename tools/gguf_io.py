"""tools/gguf_io.py - Abyss's own GGUF v3 reader and writer for the pack tools (no gguf-py, no ggml).

Reader: every key with its value type, every tensor's info and its bytes (a read-only memmap).
Writer: the file a tool builds key by key and tensor by tensor - laid out the way the tools' files have always been
laid out (they were written with llama.cpp's gguf-py before): the header, the keys in the order added (the
architecture first when one is given), the tensor infos with offsets padded to the alignment, padding up to the
alignment, then each tensor's bytes padded to the alignment.  test_gguf_io.py checks that the bytes equal gguf-py's
for the same calls when gguf-py is at hand (STRATA_GGUF_PY), and the pack tools' outputs were compared by SHA-256 on
real models (worklog 2026-11-16_no-llamacpp).

Also here: the row dequantizers of the plain types (F32, F16, BF16, Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0, Q2_K .. Q6_K)
in single-precision steps (each product and sum rounded on its own, as numpy float32 does), the round-to-nearest-even
BF16 conversion, and the GGUF type table.  The i-quants and the other formats have no Python dequantizer here: a tool
that needs one says so (dequantize raises NeedsLlamaCpp).
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x46554747  # "GGUF"
DEFAULT_ALIGNMENT = 32

# value types
U8, I8, U16, I16, U32, I32, F32V, BOOL, STRING, ARRAY, U64, I64, F64V = range(13)
_FMT = {U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i", F32V: "f", BOOL: "?", U64: "Q", I64: "q",
        F64V: "d"}
_NP = {U8: "<u1", I8: "<i1", U16: "<u2", I16: "<i2", U32: "<u4", I32: "<i4", F32V: "<f4", BOOL: "<u1",
       U64: "<u8", I64: "<i8", F64V: "<f8"}

# tensor types: id -> (name, values per block, bytes per block)
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
}
TYPE_ID = {v[0]: k for k, v in TYPES.items()}


class NeedsLlamaCpp(RuntimeError):
    """The step needs llama.cpp's gguf-py or ggml (trench-gguf's build, or STRATA_GGUF_PY / LIBGGML)."""


def type_id(t) -> int:
    return t if isinstance(t, int) else TYPE_ID[str(t).upper()]


def type_name(t: int) -> str:
    return TYPES[t][0] if t in TYPES else "type%d" % t


def row_bytes(t: int, ne0: int) -> int:
    _, b, s = TYPES[t]
    if ne0 % b:
        raise ValueError("%d values are not whole %s blocks" % (ne0, type_name(t)))
    return ne0 // b * s


def nbytes(t: int, shape) -> int:
    n = 1
    for d in shape[1:]:
        n *= int(d)
    return row_bytes(t, int(shape[0])) * n


# ------------------------------------------------------------------------------------------------ reader
class Field:
    __slots__ = ("key", "vtype", "etype", "value")

    def __init__(self, key, vtype, value, etype=None):
        self.key, self.vtype, self.value, self.etype = key, vtype, value, etype

    def __repr__(self):
        return "Field(%r, %d, %r)" % (self.key, self.vtype, self.value if self.vtype != ARRAY else "[...]")


class Tensor:
    __slots__ = ("name", "shape", "type", "offset", "nbytes", "data")

    def __init__(self, name, shape, t, offset):
        self.name, self.shape, self.type, self.offset = name, shape, t, offset
        self.nbytes = nbytes(t, shape) if t in TYPES else 0
        self.data = None

    @property
    def type_name(self) -> str:
        return type_name(self.type)

    def rows(self) -> np.ndarray:
        """The bytes as (rows, row bytes)."""
        return self.data.reshape(-1, row_bytes(self.type, int(self.shape[0])))


class Reader:
    """A GGUF v3 file: .fields (key -> Field, in file order), .tensors (in file order), .by_name, .alignment,
    .data_start.  Tensor bytes are memmapped (read only)."""

    def __init__(self, path, data: bool = True):
        self.path = Path(path)
        self.fields: dict[str, Field] = {}
        self.tensors: list[Tensor] = []
        with open(self.path, "rb") as f:
            magic, ver = struct.unpack("<II", f.read(8))
            if magic != MAGIC:
                raise ValueError("%s: not a GGUF file" % self.path)
            if ver != 3:
                raise ValueError("%s: GGUF v%d (v3 is read)" % (self.path, ver))
            n_t, n_kv = struct.unpack("<QQ", f.read(16))
            for _ in range(n_kv):
                k = self._str(f)
                (vt,) = struct.unpack("<I", f.read(4))
                if vt == ARRAY:
                    et, n = struct.unpack("<IQ", f.read(12))
                    if et == STRING:
                        v = [self._str(f) for _ in range(n)]
                    elif et == ARRAY:
                        raise ValueError("%s: nested arrays are not read" % k)
                    else:
                        v = np.frombuffer(f.read(n * np.dtype(_NP[et]).itemsize), dtype=_NP[et]).copy()
                    self.fields[k] = Field(k, vt, v, et)
                elif vt == STRING:
                    self.fields[k] = Field(k, vt, self._str(f))
                else:
                    fmt = "<" + _FMT[vt]
                    (v,) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
                    self.fields[k] = Field(k, vt, v)
            for _ in range(n_t):
                name = self._str(f)
                (nd,) = struct.unpack("<I", f.read(4))
                shape = list(struct.unpack("<%dQ" % nd, f.read(8 * nd)))
                t, off = struct.unpack("<IQ", f.read(12))
                self.tensors.append(Tensor(name, shape, t, off))
            al = self.fields.get("general.alignment")
            self.alignment = int(al.value) if al is not None and al.vtype in _FMT else DEFAULT_ALIGNMENT
            pos = f.tell()
        self.data_start = (pos + self.alignment - 1) // self.alignment * self.alignment
        self.by_name = {t.name: t for t in self.tensors}
        if data and self.tensors:
            mm = np.memmap(self.path, dtype=np.uint8, mode="r")
            for t in self.tensors:
                o = self.data_start + t.offset
                t.data = mm[o:o + t.nbytes]

    @staticmethod
    def _str(f) -> str:
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8")

    def get(self, key, default=None):
        f = self.fields.get(key)
        if f is None:
            return default
        return list(f.value) if f.vtype == ARRAY and f.etype != STRING else f.value


# ------------------------------------------------------------------------------------------------ writer
def _infer(v):
    if isinstance(v, bool):
        return BOOL
    if isinstance(v, int):
        return I32
    if isinstance(v, float):
        return F32V
    if isinstance(v, str):
        return STRING
    raise TypeError("no GGUF type for %r" % (v,))


class Writer:
    """Builds a GGUF file.  add_*() keys, add_tensor() tensors (their bytes are kept by reference until write())."""

    def __init__(self, path, arch: str | None = None, alignment: int = DEFAULT_ALIGNMENT):
        self.path = Path(path)
        self.alignment = alignment
        self.kv: dict[str, tuple[int, object, int | None]] = {}
        self.tensors: list[tuple[str, list[int], int, object]] = []
        if arch is not None:
            self.add_string("general.architecture", arch)

    # keys
    def add(self, key: str, vtype: int, value, etype: int | None = None) -> None:
        if key in self.kv:
            raise ValueError("duplicate key %s" % key)
        if vtype == ARRAY and etype is None:
            etype = _infer(value[0]) if len(value) else I32
        self.kv[key] = (vtype, value, etype)

    def add_field(self, f: Field) -> None:
        self.add(f.key, f.vtype, f.value, f.etype)

    def add_string(self, k, v):
        self.add(k, STRING, str(v))

    def add_uint32(self, k, v):
        self.add(k, U32, int(v))

    def add_int32(self, k, v):
        self.add(k, I32, int(v))

    def add_uint64(self, k, v):
        self.add(k, U64, int(v))

    def add_float32(self, k, v):
        self.add(k, F32V, float(v))

    def add_bool(self, k, v):
        self.add(k, BOOL, bool(v))

    def add_array(self, k, v, etype: int | None = None):
        self.add(k, ARRAY, list(v), etype)

    def add_alignment(self, a: int) -> None:
        """general.alignment as a u32 key, and the padding the writer uses."""
        self.alignment = int(a)
        self.add_uint32("general.alignment", int(a))

    # tensors
    def add_tensor(self, name: str, shape, t, data) -> None:
        """shape: GGUF order (ne0 first, the row length); t: a type id or name; data: bytes / a numpy array of
        exactly nbytes(t, shape) bytes, or a function giving them when write() gets there (computed then)."""
        t = type_id(t)
        shape = [int(x) for x in shape]
        n = nbytes(t, shape)
        got = n if callable(data) else data.nbytes if isinstance(data, np.ndarray) else len(data)
        if got != n:
            raise ValueError("%s: %d bytes for %s %s (%d expected)" % (name, got, type_name(t), shape, n))
        if any(x[0] == name for x in self.tensors):
            raise ValueError("duplicate tensor %s" % name)
        self.tensors.append((name, shape, t, data))

    def _val(self, vt, v) -> bytes:
        if vt == STRING:
            b = v.encode("utf-8")
            return struct.pack("<Q", len(b)) + b
        return struct.pack("<" + _FMT[vt], v)

    def header_bytes(self) -> bytes:
        out = bytearray(struct.pack("<IIQQ", MAGIC, 3, len(self.tensors), len(self.kv)))
        for k, (vt, v, et) in self.kv.items():
            out += self._val(STRING, k) + struct.pack("<I", vt)
            if vt == ARRAY:
                out += struct.pack("<IQ", et, len(v))
                if et == STRING:
                    for s in v:
                        out += self._val(STRING, s)
                else:
                    out += np.asarray(v).astype(_NP[et]).tobytes() if et != BOOL else \
                        np.asarray([bool(x) for x in v], dtype=np.uint8).tobytes()
            else:
                out += self._val(vt, v)
        off = 0
        for name, shape, t, _ in self.tensors:
            out += self._val(STRING, name) + struct.pack("<I", len(shape)) + struct.pack("<%dQ" % len(shape), *shape)
            out += struct.pack("<IQ", t, off)
            off += -(-nbytes(t, shape) // self.alignment) * self.alignment
        out += b"\0" * ((-len(out)) % self.alignment)
        return bytes(out)

    def write(self, progress=None) -> Path:
        with open(self.path, "wb") as f:
            f.write(self.header_bytes())
            for name, shape, t, data in self.tensors:
                d = data() if callable(data) else data
                b = memoryview(np.ascontiguousarray(d)).cast("B") if isinstance(d, np.ndarray) else memoryview(d)
                if len(b) != nbytes(t, shape):
                    raise ValueError("%s: %d bytes written for %d" % (name, len(b), nbytes(t, shape)))
                f.write(b)
                f.write(b"\0" * ((-len(b)) % self.alignment))
                if progress:
                    progress(name)
        return self.path


def copy_fields(r: Reader, w: Writer, skip=lambda key: False) -> None:
    """The reader's keys into the writer in file order: the architecture (the writer has it) and GGUF.* never, a
    general.alignment key as the writer's alignment (its padding, not just the key)."""
    for k, f in r.fields.items():
        if k == "general.architecture" or k.startswith("GGUF.") or skip(k):
            continue
        if k == "general.alignment":
            w.add_alignment(int(f.value))
            continue
        w.add_field(f)


# ------------------------------------------------------------------------------------------------ numbers
def _f16(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view("<f2").astype(np.float32)


def _blocks(raw: np.ndarray, t: int) -> np.ndarray:
    _, _, s = TYPES[t]
    return np.ascontiguousarray(raw).reshape(-1).view(np.uint8).reshape(-1, s)


def _k4_scales(sc: np.ndarray):
    """Q4_K / Q5_K: the 12 scale bytes -> 8 six-bit scales and 8 six-bit mins per block (as float32)."""
    a, b, c = sc[:, 0:4].astype(np.uint8), sc[:, 4:8].astype(np.uint8), sc[:, 8:12].astype(np.uint8)
    s = np.concatenate([a & 63, (c & 0xF) | ((a >> 6) << 4)], axis=1)
    m = np.concatenate([b & 63, (c >> 4) | ((b >> 6) << 4)], axis=1)
    return s.astype(np.float32), m.astype(np.float32)


def dequantize(raw, t) -> np.ndarray:
    """The values of a row-major block array as a flat float32 array (the plain types; NeedsLlamaCpp otherwise)."""
    t = type_id(t)
    raw = np.asarray(raw)
    if t == 0:
        return np.ascontiguousarray(raw).reshape(-1).view("<f4").copy()
    if t == 1:
        return np.ascontiguousarray(raw).reshape(-1).view("<f2").astype(np.float32)
    if t == 30:
        return (np.ascontiguousarray(raw).reshape(-1).view("<u2").astype(np.uint32) << 16).view(np.float32)
    f = np.float32
    if t in (2, 3, 6, 7):   # Q4_0, Q4_1, Q5_0, Q5_1
        b = _blocks(raw, t)
        has_m = t in (3, 7)
        d = _f16(b[:, 0:2])
        m = _f16(b[:, 2:4]) if has_m else None
        o = 4 if has_m else 2
        if t in (6, 7):
            qh = np.ascontiguousarray(b[:, o:o + 4]).view("<u4").reshape(-1, 1)
            o += 4
        qs = b[:, o:o + 16]
        lo = (qs & 0xF).astype(np.int32)
        hi = (qs >> 4).astype(np.int32)
        if t in (6, 7):
            j = np.arange(16, dtype=np.uint32)
            lo |= (((qh >> j) & 1) << 4).astype(np.int32)
            hi |= (((qh >> (j + 16)) & 1) << 4).astype(np.int32)
        q = np.concatenate([lo, hi], axis=1)
        if has_m:
            return (q.astype(f) * d + m).reshape(-1)
        return ((q - (8 if t == 2 else 16)).astype(f) * d).reshape(-1)
    if t == 8:
        b = _blocks(raw, t)
        return (b[:, 2:].view(np.int8).astype(f) * _f16(b[:, 0:2])).reshape(-1)
    if t == 10:   # Q2_K
        b = _blocks(raw, t)
        sc, qs = b[:, 0:16], b[:, 16:80]
        d, mn = _f16(b[:, 80:82]), _f16(b[:, 82:84])
        dl = d * (sc & 0xF).astype(f)                     # (n, 16)
        ml = mn * (sc >> 4).astype(f)
        out = np.empty((b.shape[0], 256), dtype=f)
        for half in range(2):
            q = qs[:, 32 * half:32 * half + 32]
            for k in range(4):
                v = ((q >> (2 * k)) & 3).astype(f)          # (n, 32)
                for part in range(2):
                    i = 8 * half + 2 * k + part
                    out[:, 128 * half + 32 * k + 16 * part:][:, :16] = \
                        dl[:, i:i + 1] * v[:, 16 * part:16 * part + 16] - ml[:, i:i + 1]
        return out.reshape(-1)
    if t == 11:   # Q3_K
        b = _blocks(raw, t)
        hm, qs, ps = b[:, 0:32], b[:, 32:96], b[:, 96:108]
        d = _f16(b[:, 108:110])
        j = np.arange(16)
        lo = np.where(j < 8, ps[:, j % 8] & 0xF, ps[:, j % 8] >> 4)
        hi = (ps[:, 8 + (j % 4)] >> (2 * (j // 4))) & 3
        sc = ((lo | (hi << 4)).astype(np.int32) - 32).astype(f)
        dl = d * sc
        out = np.empty((b.shape[0], 256), dtype=f)
        for half in range(2):
            q = qs[:, 32 * half:32 * half + 32]
            for k in range(4):
                bit = 1 << (4 * half + k)
                v = (((q >> (2 * k)) & 3).astype(np.int32) - np.where(hm & bit, 0, 4)).astype(f)
                for part in range(2):
                    i = 8 * half + 2 * k + part
                    out[:, 128 * half + 32 * k + 16 * part:][:, :16] = dl[:, i:i + 1] * v[:, 16 * part:16 * part + 16]
        return out.reshape(-1)
    if t in (12, 13):   # Q4_K, Q5_K
        b = _blocks(raw, t)
        d, mn = _f16(b[:, 0:2]), _f16(b[:, 2:4])
        s, m = _k4_scales(b[:, 4:16])
        qh = b[:, 16:48] if t == 13 else None
        qs = b[:, 48:176] if t == 13 else b[:, 16:144]
        out = np.empty((b.shape[0], 256), dtype=f)
        for j in range(4):
            q = qs[:, 32 * j:32 * j + 32]
            lo, hi = (q & 0xF).astype(np.int32), (q >> 4).astype(np.int32)
            if t == 13:
                lo += np.where(qh & (1 << (2 * j)), 16, 0)
                hi += np.where(qh & (2 << (2 * j)), 16, 0)
            d1, m1 = d * s[:, 2 * j:2 * j + 1], mn * m[:, 2 * j:2 * j + 1]
            d2, m2 = d * s[:, 2 * j + 1:2 * j + 2], mn * m[:, 2 * j + 1:2 * j + 2]
            out[:, 64 * j:64 * j + 32] = d1 * lo.astype(f) - m1
            out[:, 64 * j + 32:64 * j + 64] = d2 * hi.astype(f) - m2
        return out.reshape(-1)
    if t == 14:   # Q6_K
        b = _blocks(raw, t)
        ql, qh, sc = b[:, 0:128], b[:, 128:192], b[:, 192:208].view(np.int8)
        d = _f16(b[:, 208:210])
        out = np.empty((b.shape[0], 256), dtype=f)
        for half in range(2):
            L, H, S = ql[:, 64 * half:64 * half + 64], qh[:, 32 * half:32 * half + 32], sc[:, 8 * half:8 * half + 8]
            qq = [((L[:, 0:32] & 0xF) | (((H >> 0) & 3) << 4)), ((L[:, 32:64] & 0xF) | (((H >> 2) & 3) << 4)),
                  ((L[:, 0:32] >> 4) | (((H >> 4) & 3) << 4)), ((L[:, 32:64] >> 4) | (((H >> 6) & 3) << 4))]
            for k in range(4):
                v = (qq[k].astype(np.int32) - 32).astype(f)
                for p in range(2):   # 16 values share a scale
                    out[:, 128 * half + 32 * k + 16 * p:][:, :16] = (d * S[:, 2 * k + p:2 * k + p + 1].astype(f)) * \
                        v[:, 16 * p:16 * p + 16]
        return out.reshape(-1)
    raise NeedsLlamaCpp("dequantizing %s needs llama.cpp's gguf-py (STRATA_GGUF_PY)" % type_name(t))


def dequantize_any(raw, t) -> np.ndarray:
    """dequantize(), and for the other types llama.cpp's gguf-py when STRATA_GGUF_PY names it (dev checks only)."""
    try:
        return dequantize(raw, t)
    except NeedsLlamaCpp:
        p = os.environ.get("STRATA_GGUF_PY")
        if not p or not (Path(p) / "gguf").is_dir():
            raise
        import sys
        if p not in sys.path:
            sys.path.insert(0, p)
        from gguf import GGMLQuantizationType, quants
        _, _, s = TYPES[type_id(t)]
        return np.asarray(quants.dequantize(np.asarray(raw).reshape(-1, s), GGMLQuantizationType(type_id(t))),
                          dtype=np.float32).reshape(-1)


def to_bf16(values: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bits (uint16), round to nearest even.  Finite values only."""
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((u + (0x7FFF + ((u >> 16) & 1))) >> 16).astype(np.uint16)


def quantize(values, t) -> np.ndarray:
    """float32 values (rows of whole blocks) -> the bytes of F32 / F16 / BF16 / Q8_0 (Q8_0 as ggml's reference
    quantizer: d = max|x| / 127 stored fp16, codes x * (1 / d) rounded half away from zero).  Test fixtures and small
    tensors; the other types need ggml (ggml_lib)."""
    t = type_id(t)
    x = np.ascontiguousarray(values, dtype=np.float32)
    if t == 0:
        return x.reshape(-1).view(np.uint8).copy()
    if t == 1:
        return x.reshape(-1).astype("<f2").view(np.uint8)
    if t == 30:
        return to_bf16(x).view(np.uint8)
    if t == 8:
        b = x.reshape(-1, 32)
        d = np.abs(b).max(axis=1, keepdims=True) / np.float32(127)
        with np.errstate(divide="ignore"):
            inv = np.where(d > 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
        v = b * inv
        q = (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.int8)
        out = np.empty((b.shape[0], 34), dtype=np.uint8)
        out[:, :2] = d.astype("<f2").view(np.uint8)
        out[:, 2:] = q.view(np.uint8)
        return out.reshape(-1)
    raise NeedsLlamaCpp("quantizing to %s needs ggml (ggml_lib)" % type_name(t))


def ggml_lib(explicit: str = ""):
    """A shared library with ggml_quantize_chunk (llama.cpp's / trench-gguf's libggml-base), or NeedsLlamaCpp."""
    import ctypes
    import glob
    here = Path(__file__).resolve().parent
    cands = [explicit] if explicit else []
    if os.environ.get("LIBGGML"):
        cands.append(os.environ["LIBGGML"])
    for d in (here.parent / "build", here.parent / "build-hip", here.parent / "third_party" / "llama.cpp" / "build" / "bin",
              here.parent.parent / "trench-gguf" / "third_party" / "llama.cpp" / "build" / "bin"):
        for pat in ("libggml-base.so", "libggml-base.dylib", "ggml-base.dll"):
            cands += glob.glob(str(d / pat))
    for c in cands:
        try:
            lib = ctypes.CDLL(c)
            lib.ggml_quantize_chunk.restype = ctypes.c_size_t
            lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                                ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
            return lib
        except (OSError, AttributeError):
            continue
    raise NeedsLlamaCpp("this step quantizes with ggml_quantize_chunk: it needs a libggml-base shared library "
                        "(trench-gguf's or llama.cpp's build) - pass it with LIBGGML=<path> (or the tool's --ggml-lib)")
