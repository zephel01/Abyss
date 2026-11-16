"""tools/token_classes.py - every token of a vocabulary sorted by script, and the language profiles on top.

WHAT IT IS FOR.  A model asked in Japanese sometimes writes a Chinese word in the middle of the answer ("误って",
"这样的") - the token was simply the most likely one.  Banning those tokens at the sampler (the engine sets their
logits to -inf) stops it without touching the weights.  Which tokens to ban is a property of the TOKENIZER, so it is
computed once per vocabulary: each token gets a class word (uint16 bit set, below), and a language PROFILE is just the
set of class bits an answer may use - "ja-en" allows Latin + Kana + the Han characters of JIS X 0213, "zh-en" Latin +
Han, "ko-en" Latin + Hangul, "en" Latin.  A token that uses any other class is banned.  A new profile is one line in
PROFILES; it needs no per-tokenizer list.

THE CLASSES.  A token's bytes are decoded as UTF-8; each character adds its class bit (Unicode character names give
the script - `unicodedata`, no table of our own):
  LATIN, KANA (hiragana, katakana, halfwidth katakana, the prolonged sound mark), HAN_JIS (a Han character that
  JIS X 0213 encodes: Python's `euc_jis_2004` codec, which covers both planes of JIS X 0213), HAN_NONJIS (any other
  Han character), HANGUL, CYRILLIC, GREEK, ARABIC, HEBREW, THAI, DEVANAGARI, OTHER (any other letter script:
  Bopomofo, Armenian, Bengali, ...).  Digits, punctuation, symbols, whitespace and emoji add no bit: they are allowed
  in every profile, as is code.
  HAN_SIMP / HAN_TRAD are HINTS on Han characters: in GB2312 but not Big5 (simplified-leaning), in Big5 but not
  GB2312 (traditional-leaning).  Both codecs come with Python.  They are not a simplified/traditional dictionary
  (GB2312 lacks rare characters a simplified text still uses), so the profiles that read them are marked experimental.
  PARTIAL: the token's bytes end inside a UTF-8 character (byte-level BPE splits a 3-byte CJK character into pieces).
  Such a token is banned only when EVERY character it can begin is banned in the profile (the possible completions are
  enumerated) - a piece that also starts an allowed character stays.  Bytes that continue a character begun by the
  previous token are unknowable here and count as allowed.
  SPECIAL: a control / user-defined token (<|im_end|>, <think>, <tool_call>, ...): never banned, whatever it spells.

THE JAPANESE-SAFE LIST.  Han characters outside JIS X 0213 that correct Japanese text on disk was found to use
(JA_SAFE below, measured with --side-effects): allowed in ja-en.  Common Chinese-only WORDS spelled with shared Han
(啤酒, 皮肤 - characters Japanese also uses) are NOT banned: the Chinese text on disk (~140K characters) is too small
to tell a Chinese-only word from a rare Japanese one, so word-level bans are left out.

THE SHIPPED TABLES.  data/token_classes/<name>.u16.gz: one little-endian uint16 per token id, gzip'd; index.json maps
each table to the fingerprint of the vocabulary it was built from (the non-special tokens' strings, so V4 and V4.1,
which differ only in special tokens, share one).  The server loads the table whose fingerprint matches its tokenizer
and builds it in memory otherwise (about two seconds).

    python tools/token_classes.py build --tokenizer <pack>/tokenizer --name qwen35
    python tools/token_classes.py stats --tokenizer <pack>/tokenizer
    python tools/token_classes.py side-effects --tokenizer <pack>/tokenizer --profile ja-en <text files...>
"""
from __future__ import annotations

import argparse
import array
import gzip
import hashlib
import json
import pathlib
import sys
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "token_classes"

LATIN, KANA, HAN_JIS, HAN_NONJIS, HANGUL, CYRILLIC, GREEK, ARABIC = (1 << i for i in range(8))
HEBREW, THAI, DEVANAGARI, OTHER, HAN_SIMP, HAN_TRAD, PARTIAL, SPECIAL = (1 << i for i in range(8, 16))
CLASS_NAMES = {LATIN: "latin", KANA: "kana", HAN_JIS: "han_jis", HAN_NONJIS: "han_nonjis", HANGUL: "hangul",
               CYRILLIC: "cyrillic", GREEK: "greek", ARABIC: "arabic", HEBREW: "hebrew", THAI: "thai",
               DEVANAGARI: "devanagari", OTHER: "other", HAN_SIMP: "han_simp_hint", HAN_TRAD: "han_trad_hint",
               PARTIAL: "partial_utf8", SPECIAL: "special"}
HAN = HAN_JIS | HAN_NONJIS
HINTS = HAN_SIMP | HAN_TRAD
SCRIPTS = 0x0FFF   # the script bits (a character has exactly one, or none)

# Greek is allowed everywhere: α, β, π, Δ, μ are the notation of every language's technical text (JIS X 0208 itself
# carries the Greek alphabet), and banning them broke formulas.
PROFILES: dict[str, dict] = {
    "en":         {"allow": LATIN | GREEK, "about": "English: Latin script only"},
    "ja-en":      {"allow": LATIN | GREEK | KANA | HAN_JIS | HINTS, "ja_safe": True,
                   "about": "Japanese + English: Latin, Kana and the Han characters of JIS X 0213"},
    "zh-en":      {"allow": LATIN | GREEK | HAN | HINTS, "about": "Chinese + English: Latin and Han (no Kana, no Hangul)"},
    "zh-Hans-en": {"allow": LATIN | GREEK | HAN | HAN_SIMP, "experimental": True,
                   "about": "simplified Chinese + English: zh-en without the Big5-only (traditional-leaning) Han"},
    "zh-Hant-en": {"allow": LATIN | GREEK | HAN | HAN_TRAD, "experimental": True,
                   "about": "traditional Chinese + English: zh-en without the GB2312-only (simplified-leaning) Han"},
    "ko-en":      {"allow": LATIN | GREEK | HANGUL, "about": "Korean + English: Latin and Hangul (no Han, no Kana)"},
}
# Han characters outside JIS X 0213 that correct Japanese text on disk uses (side-effects run over the Japanese
# corpora listed in the worklog): none were found, so the list is empty; a character added here is allowed in ja-en.
JA_SAFE: frozenset[str] = frozenset()

_SCRIPT_WORDS = {"LATIN": LATIN, "CYRILLIC": CYRILLIC, "GREEK": GREEK, "COPTIC": GREEK, "ARABIC": ARABIC,
                 "HEBREW": HEBREW, "THAI": THAI, "DEVANAGARI": DEVANAGARI, "HANGUL": HANGUL, "HIRAGANA": KANA,
                 "KATAKANA": KANA, "KATAKANA-HIRAGANA": KANA}
_char_cache: dict[str, int] = {}


def _encodes(c: str, codec: str) -> bool:
    try:
        c.encode(codec)
        return True
    except UnicodeEncodeError:
        return False


def char_class(c: str) -> int:
    """The class bits of one character (0: neutral - digits, punctuation, symbols, spaces, emoji)."""
    b = _char_cache.get(c)
    if b is not None:
        return b
    o = ord(c)
    name = unicodedata.name(c, "")
    cat = unicodedata.category(c)
    if name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")) or o == 0x3007:
        b = HAN_JIS if _encodes(c, "euc_jis_2004") else HAN_NONJIS
        gb, big5 = _encodes(c, "gb2312"), _encodes(c, "big5")
        if gb and not big5:
            b |= HAN_SIMP
        elif big5 and not gb:
            b |= HAN_TRAD
    elif o < 0x80:
        b = LATIN if c.isalpha() else 0
    elif cat[0] not in "LM":
        b = 0
    else:
        words = name.split(" ")
        first = words[0]
        if first in ("FULLWIDTH", "HALFWIDTH", "SMALL", "CIRCLED", "PARENTHESIZED") and len(words) > 1:
            first = words[1]
        if first in _SCRIPT_WORDS:
            b = _SCRIPT_WORDS[first]
        elif o < 0x0250 or "LATIN" in words:
            b = LATIN
        elif cat[0] == "M" and first in ("COMBINING", "VARIATION"):
            b = 0   # a combining accent / variation selector: belongs to whatever it follows
        elif first in ("MODIFIER", "MATHEMATICAL", "SUPERSCRIPT", "SUBSCRIPT", "IDEOGRAPHIC", "VERTICAL", "CARON",
                       "DOUBLE-STRUCK"):
            b = 0
        else:
            b = OTHER
    _char_cache[c] = b
    return b


def _completions(prefix: bytes):
    """Every character whose UTF-8 encoding starts with `prefix` (an incomplete lead sequence)."""
    lead = prefix[0]
    n = 2 if lead >> 5 == 0b110 else 3 if lead >> 4 == 0b1110 else 4 if lead >> 3 == 0b11110 else 0
    if n == 0 or len(prefix) >= n:
        return
    missing = n - len(prefix)
    for tail in range(64 ** missing):
        b = bytearray(prefix)
        for k in range(missing - 1, -1, -1):
            b.append(0x80 | ((tail >> (6 * k)) & 63))
        try:
            yield bytes(b).decode("utf-8")
        except UnicodeDecodeError:
            pass


_frag_cache: dict[bytes, frozenset[int]] = {}


def fragment_classes(prefix: bytes) -> frozenset[int]:
    """The distinct class words of the characters an incomplete UTF-8 prefix can begin (4-byte prefixes of one
    byte - 262,144 completions, emoji and rare planes - are summarized by their plane's first characters' classes:
    those planes hold no script a profile bans except Han extensions, which stay allowed as a piece)."""
    got = _frag_cache.get(prefix)
    if got is None:
        if len(prefix) == 1 and prefix[0] >= 0xF0:
            got = frozenset({0})   # a lone 4-byte lead: emoji and supplementary planes, never banned as a piece
        else:
            got = frozenset(char_class(c) for c in _completions(prefix)) or frozenset({0})
        _frag_cache[prefix] = got
    return got


def split_utf8(raw: bytes) -> tuple[int, str, bytes]:
    """(leading continuation bytes, the decodable middle, a trailing incomplete prefix).  A malformed middle (an
    invalid byte inside) is decoded with replacement; its stray bytes add no class."""
    i = 0
    while i < len(raw) and raw[i] & 0xC0 == 0x80:
        i += 1
    j = len(raw)
    k = j - 1
    while k >= i and raw[k] & 0xC0 == 0x80:
        k -= 1
    if k >= i:
        lead = raw[k]
        n = 2 if lead >> 5 == 0b110 else 3 if lead >> 4 == 0b1110 else 4 if lead >> 3 == 0b11110 else 1
        if j - k < n:
            j = k
    return i, raw[i:j].decode("utf-8", errors="replace").replace("�", ""), raw[j:]


def token_class(raw: bytes, special: bool = False) -> int:
    if special:
        return SPECIAL
    lead, mid, tail = split_utf8(raw)
    b = 0
    for c in mid:
        b |= char_class(c)
    if tail:
        b |= PARTIAL
    return b


# ------------------------------------------------------------------ vocabularies
def is_special(tok, i: int) -> bool:
    tt = getattr(tok, "token_types", None)
    return bool(tt) and i < len(tt) and tt[i] in (3, 4)


def fingerprint(tok) -> str:
    """sha256 over the non-special tokens (id and string): what the class table depends on."""
    h = hashlib.sha256()
    for i, t in enumerate(tok.tokens):
        if not is_special(tok, i):
            h.update(b"%d\x00" % i)
            h.update(t.encode("utf-8", "surrogatepass"))
            h.update(b"\x01")
    h.update(b"n=%d" % len(tok.tokens))
    return h.hexdigest()[:24]


def build_table(tok) -> array.array:
    out = array.array("H", bytes(2 * len(tok.tokens)))
    for i in range(len(tok.tokens)):
        sp = is_special(tok, i)
        try:
            raw = b"" if sp else tok.token_bytes(i)
        except (KeyError, IndexError):
            raw = tok.tokens[i].encode("utf-8")
        out[i] = token_class(raw, sp)
    return out


def save_table(table: array.array, name: str, fp: str, vocab: int, out_dir: pathlib.Path = DATA) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    a = array.array("H", table)
    if sys.byteorder != "little":
        a.byteswap()
    p = out_dir / f"{name}.u16.gz"
    with gzip.GzipFile(p, "wb", mtime=0) as f:   # mtime 0: the same bytes every build
        f.write(a.tobytes())
    idx_p = out_dir / "index.json"
    idx = json.loads(idx_p.read_text()) if idx_p.exists() else {}
    idx[name] = {"fingerprint": fp, "vocab": vocab, "file": p.name}
    idx_p.write_text(json.dumps(dict(sorted(idx.items())), indent=1) + "\n")
    return p


def load_table(tok, data_dir: pathlib.Path = DATA) -> tuple[array.array, str]:
    """The shipped table for this vocabulary, else one built now: (table, where it came from)."""
    fp = fingerprint(tok)
    idx_p = data_dir / "index.json"
    if idx_p.exists():
        for name, ent in json.loads(idx_p.read_text()).items():
            if ent.get("fingerprint") == fp and ent.get("vocab") == len(tok.tokens):
                a = array.array("H")
                a.frombytes(gzip.decompress((data_dir / ent["file"]).read_bytes()))
                if sys.byteorder != "little":
                    a.byteswap()
                if len(a) == len(tok.tokens):
                    # special tokens from THIS tokenizer's types (V4 and V4.1 share a table, not their specials)
                    for i in range(len(a)):
                        if is_special(tok, i):
                            a[i] = SPECIAL
                        elif a[i] == SPECIAL:
                            a[i] = token_class(tok.token_bytes(i))
                    return a, name
    return build_table(tok), "built"


# ------------------------------------------------------------------ profiles -> banned ids
def char_allowed(c: str, prof: dict) -> bool:
    b = char_class(c)
    if b & ~prof["allow"] == 0:
        return True
    return bool(prof.get("ja_safe")) and c in JA_SAFE


def banned_ids(tok, table, profile: str, keep=()) -> list[int]:
    """The ids a profile bans: a token with a class bit outside the profile (checked per character where the
    Japanese-safe list could allow it), or a PARTIAL piece none of whose possible characters is allowed.  Specials
    and `keep` (the request's stop / think / tool ids) are never banned."""
    prof = PROFILES[profile]
    allow = prof["allow"]
    keep = set(keep)
    out = []
    for i, b in enumerate(table):
        if b & SPECIAL or i in keep or b == 0:
            continue
        if b & ~(allow | PARTIAL) == 0 and not b & PARTIAL:
            continue
        raw = tok.token_bytes(i)
        _, mid, tail = split_utf8(raw)
        bad = any(not char_allowed(c, prof) for c in mid)
        if not bad and tail:
            bad = all(cl & ~allow != 0 for cl in fragment_classes(tail))
        if bad:
            out.append(i)
    return out


def text_chars_banned(text: str, profile: str) -> list[str]:
    """The characters of `text` a profile bans (what a leak check counts)."""
    prof = PROFILES[profile]
    return [c for c in text if not char_allowed(c, prof)]


# ------------------------------------------------------------------ the command line
def _load_tok(path):
    sys.path.insert(0, str(HERE))
    import strata_tokenizer as ST
    return ST.Tokenizer.from_dir(path)


def _read_texts(paths) -> list[tuple[str, str]]:
    out = []
    for p in paths:
        p = pathlib.Path(p)
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            for r in pq.read_table(p).to_pylist():
                out.append((str(p), "\n".join(str(v) for v in r.values())))
        elif p.suffix == ".json":
            def walk(x):
                if isinstance(x, str):
                    yield x
                elif isinstance(x, dict):
                    for v in x.values():
                        yield from walk(v)
                elif isinstance(x, list):
                    for v in x:
                        yield from walk(v)
            out.append((str(p), "\n".join(walk(json.loads(p.read_text(encoding="utf-8"))))))
        else:
            out.append((str(p), p.read_text(encoding="utf-8", errors="replace")))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write data/token_classes/<name>.u16.gz for a pack's tokenizer")
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--out", default=str(DATA))
    s = sub.add_parser("stats", help="class counts and the banned-id count of every profile")
    s.add_argument("--tokenizer", required=True)
    e = sub.add_parser("side-effects", help="how often correct text needs a token the profile bans")
    e.add_argument("--tokenizer", required=True)
    e.add_argument("--profile", required=True, choices=sorted(PROFILES))
    e.add_argument("--examples", type=int, default=20)
    e.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    tok = _load_tok(a.tokenizer)
    if a.cmd == "build":
        t = build_table(tok)
        p = save_table(t, a.name, fingerprint(tok), len(tok.tokens), pathlib.Path(a.out))
        print(f"{p}: {len(t)} tokens, fingerprint {fingerprint(tok)}")
        return 0
    table, src = load_table(tok)
    if a.cmd == "stats":
        print(f"vocab {len(table)} (table: {src})")
        for bit, nm in CLASS_NAMES.items():
            print(f"  {nm:14s} {sum(1 for x in table if x & bit)}")
        print(f"  {'neutral':14s} {sum(1 for x in table if x == 0)}")
        for pr in PROFILES:
            print(f"  profile {pr:11s} bans {len(banned_ids(tok, table, pr))}")
        return 0
    ban = set(banned_ids(tok, table, a.profile))
    n_tok = n_hit = n_lines = n_lines_hit = 0
    examples: dict[str, int] = {}
    for name, text in _read_texts(a.files):
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            ids = tok.encode(line)
            hit = [i for i in ids if i in ban]
            n_tok += len(ids)
            n_hit += len(hit)
            n_lines += 1
            if hit:
                n_lines_hit += 1
                for i in hit:
                    piece = tok.decode([i])
                    pos = line.find(piece) if piece.strip() else -1
                    ctx = line[max(0, pos - 6):pos + len(piece) + 6] if pos >= 0 else line[:24]
                    key = f"{piece!r} in …{ctx}…"
                    examples[key] = examples.get(key, 0) + 1
    print(json.dumps({"profile": a.profile, "banned_ids": len(ban), "tokens": n_tok, "banned_tokens": n_hit,
                      "token_rate": n_hit / max(1, n_tok), "lines": n_lines, "lines_with_banned": n_lines_hit,
                      "line_rate": n_lines_hit / max(1, n_lines),
                      "examples": sorted(examples.items(), key=lambda kv: -kv[1])[:a.examples]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
