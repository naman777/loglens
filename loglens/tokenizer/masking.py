"""Ordered regex masking that turns raw log messages into template-like text.

Order matters: specific IDs (UUID, block IDs) before HEX, IPs before paths/numbers, durations before
plain numbers. Bucketed durations and small error/status codes are deliberately kept, since they
carry signal (e.g. ``<DURATION:10s+>``, ``status 503``).
"""
from __future__ import annotations

import re

DURATION_BUCKETS = ["<10ms", "<100ms", "<1s", "<10s", "10s+"]

MASK_TOKENS = [
    "<TS>", "<UUID>", "<IP>", "<PORT>", "<URL>", "<PATH>", "<HEX>", "<BLK>", "<ID>", "<NUM>",
    "<VER>", "<LOC>", *(f"<DURATION:{b}>" for b in DURATION_BUCKETS),
]

LEVELS = ["TRACE", "DEBUG", "INFO", "NOTICE", "WARN", "ERROR", "FATAL", "CRITICAL", "UNK"]
_LEVEL_ALIAS = {"WARNING": "WARN", "ERR": "ERROR", "SEVERE": "ERROR", "CRIT": "CRITICAL",
                "EMERG": "FATAL", "ALERT": "FATAL", "FAILURE": "ERROR", "FATAL": "FATAL"}


def level_token(level: str | None) -> str:
    lv = (level or "UNK").upper()
    lv = _LEVEL_ALIAS.get(lv, lv)
    return f"<LVL:{lv if lv in LEVELS else 'UNK'}>"


def service_token(service: str | None, known: set[str] | frozenset[str] | None = None) -> str:
    if not service or (known is not None and service not in known):
        return "<SVC:other>"
    return f"<SVC:{service}>"


LEVEL_TOKENS = [f"<LVL:{lv}>" for lv in LEVELS]

_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_BLK = re.compile(r"\bblk_-?\d+\b")
_TS = re.compile(
    r"\b\d{4}[-/.]\d{2}[-/.]\d{2}(?:[ T-]\d{2}[:.]\d{2}[:.]\d{2}(?:[.,]\d+)?Z?)?"  # 2005-06-03 15:42:50.3
    r"|\b\d{2}/\d{2}/\d{2,4} \d{2}:\d{2}:\d{2}\b"
    r"|\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r" +\d+ \d{2}:\d{2}:\d{2}(?: \d{4})?"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +\d+ \d{2}:\d{2}:\d{2}\b"
    r"|(?<![\w.:])\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?![\w:])"
)
_IP4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?::(\d{1,5}))?(?![\w.]\d)")
_IP4MAPPED = re.compile(r"(?<![\w:])::ffff:(?:\d{1,3}\.){3}\d{1,3}(?::(\d{1,5}))?", re.I)
_RELPATH = re.compile(r"(?<![\w<>:/.\-])[\w.\-@%+=~$*]*[\w][\w.\-@%+=~$*]*(?:/[\w.\-@%+=~$*]+){2,}/?")
_IP6 = re.compile(
    r"(?<![\w:])(?=[0-9a-fA-F:]*::|(?:[0-9a-fA-F]{1,4}:){7})(?=:*[0-9a-fA-F])[0-9a-fA-F:]{2,39}(?![\w:])"
)
_URL = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s,;\"')\]>]+")
_PATH = re.compile(r"(?<![\w<>:/.\-])(?:\.{0,2}/)(?:[\w.\-@%+=~$*]+/)*[\w.\-@%+=~$*]+/?")
_VER = re.compile(r"\bv?\d+\.\d+\.\d+(?:[.\-][0-9A-Za-z]+)*\b")
_HEX0X = re.compile(r"\b0[xX][0-9a-fA-F]+\b")
_HEXRUN = re.compile(r"(?<![\w<])(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{8,}(?![\w>])", re.I)
_LOC = re.compile(r"\bR\d{2}(?:-M\d)?(?:-[NL][0-9A-F]+)?(?:-[CI]:J\d{2}-U\d{2})?(?![\w])")
_ID = re.compile(
    r"\b(?:application|container|job|attempt|task|appattempt|blockmgr|broadcast|rdd|stage|shuffle)"
    r"_[\w\-]*\d[\w\-]*"
    r"|(?<![\w<])(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{16,}(?![\w>])"
)
_DUR = re.compile(
    r"(?<![\w.<])(\d+(?:\.\d+)?)\s?(us|µs|ms|msec|msecs|milliseconds?|s|secs?|seconds?|min|mins|minutes?)\b(?!\w)",
    re.I,
)
_KEEP_CODE = re.compile(
    r"(?i)\b(?:status|code|error|err|errno|exit|rc|returned|response|http|signal)\b[\s:=(]*$"
)
_NUM = re.compile(r"(?<![\w.<])-?\d+(?:\.\d+)?(?![\w>])")
_DOT_NUM = re.compile(r"(?<=[A-Za-z_])\.\d+(?![\w])")  # core.21370 -> core.<NUM>
_US_NUM = re.compile(r"(?<=[A-Za-z])_\d+(?![\w])")  # step_12 -> step_<NUM>
_STEM_NUM = re.compile(r"(?<![\w.<])([A-Za-z]{2,})\d{1,4}(?![\w.])")  # bglio12, storage3, sda1
_KEEP_STEM = frozenset({"ssh", "sha", "md", "utf", "ipv", "http", "tls", "sslv", "tlsv", "ec", "ed",
                        "aes", "rsa", "sv", "py", "gb", "mb", "kb", "tb"})
_HTTP_CTX = re.compile(r'(?:<URL>|")\s*$')
_MULTISPACE = re.compile(r"\s+")


def _dur_token(m: re.Match) -> str:
    v = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("us", "µs"):
        sec = v / 1e6
    elif unit.startswith("ms") or unit.startswith("milli"):
        sec = v / 1e3
    elif unit.startswith("min"):
        sec = v * 60
    else:
        sec = v
    if sec < 0.01:
        b = "<10ms"
    elif sec < 0.1:
        b = "<100ms"
    elif sec < 1:
        b = "<1s"
    elif sec < 10:
        b = "<10s"
    else:
        b = "10s+"
    return f"<DURATION:{b}>"


def _ip4(m: re.Match) -> str:
    return "<IP>:<PORT>" if m.group(1) else "<IP>"


def _path(m: re.Match) -> str:
    p = m.group(0)
    last = p.rstrip("/").rsplit("/", 1)[-1]
    if "/" in p.strip("/") and 0 < len(last) <= 16 and "." in last and not any(c.isdigit() for c in last):
        return f"<PATH>/{last}"
    return "<PATH>"


def _num(m: re.Match) -> str:
    s = m.string
    tok = m.group(0)
    if "." not in tok and not tok.startswith("-") and len(tok) <= 3:
        before = s[max(0, m.start() - 16): m.start()]
        if _KEEP_CODE.search(before):
            return tok
        if len(tok) == 3 and tok[0] in "12345" and _HTTP_CTX.search(before):
            return tok  # HTTP status after the request line
    return "<NUM>"


def mask_reference(message: str) -> str:
    """Unguarded reference implementation (all rules, always); tests assert mask() == this."""
    s = message
    s = _UUID.sub("<UUID>", s)
    s = _BLK.sub("<BLK>", s)
    s = _URL.sub("<URL>", s)
    s = _TS.sub("<TS>", s)
    s = _IP4MAPPED.sub(_ip4, s)
    s = _IP4.sub(_ip4, s)
    s = _IP6.sub("<IP>", s)
    s = _LOC.sub("<LOC>", s)
    s = _PATH.sub(_path, s)
    s = _RELPATH.sub("<PATH>", s)
    s = _VER.sub("<VER>", s)
    s = _HEX0X.sub("<HEX>", s)
    s = _HEXRUN.sub("<HEX>", s)
    s = _ID.sub("<ID>", s)
    s = _DUR.sub(_dur_token, s)
    s = _NUM.sub(_num, s)
    s = _DOT_NUM.sub(".<NUM>", s)
    s = _US_NUM.sub("_<NUM>", s)
    s = _STEM_NUM.sub(lambda m: m.group(0) if m.group(1).lower() in _KEEP_STEM else m.group(1) + "<NUM>", s)
    return _MULTISPACE.sub(" ", s).strip()


_HASDIGIT = re.compile(r"\d")


def mask(message: str) -> str:
    """Same result as ``mask_reference`` but skips rules whose trigger characters are absent."""
    s = message
    hd = _HASDIGIT.search(s) is not None
    if "-" in s:
        s = _UUID.sub("<UUID>", s)
    if "blk_" in s:
        s = _BLK.sub("<BLK>", s)
    if "://" in s:
        s = _URL.sub("<URL>", s)
    if hd:
        s = _TS.sub("<TS>", s)
        if "::" in s:
            s = _IP4MAPPED.sub(_ip4, s)
        if "." in s:
            s = _IP4.sub(_ip4, s)
    if ":" in s:
        s = _IP6.sub("<IP>", s)
    if hd and "R" in s:
        s = _LOC.sub("<LOC>", s)
    if "/" in s:
        s = _PATH.sub(_path, s)
        s = _RELPATH.sub("<PATH>", s)
    if hd:
        if "." in s:
            s = _VER.sub("<VER>", s)
        if "0x" in s or "0X" in s:
            s = _HEX0X.sub("<HEX>", s)
        s = _HEXRUN.sub("<HEX>", s)
        s = _ID.sub("<ID>", s)
        s = _DUR.sub(_dur_token, s)
        s = _NUM.sub(_num, s)
        if "." in s:
            s = _DOT_NUM.sub(".<NUM>", s)
        if "_" in s:
            s = _US_NUM.sub("_<NUM>", s)
        s = _STEM_NUM.sub(lambda m: m.group(0) if m.group(1).lower() in _KEEP_STEM else m.group(1) + "<NUM>", s)
    return _MULTISPACE.sub(" ", s).strip()


def encode_line(message: str, level: str | None = None, service: str | None = None,
                known_services: set[str] | frozenset[str] | None = None) -> str:
    """Structural prefix tokens + masked message, ready for the BPE tokenizer."""
    return f"{level_token(level)} {service_token(service, known_services)} {mask(message)}"
