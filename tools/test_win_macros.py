"""Sources must not use the names Windows' headers define as empty macros (minwindef.h:
IN, OUT, OPTIONAL) as identifiers: the CI builds Windows with CUDA only, so a HIP-only block that did (glm_prefill.cu's
`float* OUT`, v1.3.186-188) broke every Windows AMD build unseen (issue #49)."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMPTY_MACROS = ("IN", "OUT", "OPTIONAL")
WORD = re.compile(r"\b(" + "|".join(EMPTY_MACROS) + r")\b")


def code_only(text: str) -> str:
    """The text with comments and string / char literals blanked (line numbers kept)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i)); i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if ch == "\n" else " " for ch in text[i:j])); i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(" " * (j - i)); i = j
        else:
            out.append(c); i += 1
    return "".join(out)


class WindowsEmptyMacros(unittest.TestCase):
    def test_no_identifier_named_like_an_empty_windows_macro(self):
        bad = []
        for p in sorted([*ROOT.glob("src/**/*"), *ROOT.glob("include/**/*"), *ROOT.glob("tests/**/*")]):
            if p.suffix not in (".cu", ".cpp", ".cc", ".hpp", ".h", ".cuh") or not p.is_file():
                continue
            # every source: a header can bring <windows.h> in (glm_fast_state.hpp does)
            for ln, line in enumerate(code_only(p.read_text(errors="replace")).splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                m = WORD.search(line)
                if m:
                    bad.append(f"{p.relative_to(ROOT)}:{ln}: {m.group(1)}")
        self.assertEqual(bad, [], "names Windows' headers define as empty macros (rename them):\n" + "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
