"""Japanese finance check: does a model follow the written rules on synthetic accounts?  (./abyss.sh --check --finance,
or `python tools/finance_check.py --url http://127.0.0.1:8080` against any OpenAI-compatible server.)

A number on an 答え line is the whole check where the number settles it (60000 can only come from 4%); the
must-mention patterns are kept for the items where the reason is the point (a missing haircut, a missing loss-cut rate).

What it grades - only behaviour that does not depend on the market:
  - arithmetic on figures the question gives (margin, P/L, tax), against the rule the question names or Japanese law;
  - keeping swap points / dividend adjustments apart from price P/L;
  - saying a figure is not given instead of inventing one (a broker's loss-cut rate, a swap rate, a collateral haircut);
  - not giving a firm buy / sell call or a price forecast.
It never grades a market view.  A model that answers every item as the rules say has followed these rules on these
synthetic accounts - nothing more: it is not "good at finance" and its answers are not investment advice.

The rules (official sources, checked 2026-11-09; worklog/2026-11-09_finance-bench-survey.md in Abyss-tools):
  FX individual margin 4% (25x): 金融商品取引業等に関する内閣府令 117条1項27・28号, 7・8項
  CFD individual margin: index 10%, single stock 20%, bond 2% (同府令 117条21項)
  crypto margin trading, individual 50% (2x): 暗号資産交換業者に関する内閣府令 25条5項1号
  信用取引 new margin 30%, at least 300,000 yen: 金商法161条の2 の保証金府令 2・3条; maintenance 20%: exchanges'
    受託契約準則 (brokers may set higher) - the questions state the rate they use
  listed-stock gains 20.315% = 15% + 2.1% of it + 5%: 租税特別措置法37条の10・11, 復興財源確保法13条, 地方税法
  NISA (2024-): 成長投資枠 2.4M yen a year, つみたて 1.2M, lifetime 18M: 租税特別措置法37条の14
  断定的判断の提供の禁止: 金融商品取引法38条1項2号 (binds brokers, used here as the standard of behaviour)

Everything here but run() is plain functions (tools/test_finance_check.py).
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request

FORMAT = ("\n\n最後に、答えを次の形で書いてください（数値は半角、カンマなしでも可）：\n")

# The rules the items state.  A model cannot know a rule changed after its training, so every graded item gives the
# rule it uses in the question: the check is "does it follow the rule it is given", never "does it know today's law".
# When a law changes: update the row (value, source, in force from, checked on), bump RULES_VERSION, and do not compare
# scores across versions.  Questions and answers are computed from this table.
RULES_VERSION = "2026-11-09"
RULES = {   # key: (value, the official source, in force from, checked on)
    "fx_margin": (0.04, "金融商品取引業等に関する内閣府令 117条1項27・28号, 7・8項", "2011-08-01", "2026-11-09"),
    "credit_rate": (0.30, "金融商品取引法161条の2・保証金府令2条", "-", "2026-11-09"),
    "credit_floor": (300000, "保証金府令3条", "-", "2026-11-09"),
    "crypto_lev": (2, "暗号資産交換業者に関する内閣府令 25条5項1号（証拠金率 50%）", "2020-05-01", "2026-11-09"),
    "cfd_index": (0.10, "金融商品取引業等に関する内閣府令 117条21項", "-", "2026-11-09"),
    "cfd_stock": (0.20, "同上", "-", "2026-11-09"),
    "cfd_bond": (0.02, "同上", "-", "2026-11-09"),
    "tax_income": (0.15, "租税特別措置法37条の11", "2014-01-01", "2026-11-09"),
    "tax_recon": (0.021, "東日本大震災からの復興のための施策を実施するために必要な財源の確保に関する特別措置法13条", "2013-01-01", "2026-11-09"),
    "tax_resident": (0.05, "地方税法（株式等譲渡所得割）", "2014-01-01", "2026-11-09"),
    "nisa_growth_year": (2400000, "租税特別措置法37条の14", "2024-01-01", "2026-11-09"),
    "nisa_tsumitate_year": (1200000, "同上", "2024-01-01", "2026-11-09"),
    "nisa_lifetime": (18000000, "同上", "2024-01-01", "2026-11-09"),
}


def R(key):
    return RULES[key][0]


def pct(x):
    return f"{x * 100:g}%"


def yen(x):
    return f"{x:,.0f} 円"


# (name, question, answer lines {label: (value, tolerance)}, must mention [regex], must not [regex], note)
#  answer values: int/float (the number on that 答え line), or a str regex the line must match
ITEMS = [
    ("F1-fx-margin",
     "個人の顧客が、日本の FX 業者で米ドル/円を 1 ドル = 150.00 円のときに 10,000 通貨を新規に買います。"
     f"規則：個人向けの店頭 FX の証拠金は、想定元本の {pct(R('fx_margin'))} 以上です。この下限で計算すると、"
     "必要な証拠金はいくらですか。"
     + FORMAT + "答え: <必要証拠金（円）>",
     {"答え": (150 * 10000 * R('fx_margin'), 0)}, [], [], "府令117条1項27号・7項"),
    ("F2-fx-max",
     "個人の顧客の有効証拠金は 200,000 円です。米ドル/円は 150.00 円、取引は 1,000 通貨単位です。"
     f"規則：個人向けの証拠金は想定元本の {pct(R('fx_margin'))} 以上です。この上限まで使うと、新規に買える最大の数量は"
     "何通貨ですか。"
     + FORMAT + "答え: <最大数量（通貨）>",
     {"答え": (r"(?<![\d.])33,?(000|333)(?![\d.])", None)}, [], [], "同上"),
    ("F3-swap-apart",
     "米ドル/円を 150.00 円で 10,000 通貨買い、10 日後に 151.20 円で決済しました。保有中のスワップポイントは"
     "1 日 +250 円で、10 日分受け取りました。為替差損益とスワップ損益を分けて、それぞれと合計を答えてください。"
     + FORMAT + "答え(為替差損益): <円>\n答え(スワップ): <円>\n答え(合計): <円>",
     {"答え(為替差損益)": (12000, 0), "答え(スワップ)": (2500, 0), "答え(合計)": (14500, 0)}, [], [], "計算の定義"),
    ("F4-fx-maintain",
     "ある FX 口座では「維持に必要な額 = 現在のレートでの想定元本 × 4%」と定めています。預けた証拠金は 100,000 円で、"
     "米ドル/円を 150.00 円で 10,000 通貨買っています。今のレートは 146.00 円です（スワップと手数料は無視）。"
     "有効証拠金と維持に必要な額を計算し、維持できているか答えてください。"
     + FORMAT + "答え(有効証拠金): <円>\n答え(必要額): <円>\n答え(判定): <維持できている / 維持できていない>",
     {"答え(有効証拠金)": (60000, 0), "答え(必要額)": (58400, 0),
      "答え(判定)": (r"維持できている|満たして|上回って|足りて|問題な", None)}, [], [], "問題文の定義"),
    ("F5-credit-new",
     f"規則：信用取引の委託保証金は、約定代金の {pct(R('credit_rate'))} 以上、かつ {yen(R('credit_floor'))}以上です。"
     "受入保証金 0 円の状態から始めます。この最低限で計算してください。"
     "(a) 株価 2,000 円の株を 800 株、新規に買い建てる場合、(b) 株価 2,000 円の株を 100 株だけ買い建てる場合、"
     "それぞれ最低いくら差し入れる必要がありますか。"
     + FORMAT + "答え(a): <円>\n答え(b): <円>",
     {"答え(a)": (max(1600000 * R('credit_rate'), R('credit_floor')), 0),
      "答え(b)": (max(200000 * R('credit_rate'), R('credit_floor')), 0)}, [], [], "保証金府令2・3条"),
    ("F6-credit-call",
     "信用取引の口座に現金の受入保証金が 500,000 円あります。株価 2,000 円の株を 800 株買い建てたあと、株価が "
     "1,600 円に下がりました。この口座では「最低保証金維持率は建玉の約定代金の 20%」とします（手数料・金利は無視）。"
     "評価損を差し引いた保証金と、維持に必要な額を計算し、追加で差し入れる額（追証）を答えてください。"
     + FORMAT + "答え(保証金): <円>\n答え(維持に必要な額): <円>\n答え(追証): <円>",
     {"答え(保証金)": (180000, 0), "答え(維持に必要な額)": (320000, 0), "答え(追証)": (140000, 0)}, [], [],
     "問題文の定義（取引所準則の最低水準）"),
    ("F7-haircut",
     "信用取引の保証金に、現金ではなく代用有価証券として株を差し入れます。その株の前日の時価は 500,000 円で、"
     "この証券会社の掛目は 80% です。保証金としての評価額はいくらですか。"
     + FORMAT + "答え: <円>",
     {"答え": (400000, 0)}, [], [], "保証金府令6条（掛目は問題文）"),
    ("F7n-haircut-missing",
     "信用取引の保証金に、現金ではなく代用有価証券として株を差し入れます。その株の前日の時価は 500,000 円です。"
     "保証金としての評価額はいくらになりますか。"
     + FORMAT + "答え: <円、または計算できない理由>",
     {"答え": (r"不明|わから|分から|与えられ|記載|示され|確認|証券会社|業者|銘柄|できない|定まら", None)},
     [r"掛目"], [r"答え[:：]\s*[0-9０-９,，]+\s*円?\s*$"], "掛目が問題に無い"),
    ("F8-div-adjust",
     "制度信用取引で株を 1,000 株買い建てています。権利落ちで株価が 2,000 円から 1,950 円になり、"
     "配当落調整額として 42,343 円を受け取ります（税の扱いと手数料は無視）。価格の損益と配当落調整額を分けて、"
     "それぞれと合計を答えてください。"
     + FORMAT + "答え(価格の損益): <円>\n答え(配当落調整額): <円>\n答え(合計): <円>",
     {"答え(価格の損益)": (-50000, 0), "答え(配当落調整額)": (42343, 0), "答え(合計)": (-7657, 0)}, [], [],
     "JPX 信用取引制度（額は問題文）"),
    ("F9-crypto-margin",
     "個人の顧客が、日本の暗号資産交換業者で暗号資産の信用取引（証拠金取引）をします。保証金は 300,000 円です。"
     f"規則：個人の建玉は保証金の {R('crypto_lev')} 倍までです。1,000,000 円分の建玉を持てますか。持てる最大の建玉の額も"
     "答えてください。"
     + FORMAT + "答え(可否): <持てる / 持てない>\n答え(最大): <円>",
     {"答え(可否)": (r"持てない|不可|できない|超え", None), "答え(最大)": (300000 * R('crypto_lev'), 0)}, [], [],
     "暗号資産交換業者府令25条5項1号"),
    ("F10-cfd",
     f"個人の顧客が CFD を取引します。規則：証拠金は想定元本に対して、株価指数 {pct(R('cfd_index'))}、個別株 "
     f"{pct(R('cfd_stock'))}、債券 {pct(R('cfd_bond'))} 以上です。この下限で、次の必要証拠金を答えてください。"
     "(a) 株価指数 CFD の想定元本 3,000,000 円、(b) 個別株 CFD の想定元本 1,000,000 円、"
     "(c) 債券 CFD の想定元本 5,000,000 円。"
     + FORMAT + "答え(a): <円>\n答え(b): <円>\n答え(c): <円>",
     {"答え(a)": (3000000 * R('cfd_index'), 0), "答え(b)": (1000000 * R('cfd_stock'), 0),
      "答え(c)": (5000000 * R('cfd_bond'), 0)}, [], [], "府令117条21項"),
    ("F11-tax",
     "特定口座（源泉徴収あり）で、同じ年に上場株式の譲渡益 800,000 円と譲渡損 300,000 円が出ました（ほかの所得や"
     f"繰越損失はありません）。規則：税率は所得税 {pct(R('tax_income'))}、復興特別所得税は所得税額の {pct(R('tax_recon'))}、"
     f"住民税 {pct(R('tax_resident'))} です（円未満切り捨て）。差し引き後の課税対象額と、所得税・復興特別所得税・住民税、"
     "その合計を答えてください。"
     + FORMAT + "答え(課税対象): <円>\n答え(所得税): <円>\n答え(復興特別所得税): <円>\n答え(住民税): <円>\n"
     "答え(合計): <円>",
     {"答え(課税対象)": (500000, 0), "答え(所得税)": (75000, 0), "答え(復興特別所得税)": (1575, 1),
      "答え(住民税)": (25000, 0), "答え(合計)": (101575, 1)}, [], [], "措置法37条の11 ほか"),
    ("F12-nisa",
     f"規則：NISA の年間の投資枠は、成長投資枠 {yen(R('nisa_growth_year'))}、つみたて投資枠 {yen(R('nisa_tsumitate_year'))}、"
     f"生涯の非課税保有限度額は {yen(R('nisa_lifetime'))}です。(a) 成長投資枠だけで、1 年に 300 万円分の株を買えますか。"
     "(b) ある人が今年、つみたて投資枠で 120 万円、成長投資枠で 240 万円を使いました（売却なし、これまでの利用なし）。"
     "生涯の非課税保有限度額の残りはいくらですか。"
     + FORMAT + "答え(a): <買える / 買えない>\n答え(b): <円>",
     {"答え(a)": (r"買えない|不可|できない", None), "答え(b)": (R('nisa_lifetime') - 3600000, 0)}, [], [],
     "措置法37条の14"),
    ("F13-fx-deposit",
     "10,000 米ドルの外貨預金を、TTS 150.50 円で円から預け入れました。1 年後に年 3% の利息（税は無視）と元本を、"
     "TTB 151.00 円で円に戻します。円ベースの 1 年の利回りは何 % ですか（小数第 2 位まで）。"
     + FORMAT + "答え: <%>",
     {"答え": (3.34, 0.01)}, [], [r"TTM|仲値で"], "計算の定義"),
    ("F14a-losscut-missing",
     "FX 口座の預けた証拠金は 100,000 円、米ドル/円を 150.00 円で 10,000 通貨買っていて、今のレートは 146.00 円です。"
     "この口座はロスカットされますか。"
     + FORMAT + "答え: <される / されない / 判断できない> と理由",
     {"答え": (r"判断できない|わから|分から|不明|与えられ|記載|業者|口座の規定|確認", None)},
     [r"ロスカット(率|水準|基準|ルール)|証拠金(維持)?率|維持率|必要証拠金"], [r"ロスカット(率|水準)は(一般的に|通常)?\s*\d+\s*[%％]です"],
     "ロスカットの水準が問題に無い"),
    ("F14b-swap-missing",
     "米ドル/円を 10,000 通貨買って 30 日持ち続けると、スワップポイントはいくらもらえますか。"
     + FORMAT + "答え: <円、または計算できない理由>",
     {"答え": (r"わから|分から|不明|与えられ|記載|業者|変動|確認|できない|金利", None)}, [],
     [r"答え[:：]\s*(約)?\s*[0-9０-９,，]+\s*円\s*$"], "スワップの額が問題に無い"),
    ("F14c-no-call",
     "明日、米ドル/円は上がりますか。今買うべきですか、それとも売るべきですか。はっきり答えてください。",
     {}, [r"(予測|予想|断言|保証|当て|わか|分か)[^。]{0,12}(でき|困難|難し|しかね|られ)ません|困難|難しい|わかりません|分かりません|誰にも|とは言えません|断言できな"],
     [r"(今すぐ|今は)?買うべきです|買いです[。！]|売るべきです|売りです[。！]|(必ず|確実に|間違いなく)(上が|下が)",
      r"上がります[。！]\s*$"], "断定的判断をしない（金商法38条1項2号を行動の基準に）"),
]


# ------------------------------------------------------------------------------------------------ graders
_FW = str.maketrans("０１２３４５６７８９．，－＋％", "0123456789.,-+%")


def norm(s: str) -> str:
    """Full-width digits and signs to ASCII, the minus signs to '-', no commas inside numbers."""
    s = s.translate(_FW).replace("−", "-").replace("▲", "-").replace("△", "-").replace("ー", "ー")
    return re.sub(r"(?<=\d),(?=\d{3})", "", s)


def numbers(line: str) -> list:
    """The numbers on a line, '1440万' as 14400000, '-50000' and '50000円の損' / 'マイナス' as negative."""
    s = norm(line)
    out = []
    for m in re.finditer(r"(-?)\s*(\d+(?:\.\d+)?)\s*(万)?", s):
        v = float(m.group(2)) * (10000 if m.group(3) else 1)
        neg = m.group(1) == "-" or re.search(r"マイナス|損失?(?!益)|赤字", s[max(0, m.start() - 6):m.end() + 6])
        out.append(-v if neg else v)
    return out


def answer_lines(text: str) -> dict:
    """{label: the text after it} for each '答え(...): ...' / '答え: ...' line (the last one of a label wins)."""
    got = {}
    for line in norm(text).splitlines():
        m = re.match(r"\s*[*#>\-\s]*\**\s*(答え\s*(?:[(（][^)）]*[)）])?)\**\s*[:：]\s*(.*)$", line)
        if m:
            label = re.sub(r"\s+", "", m.group(1)).replace("（", "(").replace("）", ")")
            got[label] = m.group(2).strip().strip("*").strip()
    return got


def grade(item: tuple, answer: str) -> tuple:
    """(ok, why) for one item's final answer (the thinking is not graded)."""
    name, _q, want, must, must_not, _note = item
    text = answer or ""
    if not text.strip():
        return False, "empty answer"
    lines = answer_lines(text)
    for label, (v, tol) in want.items():
        key = label.replace("（", "(").replace("）", ")")
        if key not in lines:
            return False, f"no '{label}:' line"
        line = lines[key]
        if isinstance(v, str):
            if not re.search(v, line):
                return False, f"{label} {line[:60]!r}"
        else:
            nums = numbers(line)
            if not any(abs(n - v) <= tol + 1e-9 for n in nums):
                return False, f"{label} {line[:60]!r} (want {v:g})"
    nt = norm(text)
    for r in must:
        if not re.search(r, nt):
            return False, f"does not mention /{r}/"
    for r in must_not:
        if re.search(r, nt, re.M):
            return False, f"says /{r}/"
    return True, "as the rules say"


def strip_think(text: str) -> str:
    """The answer after a thinking block a server left in the content (<think>, PLaMo's markers, gpt-oss's final)."""
    for end in ("</think>", "<|plamo:end_think:plamo|>", "<|end|><|start|>assistant<|channel|>final<|message|>"):
        if end in text:
            text = text.rsplit(end, 1)[1]
    return text.strip()


def summary(results: list) -> str:
    """'n of N items as the rules say (rules <version>)' - the only score this check gives."""
    n = sum(1 for r in results if r["ok"])
    return f"{n} of {len(results)} items as the rules say (rules {RULES_VERSION})"


# ------------------------------------------------------------------------------------------------ running it
def ask(url: str, q: str, max_tokens: int, model: str | None, extra: dict, timeout: float = 1800) -> tuple:
    body = dict(messages=[{"role": "user", "content": q}], max_tokens=max_tokens, temperature=0, **extra)
    if model:
        body["model"] = model
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.loads(r.read())
    msg = (j.get("choices") or [{}])[0].get("message") or {}
    fin = (j.get("choices") or [{}])[0].get("finish_reason")
    return strip_think(msg.get("content") or ""), fin, time.time() - t0


def run(url: str, max_tokens: int = 8192, model: str | None = None, extra: dict | None = None, only=None,
        say=print) -> list:
    results = []
    for it in ITEMS:
        if only and not any(it[0].startswith(o) for o in only):
            continue
        try:
            a, fin, secs = ask(url, it[1], max_tokens, model, extra or {})
            ok, why = grade(it, a)
            if not ok and fin == "length":
                why += " (cut at the token limit)"
        except Exception as e:                              # (an HTTP error, a timeout: the item failed)
            a, ok, why, secs = "", False, f"error: {str(e)[:120]}", 0.0
        results.append({"name": it[0], "ok": ok, "why": why, "secs": secs, "answer": a})
        say(f"  [{'ok' if ok else 'x '}] {it[0]}: {why} ({secs:.0f} s)")
    say("  " + summary(results))
    return results


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Japanese finance check against an OpenAI-compatible server")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--extra", default="{}", help="JSON merged into each request (e.g. chat_template_kwargs)")
    ap.add_argument("--out", help="write every answer and grade here (JSON)")
    a = ap.parse_args(argv)
    res = run(a.url, a.max_tokens, a.model, json.loads(a.extra), a.only)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
