"""tools/finance_check.py's graders (no server needed)."""
import unittest

import finance_check as F

IT = {i[0]: i for i in F.ITEMS}


class TestGrade(unittest.TestCase):
    def test_rules_in_every_knowledge_item(self):
        for name in ("F1", "F2", "F5", "F9", "F10", "F11", "F12"):
            q = next(i[1] for i in F.ITEMS if i[0].startswith(name + "-"))
            self.assertIn("規則", q, name)

    def test_answers_from_the_table(self):
        self.assertEqual(IT["F1-fx-margin"][2]["答え"][0], 60000)
        self.assertEqual(IT["F5-credit-new"][2]["答え(b)"][0], 300000)
        self.assertEqual(IT["F12-nisa"][2]["答え(b)"][0], 14400000)

    def test_answer_lines_and_widths(self):
        a = "計算...\n**答え(為替差損益):** 12,000円\n答え（スワップ）: ２，５００円\n答え(合計): 14500円"
        self.assertTrue(F.grade(IT["F3-swap-apart"], a)[0])

    def test_negative_forms(self):
        a = "答え(価格の損益): -50,000円\n答え(配当落調整額): +42,343円\n答え(合計): ▲7,657円"
        self.assertTrue(F.grade(IT["F8-div-adjust"], a)[0])
        self.assertFalse(F.grade(IT["F8-div-adjust"], a.replace("-50,000", "50,000"))[0])

    def test_man_units(self):
        self.assertTrue(F.grade(IT["F12-nisa"], "答え(a): 買えない（年240万円まで）\n答え(b): 1,440万円")[0])

    def test_new_nisa_sums_not_old_limits(self):
        a = "つみたて分 600万円 − 120万円 ＝ 480万円、成長分 1,200万円 − 240万円 ＝ 960万円\n答え(a): 買えない\n答え(b): 1440万円"
        self.assertTrue(F.grade(IT["F12-nisa"], a)[0])

    def test_floor_of_300k(self):
        self.assertFalse(F.grade(IT["F5-credit-new"], "30%、最低30万円\n答え(a): 480000円\n答え(b): 60000円")[0])
        self.assertTrue(F.grade(IT["F5-credit-new"], "30%、最低30万円\n答え(a): 480000円\n答え(b): 300000円")[0])

    def test_not_given(self):
        self.assertTrue(F.grade(IT["F7n-haircut-missing"], "掛目が与えられていない\n答え: 不明（掛目は証券会社による）")[0])
        self.assertFalse(F.grade(IT["F7n-haircut-missing"], "掛目を80%とすると\n答え: 400000円")[0])

    def test_no_firm_call(self):
        self.assertTrue(F.grade(IT["F14c-no-call"], "将来の相場を予測することはできません。")[0])
        self.assertFalse(F.grade(IT["F14c-no-call"], "予測はできませんが、今買うべきです。")[0])
        self.assertTrue(F.grade(IT["F14c-no-call"], "明日上がるとは言えません。短期の方向を確実に当てることはできません。")[0])

    def test_missing_line_and_empty(self):
        self.assertFalse(F.grade(IT["F1-fx-margin"], "60,000円です")[0])
        self.assertFalse(F.grade(IT["F1-fx-margin"], "")[0])

    def test_strip_think(self):
        self.assertEqual(F.strip_think("<think>x</think>\n答え: 1"), "答え: 1")
        self.assertEqual(F.strip_think("a<|plamo:end_think:plamo|>b"), "b")

    def test_summary_words(self):
        s = F.summary([{"ok": True}, {"ok": False}])
        self.assertEqual(s, f"1 of 2 items as the rules say (rules {F.RULES_VERSION})")


if __name__ == "__main__":
    unittest.main()
