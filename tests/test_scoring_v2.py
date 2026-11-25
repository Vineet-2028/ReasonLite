import unittest
import tempfile
from pathlib import Path
from reasonlite.scoring import grade
from reasonlite.data import write_rows, load_rows, dump_json, digest
class FinalAnswerTests(unittest.TestCase):
    def test_box(self): self.assertTrue(grade(r'Thus, Tom has \(\boxed{18}\) apples.',18)['correct'])
    def test_fraction(self): self.assertEqual(grade(r'\boxed{\frac{1}{2}}',0)['parsed'],'1/2')
    def test_bold(self): self.assertTrue(grade('Therefore, the cost is **$7.00**.',7)['correct'])
    def test_prose(self): self.assertTrue(grade('So, it takes 7 days.',7)['correct'])
    def test_not_strict(self): self.assertFalse(grade(r'\boxed{18}',18)['format_ok'])
    def test_truncated(self): self.assertFalse(grade(r'\boxed{18}',18,truncated=True)['valid'])
    def test_ambiguous(self): self.assertFalse(grade('Therefore, 7 or 9.',7)['valid'])
    def test_intermediate(self): self.assertFalse(grade('Step 1: 7\n\nStill working.',7)['valid'])
    def test_multiple_boxes(self): self.assertFalse(grade(r'\boxed{1} and \boxed{2}',2)['valid'])
    def test_unicode_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); f=p/'sft.jsonl'
            write_rows(f,[{'question':'one\u2028two','answer':'#### 2'}],[0],'train')
            dump_json(p/'manifest.json',{'files':{'sft':{'sha256':digest(f),'count':1}}})
            self.assertEqual(len(load_rows(p,'sft')[1]),1)
