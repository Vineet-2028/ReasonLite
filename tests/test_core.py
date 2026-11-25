import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from reasonlite.scoring import extract, reference, grade, reward
from reasonlite.data import choose, write_rows, load_rows, digest, dump_json
from reasonlite.evaluate import summarize
from reasonlite.compare import compare, KEYS

class ScoringTests(unittest.TestCase):
    def test_strict_marker(self): self.assertEqual(extract('work 12\n#### 1,200', True), 1200)
    def test_negative_decimal(self): self.assertEqual(extract('#### -0.50'), Fraction(-1,2))
    def test_fraction(self): self.assertEqual(extract('Final answer: 1/2'), Fraction(1,2))
    def test_relaxed_only(self): self.assertIsNone(extract('Answer: 42', True))
    def test_no_intermediate_guess(self): self.assertIsNone(extract('We need 42 items, but I am unsure.'))
    def test_trailing_content_invalid(self): self.assertIsNone(extract('#### 42\nBut wait...'))
    def test_malformed_numbers(self):
        for x in ['#### 1,2','#### 1/0','#### 3.5/2','#### 42 or 43','#### 50%']:
            self.assertIsNone(extract(x),x)
    def test_reference_strict(self):
        with self.assertRaises(ValueError): reference('42')
    def test_grade(self): self.assertEqual(grade('Answer: 42',42),{'correct':True,'strict_correct':False,'valid':True,'format_ok':False,'parsed':'42'})
    def test_reward_order(self):
        self.assertGreater(reward('#### 8',8,300,True),reward('#### 9',8,1,True))
        self.assertGreater(reward('#### 8',8,50,True),reward('#### 8',8,300,True))
        self.assertEqual(reward('#### 8',8,300),1)
    def test_zero_answer(self): self.assertTrue(grade('#### 0',0)['correct'])

class DataTests(unittest.TestCase):
    def test_splits(self):
        a=choose(100,42,40,20,10); self.assertEqual(a,choose(100,42,40,20,10))
        self.assertEqual(len(set(sum(a.values(),[]))),70)
        self.assertNotEqual(a,choose(100,43,40,20,10))
    def test_invalid_size(self):
        with self.assertRaises(ValueError): choose(10,42,5,5,5)
    def test_file_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); f=p/'sft.jsonl'; write_rows(f,[{'question':'1+1?','answer':'#### 2'}],[0],'train')
            dump_json(p/'manifest.json',{'files':{'sft':{'sha256':digest(f),'count':1}}})
            self.assertEqual(load_rows(p,'sft')[1][0]['gold'],'2')
            f.write_text('tampered')
            with self.assertRaises(ValueError): load_rows(p,'sft')

class MetricTests(unittest.TestCase):
    def test_metrics(self):
        a={'correct':True,'strict_correct':True,'valid':True,'format_ok':True,'tokens':10,'seconds':1,'hit_token_limit':False}
        b={**a,'correct':False,'strict_correct':False,'valid':False,'format_ok':False,'tokens':30,'seconds':3,'hit_token_limit':True}
        s=summarize([a,b]); self.assertEqual(s['accuracy'],.5); self.assertEqual(s['mean_tokens'],20)
        self.assertEqual(s['p95_latency_s'],3); self.assertEqual(s['invalid_rate'],.5); self.assertEqual(s['correct_per_1000_generated_tokens'],25)
        self.assertLess(s['wilson_95'][0],.5); self.assertGreater(s['wilson_95'][1],.5)
    def test_empty(self):
        with self.assertRaises(ValueError): summarize([])
    def test_paired_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); a=p/'a'; b=p/'b'; a.mkdir(); b.mkdir()
            m={**dict.fromkeys(KEYS,'same'),'n':1,'accuracy':0,'strict_accuracy':0,'mean_tokens':10}
            for folder, correct in [(a,False),(b,True)]:
                dump_json(folder/'metrics.json',{**m,'accuracy':int(correct)})
                (folder/'predictions.jsonl').write_text(json.dumps({'id':'train:1','gold':'4','correct':correct})+'\n')
            self.assertEqual(compare(a,b)['wrong_to_correct'],1)
            dump_json(b/'metrics.json',{**m,'max_new_tokens':5})
            with self.assertRaises(ValueError): compare(a,b)

if __name__=='__main__': unittest.main()
