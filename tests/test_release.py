import math, unittest, json
from reasonlite.logging_utils import sanitize
from reasonlite.final_report import exact_mcnemar, holm
from reasonlite.final_eval import completion
class ReleaseTests(unittest.TestCase):
    def test_nonfinite(self):
        v, issues = sanitize([{'grad_norm':float('nan')}, {'x':float('inf')}, {'grad_norm':0.0}])
        self.assertEqual(v,[{'grad_norm':None},{'x':None},{'grad_norm':0.0}])
        self.assertEqual([x['original_value'] for x in issues],['nan','inf'])
        json.dumps(v,allow_nan=False)
    def test_mcnemar(self):
        self.assertEqual(exact_mcnemar(0,0),1)
        self.assertEqual(exact_mcnemar(0,3),.25)
        self.assertEqual(exact_mcnemar(3,3),1)
    def test_holm(self):
        self.assertEqual(holm([.02,.01,.5]),[.04,.03,.5])
    def test_eos(self):
        self.assertEqual(completion([4,2,0,0],{2}),([4,2],False))
