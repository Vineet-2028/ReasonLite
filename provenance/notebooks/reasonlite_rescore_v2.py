"""ReasonLite v2 scoring repair. Run in the project root; preserve original runs."""
from pathlib import Path
import json, re, sys, shutil, hashlib, subprocess
from fractions import Fraction
ROOT = Path.cwd()
assert (ROOT/'reasonlite/scoring.py').is_file(), 'Run in ReasonLite root'

def edit(path, old, new):
    p = ROOT/path
    s = p.read_text(encoding='utf-8')
    if new in s: return
    if s.count(old) != 1: raise RuntimeError(f'Unexpected source in {path}: {old}')
    p.write_text(s.replace(old,new),encoding='utf-8')

# Fix the JSONL reader if this is a fresh checkout; retain frozen file bytes.
edit('reasonlite/data.py', "rows = [json.loads(s) for s in file.read_text(encoding='utf-8').splitlines()]", "with file.open(encoding='utf-8') as f:\n        rows = [json.loads(line) for line in f]")
p=ROOT/'reasonlite/compare.py'; s=p.read_text(encoding='utf-8')
if '.splitlines()' in s:
    edit('reasonlite/compare.py', "return json.loads((root/'metrics.json').read_text()), [json.loads(l) for l in (root/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]", "with (root/'predictions.jsonl').open(encoding='utf-8') as f:\n            predictions = [json.loads(line) for line in f]\n        return json.loads((root/'metrics.json').read_text()), predictions")

extra = r'''

SCORING_VERSION = 'final-answer-v2'
# Only explicit final-paragraph answers; no last-number guessing.
BOX = re.compile(r"\\boxed\s*\{\s*(\\frac\s*\{[+-]?\d+\}\s*\{[+-]?\d+\}|\\?\$?\s*" + NUM + r")\s*\}")
CONCLUSION = re.compile(r"^(?:so\b|thus\b|therefore\b|hence\b|(?:the\s+)?final answer\b|(?:the\s+)?answer\b)", re.I)
UNSAFE = re.compile(r"\b(?:or|unsure|uncertain|maybe|might|cannot|could not|if|approximately|about)\b",re.I)
PLAIN_NUM = re.compile(r"(?<![\w.,])"+NUM+r"(?![\w.,/])",re.ASCII)

def final_answer(text):
    paragraph = re.split(r'\n\s*\n',text.strip())[-1].strip()
    if not paragraph or UNSAFE.search(paragraph): return None
    boxes = BOX.findall(paragraph)
    if paragraph.count(r'\boxed'):
        if len(boxes)!=1 or paragraph.count(r'\boxed')!=1: return None
        raw=boxes[0].strip().replace('\\$','').lstrip('$').strip()
        frac=re.fullmatch(r'\\frac\s*\{([+-]?\d+)\}\s*\{([+-]?\d+)\}',raw)
        if frac: raw=frac.group(1)+'/'+frac.group(2)
        return number(raw)
    # Explicit concluding prose with one number, no math expression or alternatives.
    line=paragraph.strip().replace('**','').replace('\\$','$')
    if not CONCLUSION.match(line) or '\n' in line or re.search(r'[=<>%/\\]',line): return None
    # Remove terminal sentence punctuation before checking numeric boundaries.
    line=line.rstrip('.!?').strip()
    nums=list(PLAIN_NUM.finditer(line))
    if len(nums)!=1: return None
    return number(nums[0].group())
'''
p=ROOT/'reasonlite/scoring.py'; s=p.read_text(encoding='utf-8')
if 'SCORING_VERSION' not in s:
    p.write_text(s+extra,encoding='utf-8')
edit('reasonlite/scoring.py',"def grade(text, gold):\n    strict, relaxed = extract(text, True), extract(text)","def grade(text, gold, truncated=False):\n    strict, relaxed = extract(text, True), extract(text)\n    if relaxed is None and not truncated:\n        relaxed = final_answer(text)")
edit('reasonlite/evaluate.py','from .scoring import grade','from .scoring import grade, SCORING_VERSION')
edit('reasonlite/evaluate.py',"**grade(text,Fraction(row['gold']))", "**grade(text,Fraction(row['gold']),truncated=count == a.max_new_tokens)")
edit('reasonlite/evaluate.py',"metrics.update({'model':manifest['model']", "metrics.update({'scoring_version':SCORING_VERSION,'model':manifest['model']")
edit('reasonlite/compare.py',"KEYS = ['model_revision'", "KEYS = ['scoring_version','model_revision'")

# Regressions include invalid/truncated conclusions and the Unicode JSONL issue.
p=ROOT/'tests/test_scoring_v2.py'
p.write_text(r'''import unittest
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
''',encoding='utf-8')
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],check=True)
from reasonlite.data import load_rows, dump_json, digest
from reasonlite.scoring import grade, SCORING_VERSION
from reasonlite.evaluate import summarize
manifest, frozen=load_rows(ROOT/'data/prepared','validation')
for name in ['e0_smoke','e0_validation']:
    source=ROOT/'runs'/name; dest=source.with_name(name+'_v2')
    m=json.loads((source/'metrics.json').read_text(encoding='utf-8'))
    with (source/'predictions.jsonl').open(encoding='utf-8') as f: rows=[json.loads(line) for line in f]
    assert len(rows)==m['n'] and [(r['id'],r['gold']) for r in rows]==[(r['id'],r['gold']) for r in frozen[:len(rows)]]
    assert m['data_manifest_sha256']==digest(ROOT/'data/prepared/manifest.json') and m['split_sha256']==manifest['files']['validation']['sha256']
    assert m['model_revision']==manifest['model_revision']
    dest.mkdir(exist_ok=True)
    new=[{**r,**grade(r['output'],Fraction(r['gold']),truncated=r['hit_token_limit'])} for r in rows]
    with (dest/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for r in new: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    metrics={**m,**summarize(new),'scoring_version':SCORING_VERSION,'rescored_from':name,'original_predictions_sha256':digest(source/'predictions.jsonl')}
    dump_json(dest/'metrics.json',metrics)
    print(name, json.dumps(metrics,indent=2))
