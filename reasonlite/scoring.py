"""Deterministic numeric scoring; never execute generated text."""
import re
from fractions import Fraction

# Accept standard integers, correctly grouped thousands, decimals and fractions.
NUM = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:/[+-]?\d+)?"
STRICT = re.compile(rf"####\s*\$?\s*({NUM})\s*\.?", re.ASCII)
FLEX = re.compile(rf"(?:final answer\s*:\s*|answer\s*:\s*)?\$?\s*({NUM})\s*\.?", re.I | re.ASCII)

def number(text):
    if not re.fullmatch(NUM, text, re.ASCII) or len(text) > 100:
        return None
    try:
        return Fraction(text.replace(',', ''))
    except (ValueError, ZeroDivisionError):
        return None

def extract(text, strict=False):
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    if not lines:
        return None
    m = STRICT.fullmatch(lines[-1])
    if not m and not strict:
        m = FLEX.fullmatch(lines[-1])
    return number(m.group(1)) if m else None

def reference(answer):
    # GSM8K gold answers end with a #### marker; fail loudly on malformed data.
    value = extract(answer, strict=True)
    if value is None:
        raise ValueError('Invalid reference answer')
    return value

def grade(text, gold, truncated=False):
    strict, relaxed = extract(text, True), extract(text)
    if relaxed is None and not truncated:
        relaxed = final_answer(text)
    return {'correct': relaxed is not None and relaxed == gold,
            'strict_correct': strict is not None and strict == gold,
            'valid': relaxed is not None, 'format_ok': strict is not None,
            'parsed': str(relaxed) if relaxed is not None else None}

def reward(text, gold, tokens, efficiency=False, budget=128):
    """Wrong answers get no length bonus. Correctness dominates bounded shaping."""
    if tokens < 0 or budget <= 0:
        raise ValueError('Invalid token count or budget')
    g = grade(text, gold)
    if not g['correct']:
        return 0.0
    penalty = min(1.0, max(0, tokens-budget)/budget) * 0.15 if efficiency else 0.0
    return 1.0 - penalty + (0.05 if efficiency and g['format_ok'] else 0.0)


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
