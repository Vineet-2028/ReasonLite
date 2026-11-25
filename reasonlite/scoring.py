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

def grade(text, gold):
    strict, relaxed = extract(text, True), extract(text)
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
