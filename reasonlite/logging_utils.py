"""Preserve nonfinite diagnostics as explicit JSON nulls and an audit sidecar."""
import math
def sanitize(value, path='root'):
    issues = []
    def visit(v, p):
        if isinstance(v, float) and not math.isfinite(v):
            issues.append({'path':p,'original_value':str(v)})
            return None
        if isinstance(v, list): return [visit(x, f'{p}[{i}]') for i,x in enumerate(v)]
        if isinstance(v, dict): return {k:visit(x,f'{p}.{k}') for k,x in v.items()}
        return v
    return visit(value,path), issues
