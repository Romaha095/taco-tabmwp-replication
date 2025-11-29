import re
from typing import Dict, Tuple


def _round_number_str(x: float) -> str:
    if abs(x - int(round(x))) < 1e-9:
        return str(int(round(x)))

    val = round(float(x), 3)
    s = f"{val:.3f}"
    s = re.sub(r"0+$", "", s)
    s = re.sub(r"\.$", "", s)
    return s


def _sanitize_lhs_for_eval(lhs: str) -> str:
    lhs_clean = re.sub(r"[^0-9\+\-\*x×÷\/\.\,\(\)\s]", " ", lhs)
    lhs_clean = lhs_clean.replace("x", "*").replace("×", "*").replace("÷", "/")
    lhs_clean = lhs_clean.replace("$", "")
    lhs_clean = lhs_clean.replace(",", "")
    lhs_clean = re.sub(r"\s+", " ", lhs_clean).strip()
    return lhs_clean


def _has_operator(expr: str) -> bool:
    return any(op in expr for op in ["+", "-", "*", "/"])


def _parse_number_from_rhs(rhs_num_str: str):
    if not rhs_num_str:
        return None

    s = rhs_num_str.replace(",", "").strip()

    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0])
                den = float(parts[1])
                if den != 0:
                    return num / den
            except ValueError:
                return None

    try:
        return float(s)
    except ValueError:
        return None


def fix_cot_with_calculator(cot: str) -> Tuple[str, Dict[str, int]]:
    stats = {
        "num_equations": 0,
        "num_eval_success": 0,
        "num_rhs_numeric": 0,
        "num_rhs_correct": 0,
        "num_rhs_fixed": 0,
    }

    lines = cot.split("\n")
    new_lines = []

    for line in lines:
        if "=" not in line:
            new_lines.append(line)
            continue

        lhs_orig, rhs_orig = line.split("=", 1)
        stats["num_equations"] += 1

        lhs_expr = _sanitize_lhs_for_eval(lhs_orig)
        if not lhs_expr or not _has_operator(lhs_expr):
            # Not a valid arithmetic expression, keep the line as is
            new_lines.append(line)
            continue

        try:
            # Safe eval: no builtins
            value = eval(lhs_expr, {"__builtins__": {}}, {})
        except Exception:
            # If evaluation fails, keep the original line
            new_lines.append(line)
            continue

        stats["num_eval_success"] += 1
        result_str = _round_number_str(float(value))

        # Split RHS into numeric prefix and textual tail (e.g. units)
        match = re.match(r"(\s*[\-\+0-9\.,/]+)(.*)", rhs_orig)
        if match:
            rhs_num_str = match.group(1)
            tail = match.group(2)
            rhs_value = _parse_number_from_rhs(rhs_num_str)
            if rhs_value is not None:
                stats["num_rhs_numeric"] += 1
                rhs_canon = _round_number_str(rhs_value)
                if rhs_canon == result_str:
                    # RHS already correct -> mark as correct, keep original line
                    stats["num_rhs_correct"] += 1
                    new_lines.append(line)
                    continue
            # If numeric but different, we will fix it below
        else:
            rhs_num_str = ""
            tail = rhs_orig

        # Here we set the RHS to the correct result (and keep the tail, e.g. units)
        stats["num_rhs_fixed"] += 1
        new_rhs = f" {result_str}{tail}"
        new_line = lhs_orig + "=" + new_rhs
        new_lines.append(new_line)

    fixed_cot = "\n".join(new_lines)
    return fixed_cot, stats
