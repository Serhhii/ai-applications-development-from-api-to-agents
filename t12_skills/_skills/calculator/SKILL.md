---
name: calculator
description: >
  A safe math expression evaluator. Use this skill whenever the user asks to
  calculate, compute, evaluate, or solve a numeric expression. Supports
  arithmetic, powers, trigonometry, square roots, logarithms, floor/ceiling,
  and the constants pi and e.
---

# Calculator Skill

## Quick Start

Run the script with the expression as a single quoted argument:

```
python /skills/calculator/scripts/calculate.py "<expression>"
```

Example:
```
python /skills/calculator/scripts/calculate.py "sqrt(144) + 2^8"
```

## Supported Operations

| Category | Syntax / Functions |
|---|---|
| Arithmetic | `+`, `-`, `*`, `/` |
| Power / exponentiation | `**` or `^` (both work) |
| Floor division & modulo | `//`, `%` |
| Square root | `sqrt(x)` |
| Absolute value | `abs(x)` |
| Rounding | `round(x)`, `floor(x)`, `ceil(x)` |
| Logarithms | `log(x)`, `log10(x)` |
| Trigonometry | `sin(x)`, `cos(x)`, `tan(x)` (radians) |
| Constants | `pi`, `e` |
| Grouping | `( )` |

## Workflow

1. Parse the user's request and extract the mathematical expression.
2. Translate any natural-language parts into a valid expression (e.g. "two to the power of ten" → `2^10`).
3. Run the script: `python /skills/calculator/scripts/calculate.py "<expression>"`
4. Return the printed `Result:` line to the user.
5. If the script prints an `Error:`, explain it clearly and ask the user to rephrase.
