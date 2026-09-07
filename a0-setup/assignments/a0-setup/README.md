# A0 — Environment Setup & NumPy Self-Study (Week 1, ungraded diagnostic)

NumPy is **self-study** in this course: Week 1's lectures spend their hours on
the course itself and on probability, and Week 2 consolidates the four core
NumPy ideas in lecture *after* linear algebra, where they belong. This
assignment is the self-study — it verifies your toolchain end to end and
walks you through every NumPy idea the course relies on, one drill per idea.

**Read alongside** (in this order): textbook ch. 1, NumPy section · the
week01 slide deck's "NumPy fluency" section (marked self-study) · the week01
Colab notebook, which re-runs every demo.

**The seven drills** in `starter/drills.py`, each certifying one idea —
do them in order, no Python loops anywhere:

| # | function | the idea it certifies |
|---|---|---|
| 1 | `middle_block` | indexing & slicing (and views) |
| 2 | `replace_negatives` | boolean masks; copy vs view |
| 3 | `row_normalize` | reductions along an axis; `keepdims=True` |
| 4 | `pairwise_sq_dists` | broadcasting (the course's workhorse) |
| 5 | `one_hot` | vectorized (fancy) indexing |
| 6 | `softmax_rows` | numerical stability (the max trick; returns Wks 5, 9, 11, 13) |
| 7 | `numerical_derivative` | central differences — your A1 gradient checker, built early |

Check yourself:
```
python -m pytest tests -q          # from this directory
```
All tests pass = you are set up and NumPy-ready. **Written (submit 3–5
sentences):** why is `pairwise_sq_dists` with broadcasting faster than a
double loop? What does `keepdims=True` do and why does `row_normalize` need
it? Why does subtracting the row max not change `softmax_rows`'s answer?

If A0 takes much longer than an afternoon or two, come to office hours in
Week 1 — before A1, not after.
