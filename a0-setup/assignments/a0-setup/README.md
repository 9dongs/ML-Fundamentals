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

---
**A0 과제**

`pairwise_sq_dists`는 브로드캐스팅과 행렬곱을 통해 배열 연산을 NumPy의 최적화된 내부 코드에서 수행하므로, Python 이중 반복문의 반복 처리 비용을 줄여 일반적으로 더 빠르다.
`keepdims=True`는 연산으로 축소한 축을 길이 1로 유지하는 옵션이다.
`row_normalize`에서는 각 행의 L2 노름을 `(n, 1)` 모양으로 유지하여, `(n, d)` 배열의 각 행이 자신의 노름으로 나누어지도록 브로드캐스팅을 맞춘다.
`softmax_rows`에서 한 행의 최댓값 `m`을 빼면 분자와 분모의 모든 지수항에 같은 인자 `exp(-m)`이 곱해져 약분되므로 결과는 수학적으로 같으며, 지수 함수의 입력이 0 이하가 되어 오버플로를 방지할 수 있다.

---

If A0 takes much longer than an afternoon or two, come to office hours in
Week 1 — before A1, not after.
