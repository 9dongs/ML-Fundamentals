# A0 — Environment Setup & NumPy Drills (Week 1, ungraded diagnostic)

## 1. Set up Python (once)

Any Python ≥ 3.10 with `numpy` and `pytest` works. The recommended way:

```
conda create -n ml python=3.12 numpy pytest -y
conda activate ml
```

(or: `python -m venv ml && source ml/bin/activate && pip install numpy pytest`)

Google Colab also works: upload this folder and run the same commands in a cell
with `!` in front.

## 2. Do the drills

Open `assignments/a0-setup/README.md` and implement the three functions in
`assignments/a0-setup/starter/drills.py`.

## 3. Check yourself

From THIS directory (the one containing `grade.py`):

```
python grade.py a0
```

7/7 public tests = you are set up and NumPy-ready. `python -m pytest assignments/a0-setup/tests -q`
runs the same tests directly. A0 is ungraded — it exists to make sure your
toolchain works before A1, and to hand you the tools the rest of the course uses.

**Also submit (written, 3–5 sentences):** see the question at the bottom of
`assignments/a0-setup/README.md`.

Due: Week 2. If A0 takes much longer than an afternoon or two, come to office
hours this week.
