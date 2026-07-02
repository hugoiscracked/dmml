# DMML Exercises

The exercises are designed as self-contained Jupyter notebooks. The notebook is
the source of truth: task text, scaffolding, visible self-checks, final analysis,
reflection, and optional challenge work should all live there.

## Environment

Use Python 3.10, 3.11, or 3.12. From the repository root:

```bash
python -m pip install -r requirements.txt
```

Weeks 9 and 10 download datasets and pretrained models on first run. Students
who want GPU-enabled PyTorch should install the matching `torch` and
`torchvision` wheels from the official PyTorch instructions, then install the
remaining requirements.

## Design Principles

- Keep each week recoverable. Students should not need working code from a
  previous week to complete the current one.
- Reuse habits, not fragile dependencies. Week 1 introduces general EDA helpers
  that students may reuse later, but later notebooks should still stand alone.
- Keep implementation deliberate. Aim for four or five required functions per
  notebook, then spend the rest of the work interpreting results and trade-offs.
- Use visible self-checks. Tests should define the expected contract without
  hiding the entire grading logic.
- Use different datasets across weeks so students see where methods behave
  differently in practice.

## Benchmark Table Thread

When models solve comparable tasks, notebooks should include a small numeric
benchmark table. Store results in long format so they can be appended over time:

- `week`
- `dataset`
- `task_type`
- `target`
- `model`
- `metric`
- `score`
- `split`
- `notes`

Then pivot to wide format for inspection, so datasets/splits/metrics are rows
and models become columns. The goal is a course-long numeric record of which
tools work well on which kinds of datasets.
