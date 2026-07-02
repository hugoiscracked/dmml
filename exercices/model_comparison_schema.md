# DMML Benchmark Table

The comparison table is a cumulative numeric benchmark, not a skill-tree summary
and not a prose model report.

The idea is simple: as the course introduces more modelling tools, students add
new model results to a reusable table. Over time, the table expands:

- new datasets add rows;
- new models add columns in the wide view;
- repeated metrics make strengths and weaknesses visible.

## Storage Format: Long

Use long format while collecting results. It is easy to append to:

- `week`
- `dataset`
- `task_type`
- `target`
- `model`
- `metric`
- `score`
- `split`
- `notes`

Example:

| week | dataset | task_type | target | model | metric | score | split |
|---|---|---|---|---|---|---:|---|
| W02 | AirPassengers | forecasting | passengers | seasonal_naive_12 | rmse | 42.1 | last_24_months |

## Display Format: Wide

For the final course takeaway, pivot to wide format:

| dataset | task_type | metric | naive_last_value | seasonal_naive_12 | ARIMA |
|---|---|---|---:|---:|---:|
| AirPassengers | forecasting | rmse | 93.4 | 42.1 | 39.8 |

This gives the feeling of a growing table: each new model becomes another column
students can reuse on future compatible datasets.

## Compatibility Rule

Only compare models numerically when the task, target, split, and metric are
compatible. Forecasting models belong with forecasting datasets; classifiers
belong with classification datasets; clustering needs its own metrics.

## Minimal Helper Pattern

Each notebook can build a small local `benchmark_long` table, then create:

```python
benchmark_wide = benchmark_long.pivot_table(
    index=["dataset", "task_type", "target", "metric", "split"],
    columns="model",
    values="score",
    aggfunc="first",
).reset_index()
```

The long table is what students append; the wide table is what they inspect.
