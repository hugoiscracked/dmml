# Canvas Week Pages

Canvas-ready HTML for the weekly **Overview** pages in the T-809-DATA Canvas course.
Each `wNN_*.html` is one Canvas Page.

## Pasting into Canvas

1. Create or open the week's Page (e.g. *Week 01: Overview*).
2. Click the **`</>`** button (bottom-right of the rich-content editor) to switch to the HTML editor.
3. Paste the file contents (everything after the leading `<!-- ... -->` comment is fine to include — Canvas strips comments).
4. Save.

## Images (animations + skill tree)

Image `src` values are placeholders (`REPLACE_WITH_CANVAS_FILE/...`). Canvas can't read
local repo files, so:

1. Upload the week's GIFs (`animations/wNN_*/*.gif`) and the skill-tree PNG
   (`skill_tree/out/week_NN.png`) to **Course Files**.
2. Either swap the `src` in the HTML before pasting, or click each broken image in the
   rich editor afterwards and re-point it at the uploaded file.

## Styling notes

- All styling is **inline** on purpose — Canvas strips `<style>` blocks and `<script>`.
- External interactive demos are **links/buttons**, not `<iframe>`s, because Canvas only
  allows iframe embeds from whitelisted domains.

## Image assets per page

Upload these to Course Files, then point each `REPLACE_WITH_CANVAS_FILE/...` src at them.

| Page | Animations (`animations/<dir>/`) | Skill tree (`skill_tree/out/`) |
|------|----------------------------------|-------------------------------|
| `w01_setup_eda` | EDA, DataCleaning, StatsOverview | week_01.png |
| `w02_time_series_arima` | Decomposition, Stationarity, AR1, ForecastCone | week_02.png |
| `w03_regression_classification` | BiasVariance, DecisionBoundary | week_03.png |
| `w04_svm_tuning` | SVMMargin, KernelTrick, CrossValidation | week_04.png |
| `w05_trees_ensembles` | DecisionTree, Bagging, FeatureImportance | week_05.png |
| `w06_boosting_shap` | GradientBoosting, SHAPValues | week_06.png |
| `w07_clustering_pca` | KMeans, DBSCAN, PCA | week_07.png |
| `w08_neural_networks` | MLPForward, GradientDescent, ActivationFunctions | week_08.png |
| `w09_training_cnns` | Convolution, DropoutBatchNorm, CNNArchitecture | week_09.png |
| `w10_sequences_transformers` | RNNvsLSTM, Attention, Transformer, FineTuning | week_10.png |
| `w11_midterm_project` | — (none) | — |
| `w12_case_studies` | — (none) | full.png |
| `project` | — (none) | — |

The cluster note appears on **W01** (interactive intro), **W04** (CPU batch jobs), **W09** (GPU), and as a brief reminder on **W11** and the **Project** page.

## Project page

`project.html` is the canonical project hub (running-thread model, rubric, dataset
sources, timeline). It isn't week-bound — put it in a dedicated **Project** module and
link to it from the **Week 6** module (kickoff), W9 (proposal), W11 (clinic), and W12.
The project **kicks off in W6** and runs as a semester-long thread (dataset chosen
early, methods layered on as taught); the W11 page is a *clinic*, not a kickoff.

## Page structure (every week)

- Header (week number + theme)
- Overview paragraph
- Learning objectives
- *Watch it move* — the week's Manim animations + skill-tree position
- *Play with it* — curated interactive demos (the centerpiece)
- Optional reading (light)
- This week's exercise pointer
