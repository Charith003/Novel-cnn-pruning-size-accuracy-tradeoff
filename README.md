# Novel CNN Pruning: Size–Accuracy Tradeoff

A research project exploring **AMFI (Adaptive Multi-Factor Importance) pruning** for convolutional neural networks, alongside common pruning baselines. The repository includes the research notebook, experiment summaries, and a Flask dashboard for inspecting results and running new experiments.

> **Research note:** the checked-in experiment tables report a single seed (42). Treat the values as an experimental snapshot, not as a multi-run statistical comparison.

## Project contents

| Path | Description |
| --- | --- |
| `AMFI_Complete_Research.ipynb` | Research workflow and analysis notebook |
| `app.py`, `pruning_core.py` | Flask app and pruning/training implementation |
| `static/index.html` | Dashboard interface |
| `requirements.txt` | Python dependencies |
| `results_default.csv` | Results initially displayed by the dashboard |
| `research_results/` | Experiment tables, summary statistics, runtime profile, plot, and PDF report |

## Research snapshot

The checked-in final test results compare a baseline with unstructured pruning methods at 20% and 50% target sparsity:

| Method | Baseline | 20% sparsity | 50% sparsity |
| --- | ---: | ---: | ---: |
| Baseline | 41.61% | — | — |
| Random | — | 49.17% | 49.49% |
| Magnitude | — | 47.26% | 47.67% |
| L1 | — | 48.31% | 47.83% |
| AMFI | — | 48.35% | 47.99% |
| Adaptive AMFI | — | 48.47% | 47.85% |

These values are from one recorded experiment seed. They are not evidence of statistical significance or a guarantee that another run will reproduce the same ranking. The detailed CSVs in `research_results/` include additional measurements and structured pruning experiments.

## Run the web app

Python 3.10 or newer is recommended. From the repository root:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser. The app downloads CIFAR-10 when needed. A CUDA-capable GPU is recommended for training; CPU runs may take substantially longer.

The dashboard shows the included default results until you run an experiment. The app can launch pruning experiments, stream progress, save new outputs under `amfi_outputs/`, and classify an uploaded CIFAR-10 image after a model has been trained.

## Run the notebook

Open `AMFI_Complete_Research.ipynb` in Jupyter or VS Code and run its cells in order. Install the dependencies in `requirements.txt` first. Notebook cells that train models may take time and may require downloading CIFAR-10.

## Results and generated files

- `research_results/experiment_results.csv` contains per-experiment measurements.
- `research_results/final_test_results.csv` contains the final test accuracies summarized above.
- `research_results/statistics_summary.csv` and the other CSV files contain additional summaries and ablations.
- `research_results/accuracy_vs_sparsity.png` and `research_results/amfi_research_report.pdf` provide visual and report summaries.
- New app-generated weights and run data are written to `amfi_outputs/` and are ignored by Git.

The downloaded CIFAR-10 files, local Python environments, and large model checkpoints are intentionally not included. This keeps the repository lightweight; run the notebook or app to regenerate model artifacts.

## Pruning methods

The project compares AMFI and adaptive AMFI with random, magnitude, and L1 pruning, plus structured pruning experiments. Results report accuracy and other measurements such as sparsity, checkpoint size, latency, throughput, and MACs where available.

