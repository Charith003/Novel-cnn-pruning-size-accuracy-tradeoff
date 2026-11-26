# Novel CNN Pruning: Size–Accuracy Tradeoff

This repository contains an AMFI CNN pruning research project, its interactive Flask demo, the research notebook, and summarized experiment outputs.

## Web app

The Flask + PyTorch dashboard compares pruning methods, runs experiments, and supports CIFAR-10 image predictions.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000. A GPU is recommended for training experiments. The app downloads CIFAR-10 as needed.

## Research materials

- `AMFI_Complete_Research.ipynb` — complete research notebook
- `research_results/` — experiment tables, summary plot, runtime profile, and report
- `results_default.csv` — default results shown by the web dashboard

Large training checkpoints and the local virtual environment are excluded. Run the notebook or app experiments to regenerate model weights.
