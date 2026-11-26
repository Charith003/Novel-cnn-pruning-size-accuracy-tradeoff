"""
AMFI Pruning - Flask backend.
Run:  python app.py   then open http://localhost:5000
"""
import os, io, json, threading, time
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, send_file
from PIL import Image

import pruning_core as core

app = Flask(__name__, static_folder="static")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, core.OUT_DIR)
DEFAULT_CSV = os.path.join(BASE, "results_default.csv")
LIVE_CSV = os.path.join(OUT, "pruning_results.csv")

lock = threading.Lock()
state = {"running": False, "progress": 0.0, "log": [], "error": None, "stop": False}
model_cache = {}


def log(msg):
    with lock:
        state["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        state["log"] = state["log"][-400:]


def results_path():
    return LIVE_CSV if os.path.exists(LIVE_CSV) else DEFAULT_CSV


def load_results():
    df = pd.read_csv(results_path())
    base = df[df["Method"] == "Baseline"].iloc[0]
    df["Parameter_Reduction_Percent"] = 100 * (base["Parameters"] - df["Parameters"]) / base["Parameters"]
    df["Size_Reduction_Percent"] = 100 * (base["Model_Size_MB"] - df["Model_Size_MB"]) / base["Model_Size_MB"]
    df["Accuracy_Drop"] = base["Accuracy_Percent"] - df["Accuracy_Percent"]
    return df


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/results")
def api_results():
    df = load_results().astype(object).where(pd.notnull(load_results()), None)
    return jsonify({"source": "live run" if results_path() == LIVE_CSV else "notebook results (default)",
                    "rows": df.to_dict(orient="records")})


@app.route("/api/download/csv")
def api_csv():
    return send_file(results_path(), as_attachment=True, download_name="pruning_results.csv")


@app.route("/api/device")
def api_device():
    return jsonify({"device": str(core.get_device())})


@app.route("/api/run", methods=["POST"])
def api_run():
    if state["running"]:
        return jsonify({"error": "An experiment is already running"}), 409
    d = request.get_json(force=True) or {}
    try:
        cfg = {
            "dataset": d.get("dataset", "CIFAR10"),
            "base_epochs": max(1, min(int(d.get("base_epochs", 5)), 50)),
            "finetune_epochs": max(1, min(int(d.get("finetune_epochs", 1)), 20)),
            "ratios": sorted({min(max(float(r), 0.05), 0.9) for r in d.get("ratios", [0.2, 0.4, 0.6])}),
            "methods": [m for m in d.get("methods", core.METHODS_ALL) if m in core.METHODS_ALL],
            "batch_size": 128, "num_workers": 2, "seed": int(d.get("seed", 42)),
        }
        if cfg["dataset"] not in ("CIFAR10", "CIFAR100") or not cfg["methods"] or not cfg["ratios"]:
            raise ValueError("invalid dataset, methods or ratios")
    except Exception as e:
        return jsonify({"error": f"Bad request: {e}"}), 400

    with lock:
        state.update(running=True, progress=0.0, log=[], error=None, stop=False)
    model_cache.clear()

    def job():
        try:
            core.Experiment(cfg, log, lambda p: state.__setitem__("progress", p),
                            lambda: state["stop"]).run()
        except InterruptedError:
            log("Stopped.")
        except Exception as e:
            state["error"] = str(e); log(f"ERROR: {e}")
        finally:
            state["running"] = False

    threading.Thread(target=job, daemon=True).start()
    return jsonify({"started": True, "config": cfg})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    state["stop"] = True
    return jsonify({"stopping": True})


@app.route("/api/status")
def api_status():
    with lock:
        return jsonify({k: state[k] for k in ("running", "progress", "log", "error")})


@app.route("/api/models")
def api_models():
    """Models that have a saved .pth file and can be used for prediction."""
    df = load_results()
    out = []
    for _, r in df.iterrows():
        name = "baseline_model.pth" if r["Method"] == "Baseline" else \
            f"{r['Method'].lower()}_{round(r['Pruning_Ratio']*100)}pct.pth"
        if os.path.exists(os.path.join(OUT, name)):
            out.append({"method": r["Method"], "ratio": float(r["Pruning_Ratio"]),
                        "file": name, "accuracy": float(r["Accuracy_Percent"])})
    return jsonify(out)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    f = request.files.get("image")
    method = request.form.get("method", "Baseline")
    ratio = float(request.form.get("ratio", 0))
    if f is None:
        return jsonify({"error": "No image uploaded"}), 400
    df = load_results()
    row = df[(df["Method"] == method) & ((df["Pruning_Ratio"] - ratio).abs() < 1e-6)]
    if row.empty:
        return jsonify({"error": "Unknown model"}), 404
    fname = "baseline_model.pth" if method == "Baseline" else f"{method.lower()}_{round(ratio*100)}pct.pth"
    path = os.path.join(OUT, fname)
    if not os.path.exists(path):
        return jsonify({"error": "No trained weights yet. Run an experiment first (Run tab)."}), 404
    if fname not in model_cache:
        model_cache[fname] = core.load_model(json.loads(row.iloc[0]["Channels"]), path)
    try:
        img = Image.open(io.BytesIO(f.read()))
        return jsonify({"model": fname, "predictions": core.predict_image(model_cache[fname], img)})
    except Exception as e:
        return jsonify({"error": f"Could not process image: {e}"}), 400


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
