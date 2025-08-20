
import argparse
import subprocess
import yaml
import time
import re
import csv
import os
import sys
from pathlib import Path

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_metrics(output):
    # Robust regex for EM and F1 (case-insensitive, allow f1=... em=... or reversed order)
    em = f1 = None
    em_match = re.search(r"(?i)em\s*[=:]\s*([0-9.]+)", output)
    f1_match = re.search(r"(?i)f1\s*[=:]\s*([0-9.]+)", output)
    if em_match:
        em = float(em_match.group(1))
    if f1_match:
        f1 = float(f1_match.group(1))
    return em, f1

def run_script(args):
    start = time.time()
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.time() - start
    if proc.returncode == 0:
        em, f1 = parse_metrics(proc.stdout)
    else:
        em, f1 = None, None
    return {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
        "em": em,
        "f1": f1,
        "time": elapsed
    }

def run_ablation(config_path):
    config = load_config(config_path)
    results = []
    # Validate config
    if config.get("run_baseline", True):
        baseline_variant = next((v for v in config["variants"] if v["name"] == "baseline"), None)
        if not baseline_variant:
            print("ERROR: Baseline variant missing in config.")
            return
        eval_method = baseline_variant.get("eval_method", "stability")
        # Ensure topk/noise are bracketed if needed
        topk = config["topk"] if str(config["topk"]).startswith("[") else f"[{config['topk']}]"
        noise = config["noise"] if str(config["noise"]).startswith("[") else f"[{config['noise']}]"
        # Use sys.executable and resolve paths
        script_path = str(Path(config["baseline_script"]).resolve())
        # Support custom input file for baseline
        plus_file = baseline_variant.get("plus_file")
        # Use plus_file directly for baseline input file if set
        if plus_file:
            args = [sys.executable, script_path, config["date"], config["dataset"], eval_method, topk, noise, config.get("benchmark", "baseline"), plus_file]
        else:
            args = [sys.executable, script_path, config["date"], config["dataset"], eval_method, topk, noise, config.get("benchmark", "baseline")]
        if "seed" in config:
            args += [str(config["seed"])]
        if "max_eval" in config:
            args += [str(config["max_eval"])]
        print(f"Running BASELINE: {' '.join(args)}")
        res = run_script(args)
        results.append({
            "variant": "baseline",
            "em": round(res["em"],2) if res["em"] is not None else None,
            "f1": round(res["f1"],2) if res["f1"] is not None else None,
            "time": round(res["time"],2),
            "stdout": res["stdout"].strip(),
            "stderr": res["stderr"].strip(),
            "returncode": res["returncode"]
        })
        if res["returncode"] != 0:
            print(f"Error in baseline: {res['stderr']}")
    # Variants (skip baseline entry)
    for v in [vv for vv in config["variants"] if vv["name"] != "baseline"]:
        name = v["name"]
        plus_file = v.get("plus_file")
        eval_method = v.get("eval_method", "gpt35_turbo")
        topk = config["topk"] if str(config["topk"]).startswith("[") else f"[{config['topk']}]"
        noise = config["noise"] if str(config["noise"]).startswith("[") else f"[{config['noise']}]"
        script_path = str(Path(config["plus_script"]).resolve())
        args = [sys.executable, script_path, config["date"], config["dataset"], eval_method, topk, noise, config.get("benchmark", "baseline")]
        if plus_file:
            plus_file_path = f"{config['date']}_{config['dataset']}_edc2plus_compress_gpt35_turbo_noise{config['noise']}_topk{config['topk']}_{name.replace('+','')}.json"
            args.append(plus_file_path)
        if "seed" in config:
            args += [str(config["seed"])]
        if "max_eval" in config:
            args += [str(config["max_eval"])]
        print(f"Running {name}: {' '.join(args)}")
        res = run_script(args)
        results.append({
            "variant": name,
            "em": round(res["em"],2) if res["em"] is not None else None,
            "f1": round(res["f1"],2) if res["f1"] is not None else None,
            "time": round(res["time"],2),
            "stdout": res["stdout"].strip(),
            "stderr": res["stderr"].strip(),
            "returncode": res["returncode"]
        })
        if res["returncode"] != 0:
            print(f"Error in {name}: {res['stderr']}")
    # Output hygiene: write to runs/ folder
    runs_dir = Path("runs").resolve()
    runs_dir.mkdir(exist_ok=True)
    csv_path = str(runs_dir / "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "em", "f1", "time"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    md_path = str(runs_dir / "ablation_results.md")
    with open(md_path, "w") as f:
        f.write("| Variant | EM | F1 | Time (s) |\n|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['variant']} | {r['em']} | {r['f1']} | {r['time']} |\n")
    print(f"Results written to {csv_path} and {md_path}")

    config = load_config(config_path)
    results = []
    # Validate config
    if config.get("run_baseline", True):
        baseline_variant = next((v for v in config["variants"] if v["name"] == "baseline"), None)
        if not baseline_variant:
            print("ERROR: Baseline variant missing in config.")
            return
        eval_method = baseline_variant.get("eval_method", "stability")
        # Ensure topk/noise are bracketed if needed
        topk = config["topk"] if str(config["topk"]).startswith("[") else f"[{config['topk']}]"
        noise = config["noise"] if str(config["noise"]).startswith("[") else f"[{config['noise']}]"
        # Use sys.executable and resolve paths
        script_path = str(Path(config["baseline_script"]).resolve())
        args = [sys.executable, script_path, config["date"], config["dataset"], eval_method, topk, noise, config.get("benchmark", "baseline")]
        # Add seed and max_eval if present
        if "seed" in config:
            args += [str(config["seed"])]
        if "max_eval" in config:
            args += [str(config["max_eval"])]
        print(f"Running BASELINE: {' '.join(args)}")
        res = run_script(args)
        results.append({
            "variant": "baseline",
            "em": round(res["em"],2) if res["em"] is not None else None,
            "f1": round(res["f1"],2) if res["f1"] is not None else None,
            "time": round(res["time"],2),
            "stdout": res["stdout"].strip(),
            "stderr": res["stderr"].strip(),
            "returncode": res["returncode"]
        })
        if res["returncode"] != 0:
            print(f"Error in baseline: {res['stderr']}")
    # Variants (skip baseline entry)
    for v in [vv for vv in config["variants"] if vv["name"] != "baseline"]:
        name = v["name"]
        plus_file = v.get("plus_file")
        eval_method = v.get("eval_method", "gpt35_turbo")
        topk = config["topk"] if str(config["topk"]).startswith("[") else f"[{config['topk']}]"
        noise = config["noise"] if str(config["noise"]).startswith("[") else f"[{config['noise']}]"
        script_path = str(Path(config["plus_script"]).resolve())
        args = [sys.executable, script_path, config["date"], config["dataset"], eval_method, topk, noise, config.get("benchmark", "baseline")]
        if plus_file:
            args.append(plus_file)
        if "seed" in config:
            args += [str(config["seed"])]
        if "max_eval" in config:
            args += [str(config["max_eval"])]
        print(f"Running {name}: {' '.join(args)}")
        res = run_script(args)
        results.append({
            "variant": name,
            "em": round(res["em"],2) if res["em"] is not None else None,
            "f1": round(res["f1"],2) if res["f1"] is not None else None,
            "time": round(res["time"],2),
            "stdout": res["stdout"].strip(),
            "stderr": res["stderr"].strip(),
            "returncode": res["returncode"]
        })
        if res["returncode"] != 0:
            print(f"Error in {name}: {res['stderr']}")
    # Output hygiene: write to runs/ folder
    runs_dir = Path("runs").resolve()
    runs_dir.mkdir(exist_ok=True)
    csv_path = str(runs_dir / "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "em", "f1", "time"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    md_path = str(runs_dir / "ablation_results.md")
    with open(md_path, "w") as f:
        f.write("| Variant | EM | F1 | Time (s) |\n|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['variant']} | {r['em']} | {r['f1']} | {r['time']} |\n")
    print(f"Results written to {csv_path} and {md_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDC2plus ablation experiments.")
    parser.add_argument("--config", type=str, default="Codespace/Experiments/ablation.yaml", help="Path to ablation config YAML")
    args = parser.parse_args()
    run_ablation(args.config)