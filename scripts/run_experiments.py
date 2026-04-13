"""
Executa todos os experimentos (modelo principal, baselines, ablation study)
sequencialmente e coleta os resultados em uma tabela consolidada.
"""
import subprocess
import sys
import os
import json
import glob
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

EXPERIMENTS = [
    {
        "name": "CNN-LSTM (full model)",
        "args": ["--model", "cnn_lstm", "--tag", "full-model"],
    },
    {
        "name": "Baseline: CNN-only (sem LSTM)",
        "args": ["--model", "cnn_only", "--tag", "baseline-cnn-only"],
    },
    {
        "name": "Baseline: CNN-LSTM frozen (sem fine-tuning)",
        "args": ["--model", "cnn_lstm_frozen", "--tag", "baseline-frozen"],
    },
    {
        "name": "Ablation: sem fine-tuning",
        "args": ["--model", "cnn_lstm", "--fine-tune-layers", "0", "--tag", "ablation-no-finetune"],
    },
    {
        "name": "Ablation: sem augmentation",
        "args": ["--model", "cnn_lstm", "--no-augmentation", "--tag", "ablation-no-aug"],
    },
    {
        "name": "Ablation: LSTM 128 unidades",
        "args": ["--model", "cnn_lstm", "--lstm-units", "128", "--tag", "ablation-lstm128"],
    },
]

ALL_USE_VIDEO_SPLIT = ["--split", "video"]


def run_experiment(name, args):
    print(f"\n{'='*60}")
    print(f"  EXPERIMENTO: {name}")
    print(f"{'='*60}")

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "train_model.py")] + args + ALL_USE_VIDEO_SPLIT
    print(f"  CMD: {' '.join(cmd)}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start

    print(f"\n  Tempo: {elapsed/60:.1f} min | Exit code: {result.returncode}")
    return result.returncode


def collect_results():
    """Lê todos os run_metadata.json + final_metrics.json + classification_report.txt"""
    logs_dir = PROJECT_ROOT / "logs"
    runs = sorted(logs_dir.glob("run-*"), key=os.path.getmtime)

    results = []
    for run_dir in runs:
        meta_path = run_dir / "run_metadata.json"
        metrics_path = run_dir / "final_metrics.json"
        report_path = run_dir / "classification_report.txt"

        if not meta_path.exists() or not metrics_path.exists():
            continue

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        report = ""
        if report_path.exists():
            with open(report_path, encoding="utf-8") as f:
                report = f.read()

        results.append({
            "run_dir": str(run_dir.name),
            "tag": meta.get("tag", ""),
            "model_type": meta.get("model_type", ""),
            "split_strategy": meta.get("split_strategy", ""),
            "augmentation": meta.get("augmentation", True),
            "lstm_units": meta.get("lstm_units", 64),
            "fine_tune_layers": meta.get("fine_tune_layers_actual", ""),
            "img_size": meta.get("img_size", 128),
            "n_samples": meta.get("n_samples", 0),
            "n_videos": meta.get("n_videos", 0),
            "accuracy": metrics.get("accuracy", 0),
            "loss": metrics.get("loss", 0),
            "classification_report": report,
        })

    return results


def print_summary(results):
    print(f"\n{'='*80}")
    print("  RESUMO DE TODOS OS EXPERIMENTOS")
    print(f"{'='*80}")
    print(f"{'Tag':<30} {'Model':<18} {'Split':<8} {'Aug':<5} {'FT':<4} {'LSTM':<5} {'Acc':<8} {'Loss':<8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['tag']:<30} {r['model_type']:<18} {r['split_strategy']:<8} "
            f"{'Y' if r['augmentation'] else 'N':<5} {str(r['fine_tune_layers']):<4} "
            f"{r['lstm_units']:<5} {r['accuracy']:.4f}  {r['loss']:.4f}"
        )

    summary_path = PROJECT_ROOT / "logs" / "experiments_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResumo salvo em: {summary_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  BATCH DE EXPERIMENTOS — BRACIS 2026")
    print(f"  Total de experimentos: {len(EXPERIMENTS)}")
    print("=" * 60)

    failed = []
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}]")
        code = run_experiment(exp["name"], exp["args"])
        if code != 0:
            failed.append(exp["name"])
            print(f"  *** FALHOU: {exp['name']} ***")

    results = collect_results()
    print_summary(results)

    if failed:
        print(f"\n*** {len(failed)} experimento(s) falharam: {failed}")
    else:
        print(f"\nTodos os {len(EXPERIMENTS)} experimentos concluídos com sucesso!")
