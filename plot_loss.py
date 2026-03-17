"""Parse log.out and plot training loss curves.

Usage:
    python plot_loss.py workdir/8AA_sim_912_v1/log.out
    python plot_loss.py workdir/8AA_sim_912_v1/log.out --save  # save to PNG instead of showing
"""
import re
import sys
import ast
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_log(log_path):
    metrics = defaultdict(list)
    with open(log_path) as f:
        for line in f:
            # Match any dict-like string in the line
            match = re.search(r"\{[^{}]+\}", line)
            if not match:
                continue
            try:
                d = ast.literal_eval(match.group())
            except (ValueError, SyntaxError):
                continue
            if not isinstance(d, dict):
                continue
            # Only keep entries that have epoch (structured log lines)
            if 'epoch' not in d:
                continue
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    metrics[k].append(v)
    return metrics

def plot(metrics, save=False, out_path="loss_curve.png"):
    # Auto-detect loss keys: train_loss, iter_loss, loss, loss_continuous, etc.
    loss_keys = [k for k in metrics if 'loss' in k.lower()]
    if not loss_keys:
        print("No loss metrics found in log. Available keys:", list(metrics.keys()))
        return

    print(f"Plotting: {loss_keys}")

    fig, axes = plt.subplots(len(loss_keys), 1, figsize=(12, 4 * len(loss_keys)), sharex=True)
    if len(loss_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, loss_keys):
        vals = metrics[key]
        x = list(range(len(vals)))
        ax.plot(x, vals, linewidth=0.5, alpha=0.3, label=f'{key} (raw)')
        # moving average
        window = max(1, len(vals) // 50)
        if window > 1:
            smoothed = [sum(vals[max(0,i-window):i+1]) / len(vals[max(0,i-window):i+1]) for i in range(len(vals))]
            ax.plot(x, smoothed, linewidth=1.5, label=f'{key} (smoothed, w={window})')
        ax.set_ylabel(key)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Log Entry Index')
    axes[0].set_title('Training Loss Curves')
    plt.tight_layout()

    if save:
        plt.savefig(out_path, dpi=150)
        print(f"Saved to {out_path}")
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_loss.py <log.out> [--save]")
        sys.exit(1)
    log_path = sys.argv[1]
    save = "--save" in sys.argv
    metrics = parse_log(log_path)
    print(f"Parsed {len(metrics.get('epoch', []))} log entries")
    print(f"Available metrics: {list(metrics.keys())}")
    for key in sorted(metrics.keys()):
        if 'loss' in key.lower():
            vals = metrics[key]
            print(f"  {key}: {len(vals)} entries, range [{min(vals):.4f}, {max(vals):.4f}], last 5: {[round(v,4) for v in vals[-5:]]}")
    plot(metrics, save=save, out_path=log_path.replace('.out', '_loss.png').replace('.log', '_loss.png'))
