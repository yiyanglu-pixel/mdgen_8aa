"""Parse log.out and plot training loss curves.

Usage:
    python plot_loss.py workdir/8AA_sim_912_v1/log.out
    python plot_loss.py workdir/8AA_sim_912_v1/log.out --save  # save to PNG instead of showing
"""
import re
import sys
import math
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_log(log_path):
    """Parse log.out, handling nan values that ast.literal_eval can't parse."""
    metrics = defaultdict(list)
    nan_counts = defaultdict(int)
    with open(log_path) as f:
        for line in f:
            match = re.search(r"\{[^{}]+\}", line)
            if not match:
                continue
            # Replace nan with None so eval can parse it
            raw = match.group().replace(': nan,', ': None,').replace(': nan}', ': None}')
            try:
                d = eval(raw)
            except Exception:
                continue
            if not isinstance(d, dict) or 'epoch' not in d:
                continue
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    metrics[k].append(v)
                elif v is None:
                    metrics[k].append(float('nan'))
                    nan_counts[k] += 1
    return metrics, nan_counts

def plot(metrics, save=False, out_path="loss_curve.png"):
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
        # Filter out NaN for plotting
        valid = [(i, v) for i, v in enumerate(vals) if not math.isnan(v)]
        if valid:
            x, y = zip(*valid)
            ax.plot(x, y, linewidth=0.5, alpha=0.3, label=f'{key} (raw, {len(valid)} valid)')
            # moving average
            window = max(1, len(y) // 50)
            if window > 1:
                smoothed = [sum(y[max(0,i-window):i+1]) / len(y[max(0,i-window):i+1]) for i in range(len(y))]
                ax.plot(x, smoothed, linewidth=1.5, label=f'{key} (smoothed, w={window})')
        else:
            ax.text(0.5, 0.5, f'{key}: ALL NaN', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, color='red')
        ax.set_ylabel(key)
        ax.legend()
        ax.grid(True, alpha=0.3)

    epochs = metrics.get('epoch', [])
    if epochs:
        axes[-1].set_xlabel(f'Log Entry Index (epoch 0 ~ {int(max(epochs))})')
    else:
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
    metrics, nan_counts = parse_log(log_path)

    epochs = metrics.get('epoch', [])
    print(f"Parsed {len(epochs)} log entries (epoch 0 ~ {int(max(epochs)) if epochs else '?'})")
    print(f"Available metrics: {list(metrics.keys())}")
    print()

    for key in sorted(metrics.keys()):
        if 'loss' in key.lower():
            vals = metrics[key]
            valid = [v for v in vals if not math.isnan(v)]
            nan_ct = nan_counts.get(key, 0)
            total = len(vals)
            print(f"  {key}: {total} entries, {len(valid)} valid, {nan_ct} NaN ({nan_ct*100//max(total,1)}%)")
            if valid:
                print(f"    range [{min(valid):.4f}, {max(valid):.4f}]")
                print(f"    first 5 valid: {[round(v,4) for v in valid[:5]]}")
                print(f"    last 5 valid:  {[round(v,4) for v in valid[-5:]]}")

    plot(metrics, save=save, out_path=log_path.replace('.out', '_loss.png').replace('.log', '_loss.png'))
