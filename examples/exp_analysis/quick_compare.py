#!/usr/bin/env python3
"""
Quick comparison script for the three scenarios
Focuses on the most important metrics in a clean presentation
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Keep the original style
plt.rcParams.update({
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.labelweight": "bold",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "grey",
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.facecolor": "white",
    "legend.fontsize": 10,
    "font.size": 10,
    "figure.dpi": 110,
})

def load_scenario_data(filepath):
    """Load and preprocess scenario data."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    df = df[df["type"] == "inference"].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    t0 = df["timestamp"].min()
    df["elapsed_s"] = (df["timestamp"] - t0).dt.total_seconds()
    df = df[df["elapsed_s"] > 60].copy()  # Skip warm-up
    return df

# Load all three scenarios
script_dir = Path(__file__).parent.absolute()
base_dir = script_dir.parent.parent  # This is the tinymlops-poc directory

if script_dir.name == "exp_analysis":
    # Running from the exp_analysis directory
    data_dir = Path("data")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
else:
    # Running from another directory (e.g., project root)
    data_dir = Path("examples/exp_analysis/data")
    output_dir = Path("examples/exp_analysis/output")
    output_dir.mkdir(parents=True, exist_ok=True)

# Find the files for each scenario - handling both naming patterns
try:
    scenario_files = {
        "Baseline (S0)": data_dir / "performance_scenario0.json",
        "TinyLCM (S1)": data_dir / "performance_scenario1.json",
        "TinyLCM+Drift (S2.1)": data_dir / "performance_scenario2_1.json"
    }

    # Verify files exist
    for name, filepath in scenario_files.items():
        if not filepath.exists():
            print(f"Warning: File {filepath} not found.")
except Exception as e:
    print(f"Error finding scenario files: {e}")

# Load data
scenarios = {}
for name, filepath in scenario_files.items():
    print(f"Loading {name}: {filepath.name}")
    scenarios[name] = load_scenario_data(filepath)

# Create the main comparison figure
fig = plt.figure(figsize=(16, 10))

# Define grid for subplots
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Colors for each scenario
colors = {
    "Baseline (S0)": "#1f77b4",
    "TinyLCM (S1)": "#ff7f0e", 
    "TinyLCM+Drift (S2.1)": "#2ca02c"
}

# 1. Time series comparison (top row - full width)
ax_time = fig.add_subplot(gs[0, :])

for name, df in scenarios.items():
    # Use appropriate column name
    inf_col = "inference_time_ms" if "inference_time_ms" in df.columns else "total_time_ms"
    
    # Plot with some transparency for overlapping
    ax_time.plot(df["elapsed_s"], df[inf_col], 
                label=name, color=colors[name], alpha=0.7, lw=1.5)

ax_time.set_xlabel("Time (seconds)", fontweight="bold")
ax_time.set_ylabel("Inference Time (ms)", fontweight="bold")
ax_time.set_title("Inference Time Over Time", fontweight="bold", fontsize=12)
ax_time.legend(loc="upper right")
ax_time.grid(True, alpha=0.3)

# 2. Average metrics comparison (middle left)
ax_avg = fig.add_subplot(gs[1, 0])

avg_data = []
for name, df in scenarios.items():
    inf_col = "inference_time_ms" if "inference_time_ms" in df.columns else "total_time_ms"
    avg_data.append({
        'Scenario': name,
        'CPU (%)': df["cpu_percent"].mean(),
        'Memory (MB)': df["memory_mb"].mean(),
        'Inference (ms)': df[inf_col].mean()
    })

avg_df = pd.DataFrame(avg_data)

# Bar plot for averages
x = np.arange(len(avg_df))
width = 0.25

ax_avg.bar(x - width, avg_df['CPU (%)'], width, label='CPU (%)', color='skyblue')
ax_avg.bar(x, avg_df['Memory (MB)'], width, label='Memory (MB)', color='lightcoral')
ax_avg.bar(x + width, avg_df['Inference (ms)'], width, label='Inference (ms)', color='lightgreen')

ax_avg.set_xlabel("Scenario", fontweight="bold")
ax_avg.set_ylabel("Average Value", fontweight="bold")
ax_avg.set_title("Average Performance Metrics", fontweight="bold", fontsize=12)
ax_avg.set_xticks(x)
ax_avg.set_xticklabels([s.split()[0] for s in avg_df['Scenario']], rotation=15)
ax_avg.legend()
ax_avg.grid(True, axis='y', alpha=0.3)

# 3. Distribution comparison (middle center and right)
# CPU distribution
ax_cpu_dist = fig.add_subplot(gs[1, 1])
cpu_data = [df["cpu_percent"].values for df in scenarios.values()]
bp1 = ax_cpu_dist.boxplot(cpu_data, patch_artist=True)
for patch, color in zip(bp1['boxes'], colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax_cpu_dist.set_xticklabels([s.split()[0] for s in scenarios.keys()], rotation=15)
ax_cpu_dist.set_ylabel("CPU Usage (%)", fontweight="bold")
ax_cpu_dist.set_title("CPU Usage Distribution", fontweight="bold", fontsize=12)
ax_cpu_dist.grid(True, axis='y', alpha=0.3)

# Memory distribution
ax_mem_dist = fig.add_subplot(gs[1, 2])
mem_data = [df["memory_mb"].values for df in scenarios.values()]
bp2 = ax_mem_dist.boxplot(mem_data, patch_artist=True)
for patch, color in zip(bp2['boxes'], colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax_mem_dist.set_xticklabels([s.split()[0] for s in scenarios.keys()], rotation=15)
ax_mem_dist.set_ylabel("Memory Usage (MB)", fontweight="bold")
ax_mem_dist.set_title("Memory Usage Distribution", fontweight="bold", fontsize=12)
ax_mem_dist.grid(True, axis='y', alpha=0.3)

# 4. Statistical summary table (bottom row)
ax_table = fig.add_subplot(gs[2, :])
ax_table.axis('tight')
ax_table.axis('off')

# Create summary statistics
summary_data = []
for name, df in scenarios.items():
    inf_col = "inference_time_ms" if "inference_time_ms" in df.columns else "total_time_ms"
    
    row = [
        name,
        f"{df['cpu_percent'].mean():.1f} ± {df['cpu_percent'].std():.1f}",
        f"{df['memory_mb'].mean():.1f} ± {df['memory_mb'].std():.1f}",
        f"{df[inf_col].mean():.1f} ± {df[inf_col].std():.1f}",
        f"{df[inf_col].max():.1f}",
        len(df)
    ]
    
    # Add drift info if available
    if "drift_detected" in df.columns:
        drift_count = df["drift_detected"].sum()
        drift_rate = (drift_count / len(df)) * 100
        row.append(f"{drift_count} ({drift_rate:.1f}%)")
    else:
        row.append("N/A")
    
    summary_data.append(row)

# Create table
columns = ['Scenario', 'CPU (%) μ±σ', 'Memory (MB) μ±σ', 'Inference (ms) μ±σ', 
           'Max Inf. (ms)', 'Samples', 'Drift Events']

table = ax_table.table(cellText=summary_data, 
                      colLabels=columns,
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Style the header row
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data) + 1):
    for j in range(len(columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f1f1f2')
        table[(i, j)].set_text_props(weight='normal')

ax_table.set_title("Performance Summary Statistics", fontweight="bold", fontsize=12, pad=20)

# Overall title
fig.suptitle("TinyMLOps Performance Comparison: Baseline vs TinyLCM vs TinyLCM+Drift", 
             fontsize=16, fontweight="bold")

plt.tight_layout()
plt.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# Print summary to console
print("\n=== Performance Impact Summary ===")
for i, (name, df) in enumerate(scenarios.items()):
    if i == 0:
        baseline_inf = df["inference_time_ms"].mean() if "inference_time_ms" in df.columns else df["total_time_ms"].mean()
        baseline_cpu = df["cpu_percent"].mean()
        baseline_mem = df["memory_mb"].mean()
        print(f"{name}: BASELINE")
    else:
        inf_col = "inference_time_ms" if "inference_time_ms" in df.columns else "total_time_ms"
        inf_overhead = ((df[inf_col].mean() - baseline_inf) / baseline_inf) * 100
        cpu_overhead = ((df["cpu_percent"].mean() - baseline_cpu) / baseline_cpu) * 100
        mem_overhead = ((df["memory_mb"].mean() - baseline_mem) / baseline_mem) * 100
        
        print(f"{name}:")
        print(f"  - Inference overhead: {inf_overhead:+.1f}%")
        print(f"  - CPU overhead: {cpu_overhead:+.1f}%")
        print(f"  - Memory overhead: {mem_overhead:+.1f}%")