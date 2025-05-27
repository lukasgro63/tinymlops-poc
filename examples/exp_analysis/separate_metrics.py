#!/usr/bin/env python3
"""
Create separate charts for each metric across all scenarios
Clean, publication-ready plots
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Style settings
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
    "legend.fontsize": 11,
    "font.size": 11,
    "figure.dpi": 150,
})

# Load data function
def load_scenario_data(filepath):
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

# Setup
script_dir = Path(__file__).parent.absolute()
base_dir = script_dir.parent.parent  # This is the tinymlops-poc directory

if script_dir.name == "exp_analysis":
    # Running from the exp_analysis directory
    data_dir = Path("data")
    output_dir = Path("output")
else:
    # Running from another directory (e.g., project root)
    data_dir = Path("examples/exp_analysis/data")
    output_dir = Path("examples/exp_analysis/output")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Find and load files
try:
    scenario_files = {
        "Baseline": data_dir / "performance_scenario0.json",
        "TinyLCM": data_dir / "performance_scenario1.json", 
        "TinyLCM+Drift": data_dir / "performance_scenario2_1.json"
    }
    
    # Verify files exist
    for name, filepath in scenario_files.items():
        if not filepath.exists():
            print(f"Warning: File {filepath} not found.")
except Exception as e:
    print(f"Error finding scenario files: {e}")

scenarios = {}
for name, filepath in scenario_files.items():
    print(f"Loading {name}: {filepath.name}")
    scenarios[name] = load_scenario_data(filepath)

# Colors
colors = {
    "Baseline": "#3498db",      # Blue
    "TinyLCM": "#e74c3c",       # Red
    "TinyLCM+Drift": "#2ecc71"  # Green
}

# 1. CPU Usage Chart
plt.figure(figsize=(10, 6))
for name, df in scenarios.items():
    # Calculate rolling mean for smoother visualization
    window_size = 10
    cpu_smooth = df["cpu_percent"].rolling(window=window_size, center=True).mean()
    
    plt.plot(df["elapsed_s"], cpu_smooth, 
             label=name, color=colors[name], linewidth=2, alpha=0.8)
    
    # Add shaded area for std deviation
    cpu_std = df["cpu_percent"].rolling(window=window_size, center=True).std()
    plt.fill_between(df["elapsed_s"], 
                     cpu_smooth - cpu_std, 
                     cpu_smooth + cpu_std,
                     color=colors[name], alpha=0.2)

plt.xlabel("Time (seconds)", fontsize=12, fontweight="bold")
plt.ylabel("CPU Usage (%)", fontsize=12, fontweight="bold")
plt.title("CPU Usage Comparison", fontsize=14, fontweight="bold", pad=15)
plt.legend(loc="best", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "cpu_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Memory Usage Chart
plt.figure(figsize=(10, 6))
for name, df in scenarios.items():
    # Memory is usually more stable, so less smoothing
    window_size = 5
    mem_smooth = df["memory_mb"].rolling(window=window_size, center=True).mean()
    
    plt.plot(df["elapsed_s"], mem_smooth, 
             label=name, color=colors[name], linewidth=2, alpha=0.8)

plt.xlabel("Time (seconds)", fontsize=12, fontweight="bold")
plt.ylabel("Memory Usage (MB)", fontsize=12, fontweight="bold")
plt.title("Memory Usage Comparison", fontsize=14, fontweight="bold", pad=15)
plt.legend(loc="best", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "memory_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Inference Time Chart
plt.figure(figsize=(10, 6))
for name, df in scenarios.items():
    # Determine the correct column
    inf_col = "inference_time_ms" if "inference_time_ms" in df.columns else "total_time_ms"
    
    # Use scatter plot with transparency for inference times
    plt.scatter(df["elapsed_s"], df[inf_col], 
               label=name, color=colors[name], alpha=0.5, s=20)
    
    # Add rolling mean line
    window_size = 20
    inf_smooth = df[inf_col].rolling(window=window_size, center=True).mean()
    plt.plot(df["elapsed_s"], inf_smooth, 
             color=colors[name], linewidth=2, alpha=0.9)

plt.xlabel("Time (seconds)", fontsize=12, fontweight="bold")
plt.ylabel("Inference Time (ms)", fontsize=12, fontweight="bold")
plt.title("Inference Time Comparison", fontsize=14, fontweight="bold", pad=15)
plt.legend(loc="best", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "inference_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Overhead Analysis Bar Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Calculate overheads
baseline_metrics = {
    "cpu": scenarios["Baseline"]["cpu_percent"].mean(),
    "memory": scenarios["Baseline"]["memory_mb"].mean(),
    "inference": scenarios["Baseline"]["inference_time_ms"].mean() if "inference_time_ms" in scenarios["Baseline"].columns else scenarios["Baseline"]["total_time_ms"].mean()
}

overhead_data = []
for name in ["TinyLCM", "TinyLCM+Drift"]:
    df = scenarios[name]
    inf_col = "inference_time_ms" if "inference_time_ms" in df.columns else "total_time_ms"
    
    overheads = {
        "Scenario": name,
        "CPU": ((df["cpu_percent"].mean() - baseline_metrics["cpu"]) / baseline_metrics["cpu"]) * 100,
        "Memory": ((df["memory_mb"].mean() - baseline_metrics["memory"]) / baseline_metrics["memory"]) * 100,
        "Inference": ((df[inf_col].mean() - baseline_metrics["inference"]) / baseline_metrics["inference"]) * 100
    }
    overhead_data.append(overheads)

overhead_df = pd.DataFrame(overhead_data)

# Bar chart for overheads
x = np.arange(len(overhead_df))
width = 0.25

bars1 = ax1.bar(x - width, overhead_df['CPU'], width, label='CPU', color='#3498db')
bars2 = ax1.bar(x, overhead_df['Memory'], width, label='Memory', color='#e74c3c')
bars3 = ax1.bar(x + width, overhead_df['Inference'], width, label='Inference', color='#2ecc71')

ax1.set_ylabel("Overhead (%)", fontsize=12, fontweight="bold")
ax1.set_title("Performance Overhead vs Baseline", fontsize=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(overhead_df['Scenario'])
ax1.legend()
ax1.grid(True, axis='y', alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9)

# 5. Absolute values comparison
metrics_comparison = []
for name, df in scenarios.items():
    inf_col = "inference_time_ms" if "inference_time_ms" in df.columns else "total_time_ms"
    metrics_comparison.append({
        'Scenario': name,
        'CPU (%)': df["cpu_percent"].mean(),
        'Memory (MB)': df["memory_mb"].mean(),
        'Inference (ms)': df[inf_col].mean()
    })

comp_df = pd.DataFrame(metrics_comparison)

# Grouped bar chart
x2 = np.arange(3)  # Three metrics
width2 = 0.25

for i, scenario in enumerate(comp_df['Scenario']):
    values = [comp_df.iloc[i]['CPU (%)'], 
              comp_df.iloc[i]['Memory (MB)'], 
              comp_df.iloc[i]['Inference (ms)']]
    
    bars = ax2.bar(x2 + i*width2, values, width2, 
                    label=scenario, color=list(colors.values())[i])
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax2.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)

ax2.set_ylabel("Value", fontsize=12, fontweight="bold")
ax2.set_title("Absolute Performance Metrics", fontsize=13, fontweight="bold")
ax2.set_xticks(x2 + width2)
ax2.set_xticklabels(['CPU (%)', 'Memory (MB)', 'Inference (ms)'])
ax2.legend()
ax2.grid(True, axis='y', alpha=0.3)

plt.suptitle("TinyMLOps Performance Analysis", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(output_dir / "performance_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# Print detailed statistics
print("\n=== Detailed Performance Statistics ===\n")
for name, df in scenarios.items():
    print(f"{name}:")
    inf_col = "inference_time_ms" if "inference_time_ms" in df.columns else "total_time_ms"
    
    print(f"  CPU Usage: {df['cpu_percent'].mean():.2f}% (±{df['cpu_percent'].std():.2f})")
    print(f"  Memory Usage: {df['memory_mb'].mean():.2f} MB (±{df['memory_mb'].std():.2f})")
    print(f"  Inference Time: {df[inf_col].mean():.2f} ms (±{df[inf_col].std():.2f})")
    print(f"  Max Inference: {df[inf_col].max():.2f} ms")
    print(f"  95th Percentile: {df[inf_col].quantile(0.95):.2f} ms")
    
    if "drift_detected" in df.columns:
        drift_count = df["drift_detected"].sum()
        print(f"  Drift Events: {drift_count} ({(drift_count/len(df)*100):.2f}%)")
    print()