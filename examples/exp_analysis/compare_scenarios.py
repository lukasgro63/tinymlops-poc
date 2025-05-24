#!/usr/bin/env python3
"""
Performance Comparison Script for TinyMLOps Scenarios
Compares Scenario 0 (baseline), Scenario 1 (TinyLCM no drift), and Scenario 2.1 (TinyLCM with drift)
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import seaborn as sns

# Set style (keeping the original style)
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

# Color scheme for scenarios
SCENARIO_COLORS = {
    "scenario0": "#1f77b4",  # Blue
    "scenario1": "#ff7f0e",  # Orange
    "scenario2_1": "#2ca02c",  # Green
}

SCENARIO_NAMES = {
    "scenario0": "Baseline TFLite",
    "scenario1": "TinyLCM (no drift)",
    "scenario2_1": "TinyLCM (with drift detection)",
}


def load_performance_data(data_dir: Path, pattern: str) -> pd.DataFrame:
    """Load performance data from JSON lines files."""
    files = list(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    
    # Use the most recent file
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    print(f"Loading {pattern}: {latest_file.name}")
    
    # Read JSON lines
    data = []
    with open(latest_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Filter only inference records
    df = df[df["type"] == "inference"].copy()
    
    # Convert timestamp and calculate elapsed time
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    t0 = df["timestamp"].min()
    df["elapsed_s"] = (df["timestamp"] - t0).dt.total_seconds()
    
    # Skip first minute for warm-up
    df = df[df["elapsed_s"] > 60].copy()
    
    return df


def create_comparison_summary(scenarios_data: dict) -> pd.DataFrame:
    """Create a summary comparison table."""
    summary_data = []
    
    for scenario, df in scenarios_data.items():
        # Handle different column names
        if "inference_time_ms" in df.columns:
            inference_col = "inference_time_ms"
        elif "total_time_ms" in df.columns:
            inference_col = "total_time_ms"
        else:
            continue
            
        summary = {
            "Scenario": SCENARIO_NAMES[scenario],
            "Avg CPU (%)": df["cpu_percent"].mean(),
            "Std CPU (%)": df["cpu_percent"].std(),
            "Max CPU (%)": df["cpu_percent"].max(),
            "Avg Memory (MB)": df["memory_mb"].mean(),
            "Std Memory (MB)": df["memory_mb"].std(),
            "Max Memory (MB)": df["memory_mb"].max(),
            "Avg Inference (ms)": df[inference_col].mean(),
            "Std Inference (ms)": df[inference_col].std(),
            "Max Inference (ms)": df[inference_col].max(),
            "Total Inferences": len(df),
        }
        
        # Add drift-specific metrics if available
        if "drift_detected" in df.columns:
            summary["Drift Events"] = df["drift_detected"].sum()
            summary["Drift Rate (%)"] = (df["drift_detected"].sum() / len(df)) * 100
        
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)


def plot_combined_metrics(scenarios_data: dict, save_path: Path = None):
    """Create a combined plot with all metrics for all scenarios."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # CPU Usage
    ax_cpu = axes[0]
    for scenario, df in scenarios_data.items():
        ax_cpu.plot(df["elapsed_s"], df["cpu_percent"], 
                   color=SCENARIO_COLORS[scenario], 
                   label=SCENARIO_NAMES[scenario],
                   alpha=0.8, lw=1.5)
    
    ax_cpu.set_ylabel("CPU Usage (%)", fontweight="bold")
    ax_cpu.grid(True, alpha=0.3)
    ax_cpu.legend(loc="upper right")
    ax_cpu.set_title("CPU Usage Comparison", fontweight="bold", pad=10)
    
    # Memory Usage
    ax_mem = axes[1]
    for scenario, df in scenarios_data.items():
        ax_mem.plot(df["elapsed_s"], df["memory_mb"], 
                   color=SCENARIO_COLORS[scenario], 
                   label=SCENARIO_NAMES[scenario],
                   alpha=0.8, lw=1.5)
    
    ax_mem.set_ylabel("Memory Usage (MB)", fontweight="bold")
    ax_mem.grid(True, alpha=0.3)
    ax_mem.legend(loc="upper right")
    ax_mem.set_title("Memory Usage Comparison", fontweight="bold", pad=10)
    
    # Inference Time
    ax_inf = axes[2]
    for scenario, df in scenarios_data.items():
        # Handle different column names
        if "inference_time_ms" in df.columns:
            inference_col = "inference_time_ms"
        elif "total_time_ms" in df.columns:
            inference_col = "total_time_ms"
        else:
            continue
            
        ax_inf.plot(df["elapsed_s"], df[inference_col], 
                   color=SCENARIO_COLORS[scenario], 
                   label=SCENARIO_NAMES[scenario],
                   alpha=0.6, lw=1.0, marker='o', markersize=2)
    
    ax_inf.set_ylabel("Inference Time (ms)", fontweight="bold")
    ax_inf.set_xlabel("Time (seconds)", fontweight="bold")
    ax_inf.grid(True, alpha=0.3)
    ax_inf.legend(loc="upper right")
    ax_inf.set_title("Inference Time Comparison", fontweight="bold", pad=10)
    
    plt.suptitle("TinyMLOps Performance Comparison", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_individual_scenarios(scenarios_data: dict, save_dir: Path = None):
    """Create individual plots for each scenario with all metrics."""
    for scenario, df in scenarios_data.items():
        fig, ax_cpu = plt.subplots(figsize=(10, 4.5))
        
        # CPU plot
        ax_cpu.plot(df["elapsed_s"], df["cpu_percent"], 
                   color="black", lw=1.0, label="CPU (%)")
        ax_cpu.set_xlabel("Time (s)", fontweight="bold")
        ax_cpu.set_ylabel("CPU (%)", fontweight="bold")
        ax_cpu.grid(True, axis="y", linestyle="--", color="grey", lw=0.5)
        
        # Memory on secondary y-axis
        ax_mem = ax_cpu.twinx()
        ax_mem.plot(df["elapsed_s"], df["memory_mb"], 
                   color="grey", ls="--", lw=1.0, label="Memory (MB)")
        ax_mem.set_ylabel("Memory (MB)", fontweight="bold")
        
        # Inference time on tertiary y-axis
        ax_inf = ax_cpu.twinx()
        ax_inf.spines.right.set_position(("axes", 1.14))
        
        # Handle different column names
        if "inference_time_ms" in df.columns:
            inference_col = "inference_time_ms"
        elif "total_time_ms" in df.columns:
            inference_col = "total_time_ms"
        else:
            inference_col = None
            
        if inference_col:
            ax_inf.plot(df["elapsed_s"], df[inference_col], 
                       linestyle="", marker="s", ms=5, 
                       color="lightgrey", label="Inference time (ms)")
            ax_inf.set_ylabel("Inference time (ms)", fontweight="bold")
        
        # Combined legend
        lines = ax_cpu.get_lines() + ax_mem.get_lines()
        if inference_col:
            lines += ax_inf.get_lines()
        labels = [l.get_label() for l in lines]
        ax_cpu.legend(lines, labels, loc="upper right")
        
        plt.title(f"{SCENARIO_NAMES[scenario]} — Runtime Metrics", fontweight="bold")
        plt.tight_layout()
        
        if save_dir:
            save_path = save_dir / f"{scenario}_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def plot_boxplot_comparison(scenarios_data: dict, save_path: Path = None):
    """Create box plots for metric distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare data for boxplots
    cpu_data = []
    memory_data = []
    inference_data = []
    labels = []
    
    for scenario, df in scenarios_data.items():
        cpu_data.append(df["cpu_percent"].values)
        memory_data.append(df["memory_mb"].values)
        
        # Handle different column names
        if "inference_time_ms" in df.columns:
            inference_data.append(df["inference_time_ms"].values)
        elif "total_time_ms" in df.columns:
            inference_data.append(df["total_time_ms"].values)
        else:
            inference_data.append(np.array([]))
            
        labels.append(SCENARIO_NAMES[scenario])
    
    # CPU boxplot
    bp1 = axes[0].boxplot(cpu_data, labels=labels, patch_artist=True)
    axes[0].set_ylabel("CPU Usage (%)", fontweight="bold")
    axes[0].set_title("CPU Usage Distribution", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    
    # Memory boxplot
    bp2 = axes[1].boxplot(memory_data, labels=labels, patch_artist=True)
    axes[1].set_ylabel("Memory Usage (MB)", fontweight="bold")
    axes[1].set_title("Memory Usage Distribution", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    
    # Inference time boxplot
    bp3 = axes[2].boxplot(inference_data, labels=labels, patch_artist=True)
    axes[2].set_ylabel("Inference Time (ms)", fontweight="bold")
    axes[2].set_title("Inference Time Distribution", fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    
    # Color the boxes
    for bp, color_key in zip([bp1, bp2, bp3], ["scenario0", "scenario1", "scenario2_1"]):
        for patch, scenario in zip(bp['boxes'], ["scenario0", "scenario1", "scenario2_1"]):
            patch.set_facecolor(SCENARIO_COLORS[scenario])
            patch.set_alpha(0.7)
    
    # Rotate x-labels for better readability
    for ax in axes:
        ax.tick_params(axis='x', rotation=15)
    
    plt.suptitle("Performance Metrics Distribution Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_drift_analysis(scenarios_data: dict, save_path: Path = None):
    """Analyze and plot drift detection impact (for scenario2_1)."""
    if "scenario2_1" not in scenarios_data:
        print("No drift detection data available")
        return
        
    df = scenarios_data["scenario2_1"]
    
    if "drift_detected" not in df.columns:
        print("No drift detection information in scenario2_1 data")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Drift events over time
    ax_drift = axes[0]
    drift_events = df[df["drift_detected"] == True]
    
    ax_drift.scatter(drift_events["elapsed_s"], 
                    [1] * len(drift_events), 
                    color="red", s=100, marker='v', 
                    label=f"Drift Events (n={len(drift_events)})")
    
    # Add inference time with drift overlay
    if "total_time_ms" in df.columns:
        inference_col = "total_time_ms"
    else:
        inference_col = "inference_time_ms"
        
    ax_drift.plot(df["elapsed_s"], df[inference_col], 
                 color="grey", alpha=0.5, lw=1.0, 
                 label="Inference Time")
    
    ax_drift.set_ylabel("Inference Time (ms)", fontweight="bold")
    ax_drift.set_title("Drift Detection Events and Inference Time", fontweight="bold")
    ax_drift.legend()
    ax_drift.grid(True, alpha=0.3)
    
    # Resource usage with drift overlay
    ax_resource = axes[1]
    ax_resource.plot(df["elapsed_s"], df["cpu_percent"], 
                    color="blue", alpha=0.7, lw=1.5, 
                    label="CPU Usage")
    
    # Highlight drift regions
    for _, event in drift_events.iterrows():
        ax_resource.axvline(x=event["elapsed_s"], 
                          color="red", alpha=0.3, linestyle="--")
    
    ax_resource.set_ylabel("CPU Usage (%)", fontweight="bold")
    ax_resource.set_xlabel("Time (seconds)", fontweight="bold")
    ax_resource.set_title("CPU Usage with Drift Events", fontweight="bold")
    ax_resource.legend()
    ax_resource.grid(True, alpha=0.3)
    
    plt.suptitle("Drift Detection Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main analysis function."""
    # Define data directory
    data_dir = Path("examples/exp_analysis/data")
    output_dir = Path("examples/exp_analysis/output")
    output_dir.mkdir(exist_ok=True)
    
    # Load data for all scenarios
    scenarios_data = {}
    
    try:
        # Scenario 0 - Baseline
        scenarios_data["scenario0"] = load_performance_data(
            data_dir, "performance_scenario0_*.json"
        )
        
        # Scenario 1 - TinyLCM no drift
        scenarios_data["scenario1"] = load_performance_data(
            data_dir, "performance_scenario1_*.json"
        )
        
        # Scenario 2.1 - TinyLCM with drift
        scenarios_data["scenario2_1"] = load_performance_data(
            data_dir, "performance_scenario2_1_*.json"
        )
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure all performance data files are in the data directory")
        return
    
    # Create summary comparison
    print("\n=== Performance Summary Comparison ===")
    summary_df = create_comparison_summary(scenarios_data)
    print(summary_df.round(2).to_string())
    
    # Save summary to CSV
    summary_df.to_csv(output_dir / "performance_summary.csv", index=False)
    
    # Generate plots
    print("\n=== Generating Plots ===")
    
    # 1. Combined metrics plot
    print("Creating combined metrics plot...")
    plot_combined_metrics(scenarios_data, output_dir / "combined_metrics.png")
    
    # 2. Individual scenario plots
    print("Creating individual scenario plots...")
    plot_individual_scenarios(scenarios_data, output_dir)
    
    # 3. Boxplot comparison
    print("Creating boxplot comparison...")
    plot_boxplot_comparison(scenarios_data, output_dir / "boxplot_comparison.png")
    
    # 4. Drift analysis (if available)
    print("Creating drift analysis...")
    plot_drift_analysis(scenarios_data, output_dir / "drift_analysis.png")
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()