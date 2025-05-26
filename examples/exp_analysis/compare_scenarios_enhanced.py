#!/usr/bin/env python3
"""
Enhanced Performance Comparison Script for TinyMLOps Scenarios
Compares Scenario 0 (baseline), Scenario 1 (TinyLCM no drift), and Scenario 2.1 (TinyLCM with drift)
With statistical tests and publication-ready grayscale styling
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import seaborn as sns
from scipy import stats
import pymannkendall as mk
import pingouin as pg
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready grayscale style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.linewidth": 1.0,
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "#E0E0E0",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.facecolor": "white",
    "legend.framealpha": 1.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
})

# Grayscale color scheme for scenarios
SCENARIO_COLORS = {
    "scenario0": "#000000",      # Black
    "scenario1": "#606060",      # Dark gray
    "scenario2_1": "#A0A0A0",    # Light gray
}

SCENARIO_MARKERS = {
    "scenario0": "o",
    "scenario1": "s",
    "scenario2_1": "^",
}

SCENARIO_LINESTYLES = {
    "scenario0": "-",
    "scenario1": "--",
    "scenario2_1": ":",
}

SCENARIO_NAMES = {
    "scenario0": "Baseline TFLite",
    "scenario1": "TinyLCM (no drift)",
    "scenario2_1": "TinyLCM (drift detection)",
}


def load_performance_data_fixed_samples(data_dir: Path, filename: str, num_samples: int) -> pd.DataFrame:
    """Load exactly num_samples from the end of the file."""
    file_path = data_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"No file found: {filename}")
    
    print(f"Loading {filename}")
    
    # Read JSON lines
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Filter only inference records
    df = df[df["type"] == "inference"].copy()
    
    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Take exactly num_samples from the end
    df = df.tail(num_samples).copy()
    
    # Calculate actual time span
    time_span = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
    
    # Recalculate elapsed_s to start from 0
    min_time = df["timestamp"].min()
    df["elapsed_s"] = (df["timestamp"] - min_time).dt.total_seconds()
    
    print(f"  Loaded {len(df)} samples covering {time_span:.1f} seconds")
    
    return df


def load_performance_data(data_dir: Path, filename: str, duration_seconds: int = 60) -> pd.DataFrame:
    """Load the last N seconds of data from each file."""
    file_path = data_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"No file found: {filename}")
    
    print(f"Loading {filename}")
    
    # Read JSON lines
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Filter only inference records
    df = df[df["type"] == "inference"].copy()
    
    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Get the last timestamp in the file
    last_timestamp = df["timestamp"].max()
    
    # Go back 60 seconds from the last timestamp
    cutoff_timestamp = last_timestamp - pd.Timedelta(seconds=duration_seconds)
    
    # Keep only data from the last 60 seconds
    df = df[df["timestamp"] >= cutoff_timestamp].copy()
    
    # Recalculate elapsed_s to start from 0
    min_time = df["timestamp"].min()
    df["elapsed_s"] = (df["timestamp"] - min_time).dt.total_seconds()
    
    print(f"  Total samples in file: {len(data)}")
    print(f"  Samples in last {duration_seconds} seconds: {len(df)}")
    
    return df


def perform_statistical_tests(scenarios_data: dict) -> pd.DataFrame:
    """Perform comprehensive statistical tests as per the table.
    
    Note on pairing approach:
    - Scenarios were run at different times, so we can't pair by timestamp
    - All data is from the last 60 seconds of each experiment
    - For paired tests (Wilcoxon, TOST), we truncate to minimum length and pair by sample order
    - This is valid since all scenarios measure the same 60-second workload
    - Different sampling rates reflect the actual performance impact of each scenario
    """
    results = []
    
    # Truncate to the minimum length for paired tests
    
    # Extract metrics for each scenario
    metrics = {}
    min_length = min(len(df) for df in scenarios_data.values())
    
    for scenario, df in scenarios_data.items():
        # Handle different column names
        if "inference_time_ms" in df.columns:
            inference_col = "inference_time_ms"
        elif "total_time_ms" in df.columns:
            inference_col = "total_time_ms"
        else:
            inference_col = None
        
        # Truncate to minimum length for fair comparison
        df_truncated = df.head(min_length)
            
        metrics[scenario] = {
            "cpu": df_truncated["cpu_percent"].values,
            "memory": df_truncated["memory_mb"].values,
            "latency": df_truncated[inference_col].values if inference_col else np.array([]),
            "time": df_truncated["elapsed_s"].values
        }
    
    # 1. Normality tests
    print("\n=== 1. Normality Tests (Shapiro-Wilk) ===")
    for scenario in ["scenario0", "scenario1", "scenario2_1"]:
        for metric_name, metric_values in [("CPU", metrics[scenario]["cpu"]), 
                                          ("Memory", metrics[scenario]["memory"]),
                                          ("Latency", metrics[scenario]["latency"])]:
            if len(metric_values) > 3:  # Shapiro-Wilk requires at least 3 samples
                stat, p_value = stats.shapiro(metric_values)
                normal = "Normal" if p_value > 0.05 else "Non-normal"
                results.append({
                    "Test": "Shapiro-Wilk",
                    "Scenario": SCENARIO_NAMES[scenario],
                    "Metric": metric_name,
                    "Statistic": stat,
                    "p-value": p_value,
                    "Result": normal
                })
                print(f"{SCENARIO_NAMES[scenario]} - {metric_name}: p={p_value:.4f} ({normal})")
    
    # 2. Paired comparisons (Wilcoxon signed-rank test)
    print("\n=== 2. Paired Comparisons (Wilcoxon signed-rank) ===")
    print("Note: Using sample order pairing (all from last 60s, truncated to equal length)")
    
    for comparison in [("scenario1", "scenario0"), ("scenario2_1", "scenario0")]:
        s1, s0 = comparison
        print(f"\n{SCENARIO_NAMES[s1]} vs {SCENARIO_NAMES[s0]}:")
        
        for metric_name, metric_key in [("CPU", "cpu"), ("Memory", "memory"), ("Latency", "latency")]:
            if len(metrics[s1][metric_key]) > 0 and len(metrics[s0][metric_key]) > 0:
                # Data already truncated to min_length in metrics extraction
                data1 = metrics[s1][metric_key]
                data0 = metrics[s0][metric_key]
                
                # Wilcoxon test on paired differences
                try:
                    stat, p_value = stats.wilcoxon(data1, data0, alternative='two-sided')
                    significant = "Significant" if p_value < 0.05 else "Not significant"
                    
                    # Calculate median difference for context
                    median_diff = np.median(data1 - data0)
                    
                    results.append({
                        "Test": "Wilcoxon",
                        "Comparison": f"{SCENARIO_NAMES[s1]} vs {SCENARIO_NAMES[s0]}",
                        "Metric": metric_name,
                        "Statistic": stat,
                        "p-value": p_value,
                        "Median Diff": median_diff,
                        "Result": significant
                    })
                    print(f"  {metric_name}: p={p_value:.4f} ({significant}), median diff={median_diff:.2f}")
                except Exception as e:
                    print(f"  {metric_name}: Error - {str(e)}")
    
    # 3. Effect size (Cliff's Delta)
    print("\n=== 3. Effect Size (Cliff's Delta) ===")
    for comparison in [("scenario1", "scenario0"), ("scenario2_1", "scenario0")]:
        s1, s0 = comparison
        print(f"\n{SCENARIO_NAMES[s1]} vs {SCENARIO_NAMES[s0]}:")
        
        for metric_name, metric_key in [("CPU", "cpu"), ("Memory", "memory"), ("Latency", "latency")]:
            if len(metrics[s1][metric_key]) > 0 and len(metrics[s0][metric_key]) > 0:
                # Compute Cliff's Delta manually
                # Cliff's delta = (sum(x1 > x2) - sum(x1 < x2)) / (n1 * n2)
                x1 = metrics[s1][metric_key]
                x2 = metrics[s0][metric_key]
                
                # Count how many times values in x1 are greater than values in x2
                greater = 0
                less = 0
                for val1 in x1:
                    for val2 in x2:
                        if val1 > val2:
                            greater += 1
                        elif val1 < val2:
                            less += 1
                
                delta = (greater - less) / (len(x1) * len(x2))
                
                # Interpret effect size
                abs_delta = abs(delta)
                if abs_delta < 0.147:
                    interpretation = "negligible"
                elif abs_delta < 0.33:
                    interpretation = "small"
                elif abs_delta < 0.474:
                    interpretation = "medium"
                else:
                    interpretation = "large"
                
                results.append({
                    "Test": "Cliff's Delta",
                    "Comparison": f"{SCENARIO_NAMES[s1]} vs {SCENARIO_NAMES[s0]}",
                    "Metric": metric_name,
                    "Delta": delta,
                    "Interpretation": interpretation
                })
                print(f"  {metric_name}: δ={delta:.3f} ({interpretation})")
    
    # 4. TOST (Two One-Sided Tests) for equivalence
    print("\n=== 4. TOST Equivalence Tests ===")
    # Define SESOI (Smallest Effect Size of Interest)
    sesoi_cpu = 5.0  # ±5 percentage points
    sesoi_latency = 20.0  # ±20 ms
    
    for comparison in [("scenario1", "scenario0"), ("scenario2_1", "scenario0")]:
        s1, s0 = comparison
        print(f"\n{SCENARIO_NAMES[s1]} vs {SCENARIO_NAMES[s0]}:")
        
        # CPU equivalence
        if len(metrics[s1]["cpu"]) > 0 and len(metrics[s0]["cpu"]) > 0:
            # Data already truncated to min_length
            diff_cpu = metrics[s1]["cpu"] - metrics[s0]["cpu"]
            
            # TOST: Test if mean difference is within [-sesoi, +sesoi]
            # H01: μ_diff ≤ -sesoi (test with greater)
            # H02: μ_diff ≥ +sesoi (test with less)
            t1, p1 = stats.ttest_1samp(diff_cpu, -sesoi_cpu, alternative='greater')
            t2, p2 = stats.ttest_1samp(diff_cpu, sesoi_cpu, alternative='less')
            p_tost = max(p1, p2)  # Both must be < 0.05 for equivalence
            
            mean_diff = np.mean(diff_cpu)
            equivalent = "Equivalent" if p_tost < 0.05 else "Not equivalent"
            
            results.append({
                "Test": "TOST",
                "Comparison": f"{SCENARIO_NAMES[s1]} vs {SCENARIO_NAMES[s0]}",
                "Metric": "CPU",
                "SESOI": f"±{sesoi_cpu}pp",
                "Mean Diff": mean_diff,
                "p-value": p_tost,
                "Result": equivalent
            })
            print(f"  CPU (±{sesoi_cpu}pp): mean diff={mean_diff:.2f}, p={p_tost:.4f} ({equivalent})")
        
        # Latency equivalence
        if len(metrics[s1]["latency"]) > 0 and len(metrics[s0]["latency"]) > 0:
            diff_latency = metrics[s1]["latency"] - metrics[s0]["latency"]
            
            t1, p1 = stats.ttest_1samp(diff_latency, -sesoi_latency, alternative='greater')
            t2, p2 = stats.ttest_1samp(diff_latency, sesoi_latency, alternative='less')
            p_tost = max(p1, p2)
            
            mean_diff = np.mean(diff_latency)
            equivalent = "Equivalent" if p_tost < 0.05 else "Not equivalent"
            
            results.append({
                "Test": "TOST",
                "Comparison": f"{SCENARIO_NAMES[s1]} vs {SCENARIO_NAMES[s0]}",
                "Metric": "Latency",
                "SESOI": f"±{sesoi_latency}ms",
                "Mean Diff": mean_diff,
                "p-value": p_tost,
                "Result": equivalent
            })
            print(f"  Latency (±{sesoi_latency}ms): mean diff={mean_diff:.2f}, p={p_tost:.4f} ({equivalent})")
    
    # 5. Memory trend test (Mann-Kendall)
    print("\n=== 5. Memory Trend Tests (Mann-Kendall) ===")
    # Use full data (not truncated) for trend analysis
    for scenario in ["scenario0", "scenario1", "scenario2_1"]:
        df = scenarios_data[scenario]
        if len(df) > 10:  # Need sufficient data points
            trend_result = mk.original_test(df["memory_mb"].values)
            
            trend = "No trend" if trend_result.p > 0.05 else f"{trend_result.trend} trend"
            
            # Calculate memory change over time
            mem_start = df["memory_mb"].iloc[:5].mean()
            mem_end = df["memory_mb"].iloc[-5:].mean()
            mem_change = mem_end - mem_start
            
            results.append({
                "Test": "Mann-Kendall",
                "Scenario": SCENARIO_NAMES[scenario],
                "Metric": "Memory",
                "Statistic": trend_result.z,
                "p-value": trend_result.p,
                "Memory Change (MB)": mem_change,
                "Result": trend
            })
            print(f"{SCENARIO_NAMES[scenario]}: p={trend_result.p:.4f} ({trend}), change={mem_change:.2f} MB")
    
    return pd.DataFrame(results)


def plot_grayscale_comparison(scenarios_data: dict, save_path: Path = None):
    """Create grayscale comparison plots without titles."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    # CPU Usage
    ax_cpu = axes[0]
    for scenario, df in scenarios_data.items():
        # Downsample for cleaner visualization
        step = max(1, len(df) // 500)
        df_plot = df.iloc[::step]
        
        ax_cpu.plot(df_plot["elapsed_s"], df_plot["cpu_percent"], 
                   color=SCENARIO_COLORS[scenario], 
                   linestyle=SCENARIO_LINESTYLES[scenario],
                   label=SCENARIO_NAMES[scenario],
                   alpha=0.8, lw=1.5)
    
    ax_cpu.set_ylabel("CPU Usage (%)")
    ax_cpu.grid(True, alpha=0.5)
    ax_cpu.legend(loc="upper right", frameon=True)
    
    # Memory Usage
    ax_mem = axes[1]
    for scenario, df in scenarios_data.items():
        step = max(1, len(df) // 500)
        df_plot = df.iloc[::step]
        
        ax_mem.plot(df_plot["elapsed_s"], df_plot["memory_mb"], 
                   color=SCENARIO_COLORS[scenario],
                   linestyle=SCENARIO_LINESTYLES[scenario],
                   label=SCENARIO_NAMES[scenario],
                   alpha=0.8, lw=1.5)
    
    ax_mem.set_ylabel("Memory Usage (MB)")
    ax_mem.grid(True, alpha=0.5)
    
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
        
        # Scatter plot with downsampling
        step = max(1, len(df) // 200)
        df_plot = df.iloc[::step]
        
        ax_inf.scatter(df_plot["elapsed_s"], df_plot[inference_col], 
                      color=SCENARIO_COLORS[scenario],
                      marker=SCENARIO_MARKERS[scenario],
                      label=SCENARIO_NAMES[scenario],
                      alpha=0.6, s=20)
    
    ax_inf.set_ylabel("Inference Time (ms)")
    ax_inf.set_xlabel("Time (seconds)")
    ax_inf.grid(True, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_grayscale_boxplots(scenarios_data: dict, save_path: Path = None):
    """Create grayscale box plots for metric distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Prepare data for boxplots
    cpu_data = []
    memory_data = []
    inference_data = []
    labels = []
    
    for scenario in ["scenario0", "scenario1", "scenario2_1"]:
        if scenario in scenarios_data:
            df = scenarios_data[scenario]
            cpu_data.append(df["cpu_percent"].values)
            memory_data.append(df["memory_mb"].values)
            
            # Handle different column names
            if "inference_time_ms" in df.columns:
                inference_data.append(df["inference_time_ms"].values)
            elif "total_time_ms" in df.columns:
                inference_data.append(df["total_time_ms"].values)
            else:
                inference_data.append(np.array([]))
            
            labels.append(SCENARIO_NAMES[scenario].replace(" ", "\n"))
    
    # CPU boxplot
    bp1 = axes[0].boxplot(cpu_data, labels=labels, patch_artist=True, 
                         boxprops=dict(facecolor='#D0D0D0', linewidth=1.5),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5),
                         medianprops=dict(linewidth=2, color='black'))
    axes[0].set_ylabel("CPU Usage (%)")
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Memory boxplot
    bp2 = axes[1].boxplot(memory_data, labels=labels, patch_artist=True,
                         boxprops=dict(facecolor='#D0D0D0', linewidth=1.5),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5),
                         medianprops=dict(linewidth=2, color='black'))
    axes[1].set_ylabel("Memory Usage (MB)")
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Inference time boxplot
    bp3 = axes[2].boxplot(inference_data, labels=labels, patch_artist=True,
                         boxprops=dict(facecolor='#D0D0D0', linewidth=1.5),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5),
                         medianprops=dict(linewidth=2, color='black'))
    axes[2].set_ylabel("Inference Time (ms)")
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Color the boxes with different gray shades
    for bp_idx, bp in enumerate([bp1, bp2, bp3]):
        for patch_idx, patch in enumerate(bp['boxes']):
            if patch_idx == 0:  # scenario0
                patch.set_facecolor('#FFFFFF')
            elif patch_idx == 1:  # scenario1
                patch.set_facecolor('#B0B0B0')
            else:  # scenario2_1
                patch.set_facecolor('#606060')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_drift_detector_quality(scenarios_data: dict, save_path: Path = None):
    """Analyze drift detector quality metrics (for scenario2_1)."""
    if "scenario2_1" not in scenarios_data:
        print("No drift detection data available")
        return
    
    df = scenarios_data["scenario2_1"]
    
    # Check for KNN distance data
    if "knn_distance" not in df.columns:
        print("No KNN distance data available for ROC analysis")
        return
    
    # For demonstration, we'll use a synthetic ground truth
    # In practice, this should be replaced with actual ground truth labels
    # For the last minute analysis, we assume drift might occur in the middle 20 seconds
    total_time = df["elapsed_s"].max()
    drift_start = 20
    drift_end = 40
    
    # Create ground truth labels
    y_true = ((df["elapsed_s"] >= drift_start) & (df["elapsed_s"] <= drift_end)).astype(int)
    
    # Use KNN distance as the score for ROC
    y_scores = df["knn_distance"].values
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.plot(fpr, tpr, color='black', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
            label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional drift analysis
    if "drift_detected" in df.columns:
        drift_events = df[df["drift_detected"] == True]
        
        # Calculate detection metrics
        total_drifts = len(drift_events)
        drift_rate = (total_drifts / len(df)) * 100
        
        print(f"\n=== Drift Detection Summary ===")
        print(f"Total drift events detected: {total_drifts}")
        print(f"Drift detection rate: {drift_rate:.2f}%")
        
        # Lead time analysis (time between actual drift and detection)
        if total_drifts > 0:
            detection_times = drift_events["elapsed_s"].values
            lead_times = []
            
            for det_time in detection_times:
                if drift_start <= det_time <= drift_end:
                    lead_time = det_time - drift_start
                    lead_times.append(lead_time)
            
            if lead_times:
                print(f"Average lead time: {np.mean(lead_times):.1f} seconds")
                print(f"Lead time std dev: {np.std(lead_times):.1f} seconds")


def create_summary_table(scenarios_data: dict, statistical_results: pd.DataFrame, save_path: Path = None):
    """Create a comprehensive summary table."""
    summary_data = []
    
    for scenario, df in scenarios_data.items():
        if len(df) == 0:
            continue
            
        # Handle different column names
        if "inference_time_ms" in df.columns:
            inference_col = "inference_time_ms"
        elif "total_time_ms" in df.columns:
            inference_col = "total_time_ms"
        else:
            inference_col = None
            
        summary = {
            "Scenario": SCENARIO_NAMES[scenario],
            "CPU Mean±Std (%)": f"{df['cpu_percent'].mean():.1f}±{df['cpu_percent'].std():.1f}",
            "CPU Max (%)": f"{df['cpu_percent'].max():.1f}",
            "Memory Mean±Std (MB)": f"{df['memory_mb'].mean():.1f}±{df['memory_mb'].std():.1f}",
            "Memory Max (MB)": f"{df['memory_mb'].max():.1f}",
        }
        
        if inference_col:
            summary["Latency Mean±Std (ms)"] = f"{df[inference_col].mean():.1f}±{df[inference_col].std():.1f}"
            summary["Latency Max (ms)"] = f"{df[inference_col].max():.1f}"
        
        summary["Total Inferences"] = len(df)
        
        # Add drift-specific metrics if available
        if "drift_detected" in df.columns and len(df) > 0:
            summary["Drift Events"] = df["drift_detected"].sum()
            summary["Drift Rate (%)"] = f"{(df['drift_detected'].sum() / len(df)) * 100:.2f}"
        
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    
    if save_path:
        summary_df.to_csv(save_path, index=False)
        print(f"\nSummary saved to: {save_path}")
    
    return summary_df


def plot_timing_breakdown(scenarios_data: dict, save_path: Path = None):
    """Create stacked bar chart showing timing breakdown for each scenario."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    timing_data = {}
    
    for scenario, df in scenarios_data.items():
        if len(df) == 0:
            continue
            
        timing_data[scenario] = {}
        
        if scenario == "scenario0":
            # Scenario 0 only has inference_time_ms
            timing_data[scenario]["Inference"] = df["inference_time_ms"].mean()
        else:
            # Scenarios 1 and 2.1 have component breakdown
            if "feature_extraction_time_ms" in df.columns:
                timing_data[scenario]["Feature Extraction"] = df["feature_extraction_time_ms"].mean()
            if "knn_inference_time_ms" in df.columns:
                timing_data[scenario]["KNN Inference"] = df["knn_inference_time_ms"].mean()
            if "drift_check_time_ms" in df.columns:
                timing_data[scenario]["Drift Check"] = df["drift_check_time_ms"].mean()
            
            # Calculate other overhead
            total_time = df["total_time_ms"].mean()
            component_sum = sum(timing_data[scenario].values())
            if total_time > component_sum:
                timing_data[scenario]["Other Overhead"] = total_time - component_sum
    
    # Create stacked bar chart
    scenarios = list(timing_data.keys())
    components = set()
    for data in timing_data.values():
        components.update(data.keys())
    components = sorted(list(components))
    
    # Define grayscale colors for components (lighter for readability)
    colors = {
        "Feature Extraction": "#404040",  # Dark gray instead of black
        "KNN Inference": "#808080",       # Medium gray
        "Drift Check": "#B0B0B0",         # Light gray
        "Other Overhead": "#D0D0D0",      # Very light gray
        "Inference": "#606060"            # Medium-dark gray
    }
    
    # Create bars
    bar_width = 0.6
    indices = np.arange(len(scenarios))
    bottom = np.zeros(len(scenarios))
    
    for component in components:
        values = [timing_data[s].get(component, 0) for s in scenarios]
        ax.bar(indices, values, bar_width, label=component, 
               bottom=bottom, color=colors.get(component, "#D0D0D0"))
        
        # Add value labels on bars
        for i, (val, b) in enumerate(zip(values, bottom)):
            if val > 10:  # Only show label if segment is large enough
                # Use white text for dark colors, black for light colors
                text_color = 'white' if component in ["Feature Extraction", "Inference"] else 'black'
                ax.text(i, b + val/2, f'{val:.0f}', 
                       ha='center', va='center', fontsize=9, color=text_color, fontweight='bold')
        
        bottom += values
    
    # Customize plot
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(indices)
    ax.set_xticklabels([SCENARIO_NAMES[s] for s in scenarios], rotation=15, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add total time annotations
    for i, scenario in enumerate(scenarios):
        total = sum(timing_data[scenario].values())
        ax.text(i, total + 5, f'{total:.0f} ms', 
               ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_timing_summary(scenarios_data: dict, save_path: Path = None):
    """Create detailed timing breakdown summary table."""
    summary_data = []
    
    for scenario, df in scenarios_data.items():
        if len(df) == 0:
            continue
            
        row = {"Scenario": SCENARIO_NAMES[scenario]}
        
        if scenario == "scenario0":
            # Scenario 0 only has inference_time_ms
            row["Total Time (ms)"] = f"{df['inference_time_ms'].mean():.1f}±{df['inference_time_ms'].std():.1f}"
            row["Feature Extraction (ms)"] = "N/A"
            row["KNN Inference (ms)"] = "N/A"
            row["Drift Check (ms)"] = "N/A"
            row["Other Overhead (ms)"] = "N/A"
        else:
            # Scenarios with component breakdown
            total_time = df["total_time_ms"].mean()
            total_std = df["total_time_ms"].std()
            row["Total Time (ms)"] = f"{total_time:.1f}±{total_std:.1f}"
            
            if "feature_extraction_time_ms" in df.columns:
                fe_mean = df["feature_extraction_time_ms"].mean()
                fe_std = df["feature_extraction_time_ms"].std()
                row["Feature Extraction (ms)"] = f"{fe_mean:.1f}±{fe_std:.1f}"
                row["Feature Extraction (%)"] = f"{(fe_mean/total_time*100):.1f}%"
            else:
                row["Feature Extraction (ms)"] = "N/A"
                row["Feature Extraction (%)"] = "N/A"
                
            if "knn_inference_time_ms" in df.columns:
                knn_mean = df["knn_inference_time_ms"].mean()
                knn_std = df["knn_inference_time_ms"].std()
                row["KNN Inference (ms)"] = f"{knn_mean:.1f}±{knn_std:.1f}"
                row["KNN Inference (%)"] = f"{(knn_mean/total_time*100):.1f}%"
            else:
                row["KNN Inference (ms)"] = "N/A"
                row["KNN Inference (%)"] = "N/A"
                
            if "drift_check_time_ms" in df.columns:
                drift_mean = df["drift_check_time_ms"].mean()
                drift_std = df["drift_check_time_ms"].std()
                row["Drift Check (ms)"] = f"{drift_mean:.2f}±{drift_std:.2f}"
                row["Drift Check (%)"] = f"{(drift_mean/total_time*100):.1f}%"
            else:
                row["Drift Check (ms)"] = "N/A"
                row["Drift Check (%)"] = "N/A"
            
            # Calculate overhead
            component_sum = 0
            if "feature_extraction_time_ms" in df.columns:
                component_sum += df["feature_extraction_time_ms"].mean()
            if "knn_inference_time_ms" in df.columns:
                component_sum += df["knn_inference_time_ms"].mean()
            if "drift_check_time_ms" in df.columns:
                component_sum += df["drift_check_time_ms"].mean()
                
            overhead = total_time - component_sum
            row["Other Overhead (ms)"] = f"{overhead:.1f}"
            row["Other Overhead (%)"] = f"{(overhead/total_time*100):.1f}%"
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    if save_path:
        summary_df.to_csv(save_path, index=False)
        print(f"\nTiming breakdown summary saved to: {save_path}")
    
    # Also print the summary
    print("\n=== Timing Breakdown Summary ===")
    print(summary_df.to_string(index=False))
    
    return summary_df


def plot_timing_components_over_time(scenarios_data: dict, save_path: Path = None):
    """Plot timing components over time for detailed analysis."""
    # Create subplot for each scenario with timing breakdown
    scenarios_with_breakdown = [s for s in scenarios_data.keys() if s != "scenario0"]
    
    if not scenarios_with_breakdown:
        print("No scenarios with timing breakdown available")
        return
    
    fig, axes = plt.subplots(len(scenarios_with_breakdown), 1, 
                            figsize=(10, 4*len(scenarios_with_breakdown)), 
                            sharex=True)
    
    if len(scenarios_with_breakdown) == 1:
        axes = [axes]
    
    for idx, scenario in enumerate(scenarios_with_breakdown):
        ax = axes[idx]
        df = scenarios_data[scenario]
        
        if len(df) == 0:
            continue
        
        # Plot each component
        if "feature_extraction_time_ms" in df.columns:
            ax.plot(df["elapsed_s"], df["feature_extraction_time_ms"], 
                   color="#000000", label="Feature Extraction", lw=1.5)
        
        if "knn_inference_time_ms" in df.columns:
            ax.plot(df["elapsed_s"], df["knn_inference_time_ms"], 
                   color="#606060", label="KNN Inference", lw=1.5)
        
        if "drift_check_time_ms" in df.columns and df["drift_check_time_ms"].max() > 0:
            ax.plot(df["elapsed_s"], df["drift_check_time_ms"], 
                   color="#A0A0A0", label="Drift Check", lw=1.0)
        
        # Plot total time
        ax.plot(df["elapsed_s"], df["total_time_ms"], 
               color="#000000", linestyle="--", label="Total Time", lw=2.0, alpha=0.7)
        
        ax.set_ylabel("Time (ms)")
        ax.set_title(SCENARIO_NAMES[scenario], loc='left', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel("Time (seconds)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main analysis function."""
    # Define data directory
    script_dir = Path(__file__).parent.absolute()
    
    # Always use absolute paths
    data_dir = script_dir / "data"
    output_dir = script_dir / "output"
    
    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data for all scenarios - directly use the 60 seconds approach
    scenarios_data = {}
    
    try:
        print("\n=== Loading last 60 seconds from each scenario ===")
        
        scenarios_data["scenario0"] = load_performance_data(
            data_dir, "performance_scenario0.json"
        )
        scenarios_data["scenario1"] = load_performance_data(
            data_dir, "performance_scenario1.json"
        )
        scenarios_data["scenario2_1"] = load_performance_data(
            data_dir, "performance_scenario2_1.json"
        )
        
        # Print sample counts for verification
        print(f"\nSample counts for fair comparison:")
        for scenario, df in scenarios_data.items():
            print(f"  {SCENARIO_NAMES[scenario]}: {len(df)} samples")
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure all performance data files are in the data directory")
        return
    
    # Perform statistical tests
    print("\n=== Statistical Analysis ===")
    statistical_results = perform_statistical_tests(scenarios_data)
    statistical_results.to_csv(output_dir / "statistical_tests.csv", index=False)
    
    # Create summary table
    print("\n=== Performance Summary ===")
    summary_df = create_summary_table(scenarios_data, statistical_results, 
                                    output_dir / "performance_summary_enhanced.csv")
    print(summary_df.to_string(index=False))
    
    # Generate grayscale plots
    print("\n=== Generating Grayscale Plots ===")
    
    # 1. Time series comparison
    print("Creating time series comparison...")
    plot_grayscale_comparison(scenarios_data, output_dir / "grayscale_comparison.png")
    
    # 2. Box plot comparison
    print("Creating box plot comparison...")
    plot_grayscale_boxplots(scenarios_data, output_dir / "grayscale_boxplots.png")
    
    # 3. Drift detector quality analysis
    print("Analyzing drift detector quality...")
    plot_drift_detector_quality(scenarios_data, output_dir / "drift_detector_roc.png")
    
    # 4. Timing breakdown analysis
    print("Creating timing breakdown analysis...")
    plot_timing_breakdown(scenarios_data, output_dir / "timing_breakdown.png")
    create_timing_summary(scenarios_data, output_dir / "timing_breakdown_summary.csv")
    plot_timing_components_over_time(scenarios_data, output_dir / "timing_components_timeseries.png")
    
    print(f"\nAll results saved to: {output_dir}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()