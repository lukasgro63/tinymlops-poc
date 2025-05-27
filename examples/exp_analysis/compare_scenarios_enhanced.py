#!/usr/bin/env python3
"""
Enhanced Performance Comparison Script for TinyMLOps Scenarios
Compares Scenario 0 (baseline), Scenario 1 (TinyLCM no drift), and Scenario 2 (TinyLCM with drift)
Analyzes 5 runs per configuration with statistical tests and publication-ready grayscale styling
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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
    "scenario2": "#A0A0A0",      # Light gray
}

# Box plot specific colors for consistency
BOXPLOT_COLORS = {
    "scenario0": "#FFFFFF",      # White
    "scenario1": "#B0B0B0",      # Medium gray
    "scenario2": "#606060",      # Dark gray
}

SCENARIO_NAMES = {
    "scenario0": "Baseline TFLite",
    "scenario1": "TinyLCM (no drift)",
    "scenario2": "TinyLCM (drift detection)",
}

# Statistical test parameters
ALPHA = 0.05
WARMUP_SECONDS = 30


def load_single_run_data(file_path: Path, warmup_seconds: int = WARMUP_SECONDS) -> pd.DataFrame:
    """Load data from a single run, excluding warmup period."""
    if not file_path.exists():
        raise FileNotFoundError(f"No file found: {file_path}")
    
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
    
    # Calculate elapsed time from start
    min_time = df["timestamp"].min()
    df["elapsed_s"] = (df["timestamp"] - min_time).dt.total_seconds()
    
    # Remove warmup period
    df = df[df["elapsed_s"] >= warmup_seconds].copy()
    
    # Reset elapsed_s to start from 0 after warmup
    df["elapsed_s"] = df["elapsed_s"] - warmup_seconds
    
    return df


def extract_representative_metrics(df: pd.DataFrame) -> dict:
    """Extract representative metrics from a run (after warmup removal)."""
    # Handle different column names for inference time
    if "inference_time_ms" in df.columns:
        inference_col = "inference_time_ms"
    elif "total_time_ms" in df.columns:
        inference_col = "total_time_ms"
    else:
        inference_col = None
    
    metrics = {
        "cpu_mean": df["cpu_percent"].mean(),
        "cpu_std": df["cpu_percent"].std(),
        "memory_mean": df["memory_mb"].mean(),
        "memory_peak": df["memory_mb"].max(),
        "memory_std": df["memory_mb"].std(),
        "sample_count": len(df)
    }
    
    if inference_col:
        metrics["latency_mean"] = df[inference_col].mean()
        metrics["latency_std"] = df[inference_col].std()
        metrics["latency_p95"] = df[inference_col].quantile(0.95)
    
    # Add drift-specific metrics if available
    if "drift_detected" in df.columns:
        metrics["drift_rate"] = (df["drift_detected"].sum() / len(df)) * 100
        metrics["drift_count"] = df["drift_detected"].sum()
    
    return metrics


def load_all_runs(data_dir: Path) -> dict:
    """Load all runs for each scenario."""
    scenarios_data = {
        "scenario0": [],
        "scenario1": [],
        "scenario2": []
    }
    
    # Load 5 runs for each scenario
    for scenario_num, scenario_key in [(0, "scenario0"), (1, "scenario1"), (2, "scenario2")]:
        print(f"\nLoading {SCENARIO_NAMES[scenario_key]} runs...")
        
        for run_num in range(1, 6):
            if scenario_num == 2:
                # Scenario 2 uses pattern performance_scenario2_1_*.json
                pattern = f"performance_scenario2_1_{run_num}.json"
            else:
                pattern = f"performance_scenario{scenario_num}_{run_num}.json"
            
            file_path = data_dir / pattern
            
            try:
                df = load_single_run_data(file_path)
                metrics = extract_representative_metrics(df)
                metrics["run_id"] = run_num
                metrics["raw_data"] = df  # Keep raw data for detailed analysis
                scenarios_data[scenario_key].append(metrics)
                print(f"  Run {run_num}: {len(df)} samples after warmup removal")
            except FileNotFoundError:
                print(f"  Warning: {pattern} not found")
    
    # Convert to DataFrames for easier analysis
    for scenario in scenarios_data:
        if scenarios_data[scenario]:
            # Create DataFrame from metrics (excluding raw_data)
            metrics_list = [{k: v for k, v in m.items() if k != 'raw_data'} 
                           for m in scenarios_data[scenario]]
            scenarios_data[f"{scenario}_df"] = pd.DataFrame(metrics_list)
    
    return scenarios_data


def calculate_cliffs_delta(x1: np.ndarray, x2: np.ndarray) -> tuple:
    """Calculate Cliff's Delta effect size."""
    n1, n2 = len(x1), len(x2)
    
    # Count how many times values in x1 are greater than values in x2
    greater = 0
    less = 0
    
    for val1 in x1:
        for val2 in x2:
            if val1 > val2:
                greater += 1
            elif val1 < val2:
                less += 1
    
    delta = (greater - less) / (n1 * n2)
    
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
    
    return delta, interpretation


def perform_statistical_tests(scenarios_data: dict) -> pd.DataFrame:
    """Perform comprehensive statistical tests on the runs per scenario."""
    results = []
    
    # 1. Normality tests (Shapiro-Wilk)
    print("\n=== 1. Normality Tests (Shapiro-Wilk) ===")
    for scenario in ["scenario0", "scenario1", "scenario2"]:
        df_key = f"{scenario}_df"
        if df_key not in scenarios_data or scenarios_data[df_key].empty:
            continue
            
        df = scenarios_data[df_key]
        
        for metric, col in [("CPU", "cpu_mean"), ("Memory", "memory_peak"), ("Latency", "latency_mean")]:
            if col in df.columns and len(df) >= 3:
                values = df[col].values
                stat, p_value = stats.shapiro(values)
                normal = "Normal" if p_value > ALPHA else "Non-normal"
                
                results.append({
                    "Test_Type": "Shapiro-Wilk",
                    "Metric": metric,
                    "Configuration_1": SCENARIO_NAMES[scenario],
                    "Configuration_2": "N/A",
                    "Statistic_Value": stat,
                    "p_value": p_value,
                    "Significance_Result": normal,
                    "Effect_Size_Value": "N/A",
                    "Effect_Size_Interpretation": "N/A",
                    "Hypothesis_Tested": "Normality",
                    "Hypothesis_Result": normal
                })
                print(f"{SCENARIO_NAMES[scenario]} - {metric}: p={p_value:.4f} ({normal})")
    
    # 2. Mann-Whitney U tests for scenario comparisons (unpaired samples)
    print("\n=== 2. Mann-Whitney U Tests (Scenario Comparisons) ===")
    comparisons = [("scenario1", "scenario0"), ("scenario2", "scenario0"), ("scenario2", "scenario1")]
    
    for s1, s0 in comparisons:
        df1_key = f"{s1}_df"
        df0_key = f"{s0}_df"
        
        if df1_key not in scenarios_data or df0_key not in scenarios_data:
            continue
            
        df1 = scenarios_data[df1_key]
        df0 = scenarios_data[df0_key]
        
        if df1.empty or df0.empty:
            continue
            
        print(f"\n{SCENARIO_NAMES[s1]} vs {SCENARIO_NAMES[s0]}:")
        
        for metric, col in [("CPU", "cpu_mean"), ("Memory", "memory_peak"), ("Latency", "latency_mean")]:
            if col in df1.columns and col in df0.columns:
                values1 = df1[col].values
                values0 = df0[col].values
                
                if len(values1) >= 3 and len(values0) >= 3:  # Need sufficient samples
                    # Mann-Whitney U test for unpaired samples
                    stat, p_value = stats.mannwhitneyu(values1, values0, alternative='two-sided')
                    significant = "Significant" if p_value < ALPHA else "Not significant"
                    
                    # Calculate Cliff's Delta
                    delta, delta_interp = calculate_cliffs_delta(values1, values0)
                    
                    results.append({
                        "Test_Type": "Mann-Whitney U",
                        "Metric": metric,
                        "Configuration_1": SCENARIO_NAMES[s1],
                        "Configuration_2": SCENARIO_NAMES[s0],
                        "Statistic_Value": stat,
                        "p_value": p_value,
                        "Significance_Result": f"{significant} at α={ALPHA}",
                        "Effect_Size_Value": delta,
                        "Effect_Size_Interpretation": delta_interp,
                        "Hypothesis_Tested": "Difference",
                        "Hypothesis_Result": significant
                    })
                    print(f"  {metric}: p={p_value:.4f} ({significant}), δ={delta:.3f} ({delta_interp})")
    
    # 3. Hypothesis H1: Latency Overhead < 50%
    print("\n=== 3. Hypothesis H1 (Latency Overhead < 50%) ===")
    
    if "scenario0_df" in scenarios_data and "latency_mean" in scenarios_data["scenario0_df"].columns:
        baseline_latencies = scenarios_data["scenario0_df"]["latency_mean"].values
        baseline_mean = baseline_latencies.mean()
        
        for scenario in ["scenario1", "scenario2"]:
            df_key = f"{scenario}_df"
            if df_key in scenarios_data and "latency_mean" in scenarios_data[df_key].columns:
                scenario_latencies = scenarios_data[df_key]["latency_mean"].values
                
                # Calculate percentage increases
                pct_increases = ((scenario_latencies - baseline_mean) / baseline_mean) * 100
                
                # One-sample Wilcoxon test against 50%
                stat, p_value = stats.wilcoxon(pct_increases - 50, alternative='less')
                h1_result = "H1 accepted" if p_value < ALPHA else "H1 rejected"
                
                results.append({
                    "Test_Type": "One-sample Wilcoxon",
                    "Metric": "Latency overhead",
                    "Configuration_1": SCENARIO_NAMES[scenario],
                    "Configuration_2": "50% threshold",
                    "Statistic_Value": stat,
                    "p_value": p_value,
                    "Significance_Result": f"p < {ALPHA}" if p_value < ALPHA else f"p ≥ {ALPHA}",
                    "Effect_Size_Value": np.median(pct_increases),
                    "Effect_Size_Interpretation": f"Median: {np.median(pct_increases):.1f}%",
                    "Hypothesis_Tested": "H1: Latency overhead < 50%",
                    "Hypothesis_Result": h1_result
                })
                print(f"{SCENARIO_NAMES[scenario]}: Median overhead = {np.median(pct_increases):.1f}%, p={p_value:.4f} ({h1_result})")
    
    # 4. Hypothesis H2: Resource Constraints
    print("\n=== 4. Hypothesis H2 (CPU < 50%, Memory < 256 MiB) ===")
    
    for scenario in ["scenario1", "scenario2"]:
        df_key = f"{scenario}_df"
        if df_key not in scenarios_data:
            continue
            
        df = scenarios_data[df_key]
        
        # H2a: CPU < 50%
        if "cpu_mean" in df.columns:
            cpu_values = df["cpu_mean"].values
            stat, p_value = stats.wilcoxon(cpu_values - 50, alternative='less')
            h2_cpu_result = "H2 CPU accepted" if p_value < ALPHA else "H2 CPU rejected"
            
            results.append({
                "Test_Type": "One-sample Wilcoxon",
                "Metric": "CPU usage",
                "Configuration_1": SCENARIO_NAMES[scenario],
                "Configuration_2": "50% threshold",
                "Statistic_Value": stat,
                "p_value": p_value,
                "Significance_Result": f"p < {ALPHA}" if p_value < ALPHA else f"p ≥ {ALPHA}",
                "Effect_Size_Value": np.median(cpu_values),
                "Effect_Size_Interpretation": f"Median: {np.median(cpu_values):.1f}%",
                "Hypothesis_Tested": "H2: CPU < 50%",
                "Hypothesis_Result": h2_cpu_result
            })
            print(f"{SCENARIO_NAMES[scenario]} CPU: Median = {np.median(cpu_values):.1f}%, p={p_value:.4f} ({h2_cpu_result})")
        
        # H2b: Peak Memory < 256 MiB
        if "memory_peak" in df.columns:
            memory_peaks = df["memory_peak"].values
            below_threshold = np.sum(memory_peaks < 256)
            all_below = below_threshold == len(memory_peaks)
            h2_mem_result = "H2 Memory accepted" if all_below else "H2 Memory rejected"
            
            results.append({
                "Test_Type": "Descriptive",
                "Metric": "Peak memory",
                "Configuration_1": SCENARIO_NAMES[scenario],
                "Configuration_2": "256 MiB threshold",
                "Statistic_Value": below_threshold,
                "p_value": "N/A",
                "Significance_Result": f"{below_threshold}/{len(memory_peaks)} below threshold",
                "Effect_Size_Value": np.max(memory_peaks),
                "Effect_Size_Interpretation": f"Max: {np.max(memory_peaks):.1f} MiB",
                "Hypothesis_Tested": "H2: Peak Memory < 256 MiB",
                "Hypothesis_Result": h2_mem_result
            })
            print(f"{SCENARIO_NAMES[scenario]} Memory: {below_threshold}/{len(memory_peaks)} runs below 256 MiB, Max = {np.max(memory_peaks):.1f} MiB ({h2_mem_result})")
    
    return pd.DataFrame(results)


def plot_boxplots_with_points(scenarios_data: dict, save_path: Path = None):
    """Create box plots with individual data points for all metrics."""
    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Prepare data
    plot_data = []
    
    for scenario in ["scenario0", "scenario1", "scenario2"]:
        df_key = f"{scenario}_df"
        if df_key in scenarios_data and not scenarios_data[df_key].empty:
            df = scenarios_data[df_key]
            
            # CPU data
            if "cpu_mean" in df.columns:
                for val in df["cpu_mean"].values:
                    plot_data.append({
                        "Scenario": SCENARIO_NAMES[scenario],
                        "Metric": "CPU",
                        "Value": val
                    })
            
            # Memory data
            if "memory_peak" in df.columns:
                for val in df["memory_peak"].values:
                    plot_data.append({
                        "Scenario": SCENARIO_NAMES[scenario],
                        "Metric": "Memory",
                        "Value": val
                    })
            
            # Latency data
            if "latency_mean" in df.columns:
                for val in df["latency_mean"].values:
                    plot_data.append({
                        "Scenario": SCENARIO_NAMES[scenario],
                        "Metric": "Latency",
                        "Value": val
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create box plots
    metrics = ["CPU", "Memory", "Latency"]
    ylabels = ["CPU Usage (%)", "Peak Memory (MB)", "Latency (ms)"]
    
    for idx, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        ax = axes[idx]
        metric_data = plot_df[plot_df["Metric"] == metric]
        
        if not metric_data.empty:
            # Create box plot
            positions = []
            data_arrays = []
            labels = []
            
            for i, scenario_name in enumerate(SCENARIO_NAMES.values()):
                scenario_data = metric_data[metric_data["Scenario"] == scenario_name]["Value"].values
                if len(scenario_data) > 0:
                    positions.append(i)
                    data_arrays.append(scenario_data)
                    labels.append(scenario_name.replace(" ", "\n"))
            
            if data_arrays:
                bp = ax.boxplot(data_arrays, positions=positions, labels=labels, 
                               patch_artist=True, widths=0.6,
                               boxprops=dict(facecolor='#D0D0D0', linewidth=1.5),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5),
                               medianprops=dict(linewidth=2, color='black'))
                
                # Add individual points
                for pos, data in zip(positions, data_arrays):
                    # Add slight jitter to x-position
                    jitter = np.random.normal(0, 0.04, size=len(data))
                    ax.scatter(pos + jitter, data, alpha=0.6, s=30, 
                             color='black', zorder=10)
                
                # Color boxes with consistent grayscale scheme
                scenario_keys = ["scenario0", "scenario1", "scenario2"]
                for patch_idx, patch in enumerate(bp['boxes']):
                    if patch_idx < len(scenario_keys):
                        patch.set_facecolor(BOXPLOT_COLORS[scenario_keys[patch_idx]])
        
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_histograms_and_density(scenarios_data: dict, save_path: Path = None):
    """Create histograms and density plots for each metric and configuration."""
    _, axes = plt.subplots(3, 3, figsize=(12, 10))
    
    metrics = [("cpu_mean", "CPU Usage (%)", 0),
               ("memory_peak", "Peak Memory (MB)", 1),
               ("latency_mean", "Latency (ms)", 2)]
    
    for scenario_idx, scenario in enumerate(["scenario0", "scenario1", "scenario2"]):
        df_key = f"{scenario}_df"
        if df_key not in scenarios_data or scenarios_data[df_key].empty:
            continue
            
        df = scenarios_data[df_key]
        
        for col, label, metric_idx in metrics:
            if col in df.columns:
                ax = axes[metric_idx, scenario_idx]
                values = df[col].values
                
                if len(values) > 1:
                    # Histogram with fewer bins for 5 data points
                    ax.hist(values, bins='auto', density=True, alpha=0.5, 
                           color=SCENARIO_COLORS[scenario], edgecolor='black')
                    
                    # KDE if enough data points
                    if len(values) >= 5:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(values)
                        x_range = np.linspace(values.min(), values.max(), 100)
                        ax.plot(x_range, kde(x_range), color='black', lw=2)
                    
                    # Add vertical line for mean
                    ax.axvline(values.mean(), color='black', linestyle='--', 
                              lw=1.5, label=f'Mean: {values.mean():.1f}')
                    
                    ax.set_xlabel(label)
                    ax.set_ylabel('Density' if metric_idx == 0 else '')
                    ax.set_title(SCENARIO_NAMES[scenario] if metric_idx == 0 else '')
                    ax.legend(loc='upper right', fontsize=9)
                    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_descriptive_statistics_table(scenarios_data: dict, save_path: Path = None):
    """Create comprehensive descriptive statistics table."""
    stats_data = []
    
    for scenario in ["scenario0", "scenario1", "scenario2"]:
        df_key = f"{scenario}_df"
        if df_key not in scenarios_data or scenarios_data[df_key].empty:
            continue
            
        df = scenarios_data[df_key]
        
        for metric, col in [("CPU (mean %)", "cpu_mean"), 
                           ("Memory (peak MB)", "memory_peak"), 
                           ("Latency (mean ms)", "latency_mean")]:
            if col in df.columns:
                values = df[col].values
                
                stats_row = {
                    "Configuration": SCENARIO_NAMES[scenario],
                    "Metric": metric,
                    "N": len(values),
                    "Mean": np.mean(values),
                    "Std Dev": np.std(values, ddof=1),
                    "Median": np.median(values),
                    "Min": np.min(values),
                    "Max": np.max(values),
                    "IQR": np.percentile(values, 75) - np.percentile(values, 25),
                    "CV%": (np.std(values, ddof=1) / np.mean(values) * 100) if np.mean(values) > 0 else 0
                }
                stats_data.append(stats_row)
    
    stats_df = pd.DataFrame(stats_data)
    
    # Format numeric columns
    numeric_cols = ["Mean", "Std Dev", "Median", "Min", "Max", "IQR", "CV%"]
    for col in numeric_cols:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].round(2)
    
    if save_path:
        stats_df.to_csv(save_path, index=False)
        print(f"\nDescriptive statistics saved to: {save_path}")
    
    # Print formatted table
    print("\n=== Descriptive Statistics ===")
    print(stats_df.to_string(index=False))
    
    return stats_df


def main():
    """Main analysis function."""
    # Define data directory
    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir / "data"
    output_dir = script_dir / "output"
    
    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all runs
    print("=== Loading Performance Data (5 runs per configuration) ===")
    scenarios_data = load_all_runs(data_dir)
    
    # Verify data loading
    print("\n=== Data Loading Summary ===")
    for scenario in ["scenario0", "scenario1", "scenario2"]:
        df_key = f"{scenario}_df"
        if df_key in scenarios_data:
            print(f"{SCENARIO_NAMES[scenario]}: {len(scenarios_data[df_key])} runs loaded")
    
    # Create descriptive statistics table
    print("\n=== Calculating Descriptive Statistics ===")
    create_descriptive_statistics_table(
        scenarios_data, 
        output_dir / "descriptive_statistics.csv"
    )
    
    # Perform statistical tests
    print("\n=== Performing Statistical Tests ===")
    test_results_df = perform_statistical_tests(scenarios_data)
    test_results_df.to_csv(output_dir / "statistical_test_results.csv", index=False)
    print(f"Statistical test results saved to: {output_dir / 'statistical_test_results.csv'}")
    
    # Generate plots
    print("\n=== Generating Plots ===")
    
    # 1. Box plots with individual points
    print("Creating box plots with data points...")
    plot_boxplots_with_points(scenarios_data, output_dir / "boxplots_with_points.png")
    
    # 2. Histograms and density plots
    print("Creating histograms and density plots...")
    plot_histograms_and_density(scenarios_data, output_dir / "histograms_density.png")
    
    print(f"\nAll results saved to: {output_dir}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()