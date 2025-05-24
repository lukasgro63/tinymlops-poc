#!/usr/bin/env python3
"""
Performance Analysis Script for TinyLCM Evaluation
-------------------------------------------------
Analyzes and compares performance metrics from Scenarios 0, 1, and 2.
Generates comparison tables and visualizations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def load_performance_logs(log_dir: Path, scenario: str) -> List[Dict]:
    """Load all performance logs for a specific scenario."""
    metrics = []
    pattern = f"performance_scenario{scenario}_*.json"
    
    for log_file in log_dir.glob(pattern):
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    metric = json.loads(line.strip())
                    metrics.append(metric)
                except json.JSONDecodeError:
                    continue
    
    return metrics


def extract_inference_metrics(metrics: List[Dict]) -> pd.DataFrame:
    """Extract inference metrics into a DataFrame."""
    inference_data = []
    
    for metric in metrics:
        if metric.get("type") == "inference":
            data = {
                "timestamp": metric["timestamp"],
                "cpu_percent": metric["cpu_percent"],
                "memory_mb": metric["memory_mb"]
            }
            
            # Handle different metric formats
            if "inference_time_ms" in metric:
                # Scenario 0 format
                data["total_time_ms"] = metric["inference_time_ms"]
            elif "total_time_ms" in metric:
                # Scenario 1 format
                data["total_time_ms"] = metric["total_time_ms"]
                data["feature_extraction_time_ms"] = metric.get("feature_extraction_time_ms", 0)
                data["knn_inference_time_ms"] = metric.get("knn_inference_time_ms", 0)
            
            inference_data.append(data)
    
    return pd.DataFrame(inference_data)


def calculate_statistics(df: pd.DataFrame) -> Dict:
    """Calculate summary statistics for a scenario."""
    stats = {
        "inference_count": len(df),
        "avg_total_time_ms": df["total_time_ms"].mean(),
        "std_total_time_ms": df["total_time_ms"].std(),
        "min_total_time_ms": df["total_time_ms"].min(),
        "max_total_time_ms": df["total_time_ms"].max(),
        "p95_total_time_ms": df["total_time_ms"].quantile(0.95),
        "avg_cpu_percent": df["cpu_percent"].mean(),
        "std_cpu_percent": df["cpu_percent"].std(),
        "avg_memory_mb": df["memory_mb"].mean(),
        "std_memory_mb": df["memory_mb"].std(),
        "max_memory_mb": df["memory_mb"].max()
    }
    
    # Add component timing if available
    if "feature_extraction_time_ms" in df.columns:
        stats["avg_feature_extraction_time_ms"] = df["feature_extraction_time_ms"].mean()
    if "knn_inference_time_ms" in df.columns:
        stats["avg_knn_inference_time_ms"] = df["knn_inference_time_ms"].mean()
    
    return stats


def compare_scenarios(scenario_stats: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison table between scenarios."""
    comparison_data = []
    
    # Define metrics to compare
    metrics = [
        ("Average Inference Time (ms)", "avg_total_time_ms"),
        ("Std Dev Inference Time (ms)", "std_total_time_ms"),
        ("95th Percentile Time (ms)", "p95_total_time_ms"),
        ("Average CPU Usage (%)", "avg_cpu_percent"),
        ("Average Memory (MB)", "avg_memory_mb"),
        ("Max Memory (MB)", "max_memory_mb")
    ]
    
    for display_name, metric_key in metrics:
        row = {"Metric": display_name}
        
        for scenario, stats in scenario_stats.items():
            if metric_key in stats:
                row[f"Scenario {scenario}"] = f"{stats[metric_key]:.2f}"
            else:
                row[f"Scenario {scenario}"] = "N/A"
        
        # Calculate overhead percentages if Scenario 0 exists
        if "0" in scenario_stats and metric_key in scenario_stats["0"]:
            base_value = scenario_stats["0"][metric_key]
            if base_value > 0:
                for scenario in ["1", "2"]:
                    if scenario in scenario_stats and metric_key in scenario_stats[scenario]:
                        overhead = ((scenario_stats[scenario][metric_key] - base_value) / base_value) * 100
                        row[f"Overhead vs S0 ({scenario})"] = f"{overhead:+.1f}%"
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def print_component_breakdown(scenario_stats: Dict[str, Dict]):
    """Print component timing breakdown for Scenario 1."""
    if "1" in scenario_stats:
        stats = scenario_stats["1"]
        if "avg_feature_extraction_time_ms" in stats:
            total = stats["avg_total_time_ms"]
            feature = stats["avg_feature_extraction_time_ms"]
            knn = stats.get("avg_knn_inference_time_ms", 0)
            
            print("\nScenario 1 Component Breakdown:")
            print(f"  Feature Extraction: {feature:.2f} ms ({feature/total*100:.1f}%)")
            print(f"  KNN Inference: {knn:.2f} ms ({knn/total*100:.1f}%)")
            print(f"  Total: {total:.2f} ms")


def main(log_dir: str):
    """Main analysis function."""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Error: Log directory '{log_dir}' does not exist")
        sys.exit(1)
    
    print("TinyLCM Performance Analysis")
    print("=" * 50)
    
    # Load and analyze each scenario
    scenario_stats = {}
    
    for scenario in ["0", "1", "2"]:
        print(f"\nAnalyzing Scenario {scenario}...")
        
        # Load metrics
        metrics = load_performance_logs(log_path, scenario)
        
        if not metrics:
            print(f"  No performance logs found for Scenario {scenario}")
            continue
        
        # Extract inference metrics
        df = extract_inference_metrics(metrics)
        
        if df.empty:
            print(f"  No inference metrics found for Scenario {scenario}")
            continue
        
        # Calculate statistics
        stats = calculate_statistics(df)
        scenario_stats[scenario] = stats
        
        print(f"  Found {stats['inference_count']} inference samples")
        print(f"  Average inference time: {stats['avg_total_time_ms']:.2f} ms")
        print(f"  Average CPU usage: {stats['avg_cpu_percent']:.1f}%")
        print(f"  Average memory: {stats['avg_memory_mb']:.1f} MB")
    
    # Create comparison table
    if len(scenario_stats) > 1:
        print("\n" + "=" * 50)
        print("SCENARIO COMPARISON")
        print("=" * 50)
        
        comparison_df = compare_scenarios(scenario_stats)
        print("\n" + comparison_df.to_string(index=False))
        
        # Print component breakdown
        print_component_breakdown(scenario_stats)
        
        # Save comparison to file
        output_file = log_path / f"performance_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(output_file, index=False)
        print(f"\nComparison saved to: {output_file}")
    
    # Print summary insights
    if len(scenario_stats) == 3:
        print("\n" + "=" * 50)
        print("PERFORMANCE INSIGHTS")
        print("=" * 50)
        
        # TinyLCM core overhead (S1 vs S0)
        if "0" in scenario_stats and "1" in scenario_stats:
            overhead_core = ((scenario_stats["1"]["avg_total_time_ms"] - 
                            scenario_stats["0"]["avg_total_time_ms"]) / 
                           scenario_stats["0"]["avg_total_time_ms"] * 100)
            print(f"\nTinyLCM Core Overhead (S1 vs S0): {overhead_core:+.1f}%")
        
        # Drift detection overhead (S2 vs S1)
        if "1" in scenario_stats and "2" in scenario_stats:
            overhead_drift = ((scenario_stats["2"]["avg_total_time_ms"] - 
                             scenario_stats["1"]["avg_total_time_ms"]) / 
                            scenario_stats["1"]["avg_total_time_ms"] * 100)
            print(f"Drift Detection Overhead (S2 vs S1): {overhead_drift:+.1f}%")
        
        # Total TinyLCM overhead (S2 vs S0)
        if "0" in scenario_stats and "2" in scenario_stats:
            overhead_total = ((scenario_stats["2"]["avg_total_time_ms"] - 
                             scenario_stats["0"]["avg_total_time_ms"]) / 
                            scenario_stats["0"]["avg_total_time_ms"] * 100)
            print(f"Total TinyLCM Overhead (S2 vs S0): {overhead_total:+.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TinyLCM performance across scenarios")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory containing performance log files"
    )
    
    args = parser.parse_args()
    main(args.log_dir)