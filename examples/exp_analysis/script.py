import matplotlib.pyplot as plt
import pandas as pd
from ace_tools import display_dataframe_to_user
from matplotlib.lines import Line2D

# style (unchanged)
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
    "legend.fontsize": 10,   # bigger legend font
    "font.size": 10,
    "figure.dpi": 110,
})

# load data
df = pd.read_json("/mnt/data/performance_scenario0_20250524_003550.json", lines=True)
df = df[df["type"] == "inference"].copy()
df["timestamp"] = pd.to_datetime(df["timestamp"])
t0 = df["timestamp"].min()
df["elapsed_s"] = (df["timestamp"] - t0).dt.total_seconds()
df = df[df["elapsed_s"] > 60].copy()

# summary
summary = df[["cpu_percent", "memory_mb", "inference_time_ms"]].describe()
display_dataframe_to_user("Scenario 0 – performance summary (last minute)", summary.round(2))

# bigger figure
fig, ax_cpu = plt.subplots(figsize=(10,4.5))  # height increased

ax_cpu.plot(df["elapsed_s"], df["cpu_percent"], color="black", lw=1.0, label="CPU (%)")
ax_cpu.set_xlabel("Time (s)")
ax_cpu.set_ylabel("CPU (%)", fontweight="bold")
ax_cpu.grid(True, axis="y", linestyle="--", color="grey", lw=0.5)

ax_mem = ax_cpu.twinx()
ax_mem.plot(df["elapsed_s"], df["memory_mb"], color="grey", ls="--", lw=1.0, label="Memory (MB)")
ax_mem.set_ylabel("Memory (MB)", fontweight="bold")

ax_inf = ax_cpu.twinx()
ax_inf.spines.right.set_position(("axes", 1.14))
ax_inf.plot(df["elapsed_s"], df["inference_time_ms"], linestyle="", marker="s",
            ms=5, color="lightgrey", label="Inference time (ms)")
ax_inf.set_ylabel("Inference time (ms)", fontweight="bold")

lines = ax_cpu.get_lines() + ax_mem.get_lines() + ax_inf.get_lines()
labels = [l.get_label() for l in lines]
ax_cpu.legend(lines, labels, loc="upper right")

plt.title("Scenario 0 — runtime metrics (last minute)", fontweight="bold")
plt.tight_layout()
plt.show()
