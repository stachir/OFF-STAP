"""
src/plots.py

Generates evaluation and diagnostic plots.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

def plot_spatial_alignment(df_aligned: pd.DataFrame, output_path: str):
    """Plots the 2D trajectories of EV3 and aligned VRMT."""
    if df_aligned.empty:
        return

    has_odom = all(c in df_aligned.columns for c in ["odometry_x", "odometry_y"])
    has_vrmt = all(c in df_aligned.columns for c in ["vrmt_aligned_x", "vrmt_aligned_y"])

    if not (has_odom and has_vrmt):
        return

    plt.figure(figsize=(8, 8))
    plt.plot(df_aligned["vrmt_aligned_x"], df_aligned["vrmt_aligned_y"], label="VRMT Reference", alpha=0.7)
    plt.plot(df_aligned["odometry_x"], df_aligned["odometry_y"], label="EV3 Odometry", alpha=0.7)

    plt.title(f"Spatial Alignment: {df_aligned['trial_id'].iloc[0]}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_boxplot(metrics_df: pd.DataFrame, metric: str, output_path: str):
    """Plots a boxplot of a specific metric across trials."""
    if metrics_df.empty or metric not in metrics_df.columns:
        return

    plt.figure(figsize=(8, 6))
    sns.boxplot(y=metric, data=metrics_df)
    sns.stripplot(y=metric, data=metrics_df, color=".3", size=4)

    plt.title(f"{metric.upper()} Distribution")
    plt.ylabel(f"{metric.upper()} (m)")

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_speed_comparison(df_aligned: pd.DataFrame, output_path: str):
    """Plots EV3 and Vive speed profiles to verify temporal alignment."""
    if df_aligned.empty:
        return

    has_time = "common_time_s" in df_aligned.columns
    has_ev3_speed = "left_speed" in df_aligned.columns and "right_speed" in df_aligned.columns
    has_vrmt_pos = all(c in df_aligned.columns for c in ["vrmt_aligned_x", "vrmt_aligned_y"])

    if not (has_time and has_ev3_speed and has_vrmt_pos):
        return

    import numpy as np

    WHEEL_RADIUS = 0.028

    omega_l = np.radians(df_aligned["left_speed"].fillna(0.0))
    omega_r = np.radians(df_aligned["right_speed"].fillna(0.0))
    v_l = omega_l * WHEEL_RADIUS
    v_r = omega_r * WHEEL_RADIUS
    ev3_speed = np.abs((v_r + v_l) / 2.0)

    # Calculate Vive speed using diff
    dt = np.diff(df_aligned["common_time_s"], prepend=df_aligned["common_time_s"].iloc[0])
    dx = np.diff(df_aligned["vrmt_aligned_x"], prepend=df_aligned["vrmt_aligned_x"].iloc[0])
    dy = np.diff(df_aligned["vrmt_aligned_y"], prepend=df_aligned["vrmt_aligned_y"].iloc[0])

    with np.errstate(divide='ignore', invalid='ignore'):
        vrmt_speed = np.sqrt(dx**2 + dy**2) / dt
        vrmt_speed[dt == 0] = 0

    # Smooth VRMT speed for cleaner plotting
    vrmt_speed = pd.Series(vrmt_speed).rolling(window=5, min_periods=1, center=True).mean().values

    plt.figure(figsize=(10, 4))
    plt.plot(df_aligned["common_time_s"], ev3_speed, label="EV3 Speed", alpha=0.8)
    plt.plot(df_aligned["common_time_s"], vrmt_speed, label="VRMT Speed", alpha=0.8)

    plt.title(f"Temporal Alignment (Speed Profile): {df_aligned['trial_id'].iloc[0]}")
    plt.xlabel("Common Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
