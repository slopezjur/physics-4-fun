import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

folders = [
    'Debug/telemetry_dump_20260819_001216/telemetry_dump_20260819_001216.csv',
    'Debug/telemetry_dump_20260819_001227/telemetry_dump_20260819_001227.csv',
    'Debug/telemetry_dump_20260819_001238/telemetry_dump_20260819_001238.csv'
]
titles = ['Dump 1 (Initial)', 'Dump 2 (Good Recovery)', 'Dump 3 (Multiple Hits/Restarts)']

for i, file_path in enumerate(folders):
    if not os.path.exists(file_path):
        print(f"Not found: {file_path}")
        continue
        
    df = pd.read_csv(file_path)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(titles[i], fontsize=16)
    
    # Ax 1: States
    state_map = {'Balanced': 0, 'Stumbling': 1, 'Flailing': 2, 'KnockedOut': 3, 'Recovering': 4}
    df['StateNum'] = df['State'].map(state_map)
    axes[0].plot(df['Time'], df['StateNum'], label='State', drawstyle='steps-post')
    axes[0].set_yticks([0, 1, 2, 3, 4])
    axes[0].set_yticklabels(['Balanced', 'Stumbling', 'Flailing', 'KnockedOut', 'Recovering'])
    axes[0].set_ylabel('Ragdoll State')
    axes[0].grid(True)
    
    # Add hit markers if there's a hit column (Wait, does the telemetry have hit info? We can check PelvisSpeed)
    
    # Ax 2: Pelvis Pitch/Tilt and Knee angles
    axes[1].plot(df['Time'], df['PelvisTiltDeg'], label='Pelvis Tilt (deg)', color='purple')
    if 'KneeAngleL' in df.columns:
        axes[1].plot(df['Time'], df['KneeAngleL'], label='Left Knee (deg)', alpha=0.6)
        axes[1].plot(df['Time'], df['KneeAngleR'], label='Right Knee (deg)', alpha=0.6)
    axes[1].set_ylabel('Degrees')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)
    
    # Ax 3: Pelvis Speed and Grounded
    axes[2].plot(df['Time'], df['PelvisSpeed'], label='Pelvis Speed (m/s)', color='red')
    axes[2].set_ylabel('Speed / Status')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)
    
    out_img = f'C:/Users/chenc/.gemini/antigravity/brain/0f5016b2-63cf-4c03-91aa-21868ffc8d26/scratch/dump_{i+1}.png'
    plt.tight_layout()
    plt.savefig(out_img)
    plt.close()
    print(f"Saved {out_img}")

