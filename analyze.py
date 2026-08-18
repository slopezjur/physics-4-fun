import sys
import csv
import os

folders = [f for f in os.listdir('Debug') if f.startswith('telemetry_dump_')]
folders.sort(reverse=True)
folders = folders[:3]

def analyze(folder):
    file = f'Debug/{folder}/{folder}.csv'
    recovering_data = []
    try:
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['State'] == 'Recovering':
                    recovering_data.append({
                        'Time': float(row['Time']),
                        'Progress': float(row['RecoveryProgress']),
                        'Height': float(row['PelvisPosY']),
                        'ChestHeight': float(row['Chest_PosY']),
                        'ChestPitch': float(row['Chest_EulerX']),
                        'ArmPitch': float(row['UpperArm_R_EulerX'])
                    })
    except Exception as e:
        return

    if not recovering_data:
        return

    print(f"--- {folder} ---")
    start = recovering_data[0]
    mid = recovering_data[len(recovering_data)//2]
    end = recovering_data[-1]
    
    print(f"Chest Height: Start {start['ChestHeight']:.2f} -> Mid {mid['ChestHeight']:.2f} -> End {end['ChestHeight']:.2f}")
    print(f"Chest Pitch: Start {start['ChestPitch']:.2f} -> Mid {mid['ChestPitch']:.2f} -> End {end['ChestPitch']:.2f}")
    print(f"Arm Pitch: Start {start['ArmPitch']:.2f} -> Mid {mid['ArmPitch']:.2f} -> End {end['ArmPitch']:.2f}")

for folder in folders:
    analyze(folder)
