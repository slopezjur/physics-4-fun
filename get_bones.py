import sys
import csv

file = 'Debug/telemetry_dump_20260819_005731/telemetry_dump_20260819_005731.csv'
try:
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['State'] == 'Recovering':
                time = float(row['Time'])
                chest_y = float(row['Chest_PosY'])
                arm_y = float(row['UpperArm_R_PosY'])
                forearm_y = float(row['Forearm_R_PosY'])
                print(f"Time {time:.2f} | Chest Y: {chest_y:.2f} | UArm Y: {arm_y:.2f} | FArm Y: {forearm_y:.2f}")
except Exception as e:
    pass
