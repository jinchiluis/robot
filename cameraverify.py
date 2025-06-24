import json

# Load and print raw calibration data
with open('calibration/7point800_800.json', 'r') as f:
    cal_data = json.load(f)

print("Calibration points:")
for i, point in enumerate(cal_data['calibration_points']):
    print(f"Point {i}: Pixel({point['pixel_x']}, {point['pixel_y']}) -> Robot({point['robot_x']}, {point['robot_y']})")

# Check transformation matrix
if 'transformation_matrix' in cal_data:
    print("\nTransformation matrix:")
    for row in cal_data['transformation_matrix']:
        print(row)