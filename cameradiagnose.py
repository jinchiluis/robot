import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

def diagnose_calibration(transformer, image_width=640, image_height=480):
    """
    Diagnose calibration issues by visualizing the transformation.
    
    Args:
        transformer: Your CoordinateTransformer instance
        image_width: Width of camera image
        image_height: Height of camera image
    """
    if not transformer.is_calibrated():
        print("Transformer is not calibrated!")
        return
        
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Calibration Diagnostic Report', fontsize=16)
    
    # 1. Plot calibration points
    ax = axes[0, 0]
    pixel_points = np.array(transformer.pixel_points)
    ax.scatter(pixel_points[:, 0], pixel_points[:, 1], c='red', s=100, marker='o')
    for i, (x, y) in enumerate(pixel_points):
        ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)  # Invert Y axis for image coordinates
    ax.set_title('Calibration Points Distribution')
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    ax.grid(True, alpha=0.3)
    
    # Add rectangle showing image bounds
    rect = Rectangle((0, 0), image_width, image_height, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(rect)
    
    # 2. Plot robot points
    ax = axes[0, 1]
    robot_points = np.array(transformer.robot_points)
    ax.scatter(robot_points[:, 0], robot_points[:, 1], c='blue', s=100, marker='s')
    for i, (x, y) in enumerate(robot_points):
        ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    ax.set_title('Robot Points Distribution')
    ax.set_xlabel('Robot X (mm)')
    ax.set_ylabel('Robot Y (mm)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 3. Error visualization
    ax = axes[0, 2]
    errors = []
    for i, (px, py) in enumerate(pixel_points):
        pred = transformer.pixel_to_robot(px, py)
        if pred:
            actual = robot_points[i]
            error = np.sqrt((pred[0] - actual[0])**2 + (pred[1] - actual[1])**2)
            errors.append(error)
            # Draw error vectors
            ax.arrow(actual[0], actual[1], 
                    pred[0] - actual[0], pred[1] - actual[1],
                    head_width=5, head_length=3, fc='red', ec='red', alpha=0.6)
    
    ax.scatter(robot_points[:, 0], robot_points[:, 1], c='blue', s=100, marker='o', label='Actual')
    ax.set_title(f'Prediction Errors (RMS: {np.sqrt(np.mean(np.square(errors))):.2f}mm)')
    ax.set_xlabel('Robot X (mm)')
    ax.set_ylabel('Robot Y (mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 4. Grid transformation visualization
    ax = axes[1, 0]
    grid_size = 10
    x_grid = np.linspace(0, image_width, grid_size)
    y_grid = np.linspace(0, image_height, grid_size)
    
    # Plot transformed grid
    for x in x_grid:
        points = []
        for y in y_grid:
            robot_pt = transformer.pixel_to_robot(x, y)
            if robot_pt:
                points.append(robot_pt)
        if points:
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], 'b-', alpha=0.5)
    
    for y in y_grid:
        points = []
        for x in x_grid:
            robot_pt = transformer.pixel_to_robot(x, y)
            if robot_pt:
                points.append(robot_pt)
        if points:
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], 'b-', alpha=0.5)
    
    ax.set_title('Grid Transformation (Should be regular if linear)')
    ax.set_xlabel('Robot X (mm)')
    ax.set_ylabel('Robot Y (mm)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 5. Error heatmap
    ax = axes[1, 1]
    resolution = 50
    x_test = np.linspace(0, image_width, resolution)
    y_test = np.linspace(0, image_height, resolution)
    X, Y = np.meshgrid(x_test, y_test)
    
    # For each test point, find nearest calibration point and estimate error
    error_map = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            px, py = X[i, j], Y[i, j]
            # Find nearest calibration point
            distances = [np.sqrt((px-cp[0])**2 + (py-cp[1])**2) for cp in pixel_points]
            nearest_idx = np.argmin(distances)
            
            # Estimate error based on nearest point's error
            if nearest_idx < len(errors):
                # Weight error by distance
                weight = 1 / (1 + distances[nearest_idx] / 100)
                error_map[i, j] = errors[nearest_idx] * (2 - weight)
    
    im = ax.contourf(X, Y, error_map, levels=20, cmap='hot')
    ax.scatter(pixel_points[:, 0], pixel_points[:, 1], c='blue', s=50, marker='o', edgecolor='white')
    ax.set_title('Estimated Error Distribution')
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    plt.colorbar(im, ax=ax, label='Error (mm)')
    
    # 6. Residual analysis
    ax = axes[1, 2]
    if errors:
        ax.bar(range(len(errors)), errors)
        ax.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}mm')
        ax.axhline(y=5, color='g', linestyle='--', label='5mm threshold')
        ax.set_title('Calibration Point Errors')
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Error (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print diagnostic summary
    print("\n=== CALIBRATION DIAGNOSTIC SUMMARY ===")
    print(f"Number of calibration points: {len(pixel_points)}")
    print(f"RMS Error: {transformer.calibration_error:.2f}mm")
    print(f"Max Error: {max(errors):.2f}mm")
    print(f"Mean Error: {np.mean(errors):.2f}mm")
    
    # Check point distribution
    x_span = pixel_points[:, 0].max() - pixel_points[:, 0].min()
    y_span = pixel_points[:, 1].max() - pixel_points[:, 1].min()
    coverage = (x_span * y_span) / (image_width * image_height)
    print(f"Point coverage: {coverage*100:.1f}%")
    
    # Identify problematic areas
    if max(errors) > 10:
        bad_points = [i for i, e in enumerate(errors) if e > 10]
        print(f"\n⚠️ High error points: {bad_points}")
        
    # Check for edge vs center accuracy
    center_x, center_y = image_width/2, image_height/2
    center_points = []
    edge_points = []
    
    for i, (px, py) in enumerate(pixel_points):
        dist_from_center = np.sqrt((px - center_x)**2 + (py - center_y)**2)
        if dist_from_center < min(image_width, image_height) / 4:
            center_points.append(errors[i])
        else:
            edge_points.append(errors[i])
    
    if center_points and edge_points:
        center_error = np.mean(center_points)
        edge_error = np.mean(edge_points)
        print(f"\nCenter accuracy: {center_error:.2f}mm")
        print(f"Edge accuracy: {edge_error:.2f}mm")
        
        if edge_error > 2 * center_error:
            print("\n⚠️ SIGNIFICANT EDGE DISTORTION DETECTED!")
            print("This suggests lens distortion. Consider:")
            print("1. Camera calibration to remove lens distortion")
            print("2. Using polynomial or TPS transformation model")
            print("3. Adding more calibration points at edges/corners")
    
    return fig

# Usage example:
if __name__ == "__main__":
    # Load your calibration
    with open('calibration/13point600_600.JSON', 'r') as f:
        cal_data = json.load(f)
    
    from coordinate_transformer import CoordinateTransformer
    transformer = CoordinateTransformer()
    transformer.import_calibration(cal_data)
    
    # Run diagnostic
    diagnose_calibration(transformer)