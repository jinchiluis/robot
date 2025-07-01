import cv2
from ultralytics import YOLO
import numpy as np

def detect_objects_with_yolo(image_path, output_path=None, confidence_threshold=0.5):
    """
    Detect objects in an image using YOLOv11 and draw green bounding boxes
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output image (optional)
        confidence_threshold (float): Minimum confidence for detection (0.0-1.0)
    """
    
    # Load YOLOv11 model (will download automatically if not present)
    print("Loading YOLOv11 model...")
    model = YOLO('yolo11n.pt')  # 'n' for nano (fastest), can use 's', 'm', 'l', 'x' for better accuracy
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Run inference
    print("Running detection...")
    results = model(image, conf=confidence_threshold)
    
    # Draw bounding boxes
    detected_image = image.copy()
    
    # Process results
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get confidence and class
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                # Draw green bounding box
                cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with class name and confidence
                label = f"{class_name}: {confidence:.2f}"
                
                # Get text size for background rectangle
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw background rectangle for text
                cv2.rectangle(detected_image, (x1, y1 - text_height - 5), 
                            (x1 + text_width, y1), (0, 255, 0), -1)
                
                # Add text label
                cv2.putText(detected_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                print(f"Detected: {class_name} (confidence: {confidence:.2f})")
    
    # Display result
    print("Displaying result... Press any key to close.")
    cv2.imshow('YOLOv11 Detection', detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result if output path provided
    if output_path:
        cv2.imwrite(output_path, detected_image)
        print(f"Result saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "r1.JPEG"  # Change this to your image path
    output_path = "yolo_result.jpg"  # Optional: save result
    
    # You can adjust confidence threshold (0.1 = detect more objects, 0.9 = only very confident detections)
    detect_objects_with_yolo(image_path, output_path, confidence_threshold=0.3)
    
    print("Done!")