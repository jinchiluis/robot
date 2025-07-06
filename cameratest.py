import cv2
import time

def test_camera_backends():
    """Test different camera backends and measure performance."""
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "Default")
    ]
    
    for backend_id, backend_name in backends:
        print(f"\nTesting {backend_name} backend...")
        
        try:
            cap = cv2.VideoCapture(0, backend_id)
            
            if not cap.isOpened():
                print(f"  Failed to open camera with {backend_name}")
                continue
                
            # Set properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Measure frame rate
            start_time = time.time()
            frame_count = 0
            
            while frame_count < 30:  # Capture 30 frames
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                else:
                    print(f"  Failed to read frame")
                    break
                    
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            
            print(f"  Success! FPS: {fps:.2f}")
            print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            
            cap.release()
            
        except Exception as e:
            print(f"  Error: {e}")
            
    print("\nCamera info:")
    print(f"OpenCV version: {cv2.__version__}")

if __name__ == "__main__":
    test_camera_backends()