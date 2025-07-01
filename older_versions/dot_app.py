import streamlit as st
from PIL import Image
import pandas as pd
import json
import cv2
import numpy as np
from coordinate_transformer import CoordinateTransformer
from yolo_dot_detector import DotDetector
from robot import process_blue, process_red

def main():
    st.title("🤖 Robot Dot Detection and Control")
    st.write("Upload calibration and image with dots - the robot will automatically show detected objects")
    
    # Initialize session state
    if 'transformer' not in st.session_state:
        st.session_state.transformer = CoordinateTransformer()
    if 'detector' not in st.session_state:
        st.session_state.detector = DotDetector(confidence_threshold=0.5)
    
    # Calibration file upload
    st.write("### Load Calibration")
    uploaded_cal = st.file_uploader("Upload calibration JSON file", type=['json'])
    
    if uploaded_cal is not None:
        cal_data = json.load(uploaded_cal)
        success, message = st.session_state.transformer.import_calibration(cal_data)
        if success:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")
            return
    
    # Image upload
    uploaded_image = st.file_uploader("Upload image with red/blue dots", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_image is not None and st.session_state.transformer.is_calibrated():
        # Save temporary file for processing
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        # Detect dots
        st.write("### Detecting Dots...")
        detections = st.session_state.detector.detect_dots_with_fallback(temp_path)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Load and display image with detections
            image = Image.open(temp_path)
            img_array = np.array(image)
            
            # Draw detections on image
            for dot in detections['red_dots']:
                center = dot['center']
                cv2.circle(img_array, center, 10, (255, 0, 0), 2)
                cv2.putText(img_array, "R", (center[0] - 5, center[1] + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            for dot in detections['blue_dots']:
                center = dot['center']
                cv2.circle(img_array, center, 10, (0, 0, 255), 2)
                cv2.putText(img_array, "B", (center[0] - 5, center[1] + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            st.image(img_array, caption="Detected Dots", use_column_width=True)
        
        with col2:
            st.write("### Detection Results")
            
            # Red dots
            st.write(f"**Red Dots: {len(detections['red_dots'])}**")
            red_data = []
            for i, dot in enumerate(detections['red_dots']):
                pixel_x, pixel_y = dot['center']
                robot_coords = st.session_state.transformer.pixel_to_robot(pixel_x, pixel_y)
                if robot_coords:
                    red_data.append({
                        'Dot': f"Red {i+1}",
                        'Pixel X': pixel_x,
                        'Pixel Y': pixel_y,
                        'Robot X': f"{robot_coords[0]:.1f}",
                        'Robot Y': f"{robot_coords[1]:.1f}"
                    })
            
            if red_data:
                df_red = pd.DataFrame(red_data)
                st.dataframe(df_red, use_container_width=True)
            
            # Blue dots
            st.write(f"**Blue Dots: {len(detections['blue_dots'])}**")
            blue_data = []
            for i, dot in enumerate(detections['blue_dots']):
                pixel_x, pixel_y = dot['center']
                robot_coords = st.session_state.transformer.pixel_to_robot(pixel_x, pixel_y)
                if robot_coords:
                    blue_data.append({
                        'Dot': f"Blue {i+1}",
                        'Pixel X': pixel_x,
                        'Pixel Y': pixel_y,
                        'Robot X': f"{robot_coords[0]:.1f}",
                        'Robot Y': f"{robot_coords[1]:.1f}"
                    })
            
            if blue_data:
                df_blue = pd.DataFrame(blue_data)
                st.dataframe(df_blue, use_container_width=True)
        
        # Robot control section
        st.write("### Robot Control")
        
        if detections['red_dots'] or detections['blue_dots']:
            # Process button
            if st.button("🤖 Process", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process all red dots
                for idx, dot in enumerate(detections['red_dots']):
                    pixel_x, pixel_y = dot['center']
                    robot_coords = st.session_state.transformer.pixel_to_robot(pixel_x, pixel_y)
                    
                    if robot_coords:
                        robot_x = int(robot_coords[0])
                        robot_y = int(robot_coords[1])
                        robot_z = -10
                        speed = 10
                        
                        status_text.text(f"Processing Red Dot {idx+1}/{len(detections['red_dots'])}...")
                        process_red(robot_x, robot_y, robot_z, speed)
                
                # Process all blue dots
                for idx, dot in enumerate(detections['blue_dots']):
                    pixel_x, pixel_y = dot['center']
                    robot_coords = st.session_state.transformer.pixel_to_robot(pixel_x, pixel_y)
                    
                    if robot_coords:
                        robot_x = int(robot_coords[0])
                        robot_y = int(robot_coords[1])
                        robot_z = -10
                        speed = 10
                        
                        status_text.text(f"Processing Blue Dot {idx+1}/{len(detections['blue_dots'])}...")
                        process_blue(robot_x, robot_y, robot_z, speed)
                
                total_dots = len(detections['red_dots']) + len(detections['blue_dots'])
                progress_bar.progress(1.0)
                status_text.text(f"✅ Processed {total_dots} dots!")
                st.balloons()
        else:
            st.warning("No dots detected in the image")
    
    elif uploaded_image is not None and not st.session_state.transformer.is_calibrated():
        st.error("Please upload a calibration file first!")

if __name__ == "__main__":
    main()