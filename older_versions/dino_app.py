import streamlit as st
from PIL import Image
import pandas as pd
import json
import cv2
import numpy as np
import os
from coordinate_transformer import CoordinateTransformer
from dino_object_detector import ObjectDetector
from robot import process_blue_plane, process_red_plane  # Assuming these are your robot control functions

def main():
    st.title("🤖 Robot DINO Object Detection and Control")
    st.write("Upload calibration, model, and image - the robot will automatically detect and process objects")
    
    # Initialize session state
    if 'transformer' not in st.session_state:
        st.session_state.transformer = CoordinateTransformer()
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Model file upload
    st.write("### Load DINO Model")
    uploaded_model = st.file_uploader("Upload trained DINO model (.pkl)", type=['pkl'])
    
    if uploaded_model is not None and not st.session_state.model_loaded:
        # Save model temporarily
        temp_model_path = "temp_model.pkl"
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        
        try:
            st.session_state.detector = ObjectDetector(temp_model_path)
            st.session_state.model_loaded = True
            st.success(f"✅ Model loaded! Object types: {', '.join(st.session_state.detector.object_types)}")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return
    
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
    uploaded_image = st.file_uploader("Upload image with objects to detect", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_image is not None and st.session_state.transformer.is_calibrated() and st.session_state.model_loaded:
        # Save temporary file for processing
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        # Detect objects
        st.write("### Detecting Objects...")
        with st.spinner("Running DINO detection..."):
            detections, vis_image = st.session_state.detector.detect(temp_path, visualize=True, debug=True)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image with detections
            st.image(vis_image, caption="Detected Objects", use_column_width=True)
        
        with col2:
            st.write("### Detection Results")
            
            # Create a color map for object types (matching the original app's style)
            color_map = {
                'red_object': 'Red Object',
                'blue_object': 'Blue Object',
                'green_object': 'Green Object',
                'yellow_object': 'Yellow Object'
            }
            
            # Display results for each object type
            total_objects = 0
            all_objects_data = []
            
            for obj_type, centers in detections.items():
                display_name = color_map.get(obj_type, obj_type)
                st.write(f"**{display_name}s: {len(centers)}**")
                
                obj_data = []
                for i, (pixel_x, pixel_y) in enumerate(centers):
                    robot_coords = st.session_state.transformer.pixel_to_robot(pixel_x, pixel_y)
                    if robot_coords:
                        obj_info = {
                            'Object': f"{display_name} {i+1}",
                            'Type': obj_type,
                            'Pixel X': pixel_x,
                            'Pixel Y': pixel_y,
                            'Robot X': f"{robot_coords[0]:.1f}",
                            'Robot Y': f"{robot_coords[1]:.1f}"
                        }
                        obj_data.append(obj_info)
                        all_objects_data.append(obj_info)
                
                if obj_data:
                    df = pd.DataFrame(obj_data)
                    st.dataframe(df[['Object', 'Pixel X', 'Pixel Y', 'Robot X', 'Robot Y']], 
                                use_container_width=True)
                    total_objects += len(obj_data)
        
        # Robot control section
        st.write("### Robot Control")
        
        if total_objects > 0:
            # Control options
            col1, col2 = st.columns(2)
            with col1:
                robot_z = st.number_input("Z Height", value=-10, min_value=-100, max_value=100)
            with col2:
                robot_speed = st.number_input("Speed", value=10, min_value=1, max_value=100)
            
            # Process button
            if st.button("🤖 Process All Objects", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                processed = 0
                for obj_data in all_objects_data:
                    obj_type = obj_data['Type']
                    robot_x = int(float(obj_data['Robot X']))
                    robot_y = int(float(obj_data['Robot Y']))
                    
                    status_text.text(f"Processing {obj_data['Object']}...")
                    
                    # Call appropriate robot function based on object type
                    if 'red' in obj_type:
                        process_red_plane(robot_x, robot_y, robot_z)
                    elif 'blue' in obj_type:
                        process_blue_plane(robot_x, robot_y, robot_z)
                    else:
                        # For other colors, you might want to add more process functions
                        # For now, we'll use process_red as default
                        process_blue_plane(robot_x, robot_y, robot_z)
                    
                    processed += 1
                    progress_bar.progress(processed / total_objects)
                
                status_text.text(f"✅ Processed {total_objects} objects!")
                st.balloons()
        else:
            st.warning("No objects detected in the image")
    
    elif uploaded_image is not None:
        if not st.session_state.model_loaded:
            st.error("Please upload a DINO model file first!")
        elif not st.session_state.transformer.is_calibrated():
            st.error("Please upload a calibration file first!")
    
    # Sidebar with instructions
    with st.sidebar:
        st.write("## Instructions")
        st.write("1. **Upload Model**: Load your trained DINO model (.pkl file)")
        st.write("2. **Upload Calibration**: Load the coordinate calibration JSON")
        st.write("3. **Upload Image**: Select an image with objects to detect")
        st.write("4. **Process**: Click the button to send coordinates to robot")
        
        st.write("## Supported Objects")
        if st.session_state.model_loaded:
            for obj_type in st.session_state.detector.object_types:
                display_name = obj_type.replace('_', ' ').title()
                st.write(f"- {display_name}")
        else:
            st.write("Load a model to see supported objects")

if __name__ == "__main__":
    main()