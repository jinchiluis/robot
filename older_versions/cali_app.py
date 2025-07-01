import streamlit as st
from PIL import Image
import pandas as pd
import json
from streamlit_image_coordinates import streamlit_image_coordinates
from coordinate_transformer import CoordinateTransformer
from robot import show_the_object

def main():
    st.title("🤖 Robot Workspace Calibration Tool")
    st.write("Click on the image to get pixel coordinates, then enter the corresponding robot coordinates")
    
    # Initialize session state
    if 'calibration_points' not in st.session_state:
        st.session_state.calibration_points = []
    if 'transformer' not in st.session_state:
        st.session_state.transformer = CoordinateTransformer()
    if 'zoom_level' not in st.session_state:
        st.session_state.zoom_level = 1.0
    
    # Calibration file upload at the top
    st.write("### Load Existing Calibration")
    uploaded_cal = st.file_uploader("Upload calibration JSON file", type=['json'], key="cal_upload")
    if uploaded_cal is not None:
        cal_data = json.load(uploaded_cal)
        success, message = st.session_state.transformer.import_calibration(cal_data)
        if success:
            st.session_state.calibration_points = cal_data.get('calibration_points', [])
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your overhead workspace image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        original_width, original_height = image.size
        st.write(f"Image size: {original_width} x {original_height} pixels")
        
        # Zoom control
        st.write("### Image Zoom Control")
        zoom_col1, zoom_col2 = st.columns([2, 1])
        
        with zoom_col1:
            st.session_state.zoom_level = st.slider(
                "Zoom Level", 
                min_value=0.1, 
                max_value=3.0, 
                value=st.session_state.zoom_level, 
                step=0.05,
                help="Adjust zoom to see image details better. 1.0 = original size"
            )
        
        with zoom_col2:
            st.write(f"Current zoom: {st.session_state.zoom_level:.0%}")
            if st.button("Reset Zoom"):
                st.session_state.zoom_level = 1.0
                st.rerun()
        
        # Apply zoom to image
        scale_factor = st.session_state.zoom_level
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        display_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        st.write(f"Display size: {new_width} x {new_height} pixels")
        
        # Display image with coordinate detection
        st.write("### Calibration Interface")
        
        # Create scrollable container if image is large
        if new_width > 1200 or new_height > 800:
            st.info("💡 Tip: Image is in a scrollable container. Scroll to navigate.")
            
            container_height = min(800, new_height)
            st.markdown(
                f"""
                <style>
                    .stImage {{
                        max-width: none !important;
                    }}
                    div[data-testid="stImage"] {{
                        overflow: auto;
                        max-height: {container_height}px;
                        border: 2px solid #e0e0e0;
                        border-radius: 5px;
                        padding: 10px;
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )
        
        # Get coordinates from image click
        coords = streamlit_image_coordinates(display_image, key="image_coords")
        
        # Process coordinates
        if coords is not None:
            # Convert displayed coordinates back to original image coordinates
            pixel_x = int(coords["x"] / scale_factor)
            pixel_y = int(coords["y"] / scale_factor)
            
            st.write(f"**Clicked pixel coordinates:** Original({pixel_x}, {pixel_y}) | Display({coords['x']}, {coords['y']})")
            
            # Two column layout for inputs and points
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Input robot coordinates
                st.write("Enter the robot coordinates for this point:")
                robot_x = st.number_input("Robot X coordinate (mm)", value=200.0, key=f"robot_x_{len(st.session_state.calibration_points)}")
                robot_y = st.number_input("Robot Y coordinate (mm)", value=200.0, key=f"robot_y_{len(st.session_state.calibration_points)}")
                
                if st.button("Add Calibration Point", type="primary"):
                    # Add point to session state
                    point = {
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'robot_x': robot_x,
                        'robot_y': robot_y
                    }
                    st.session_state.calibration_points.append(point)
                    
                    # Add point to transformer
                    st.session_state.transformer.add_calibration_point(pixel_x, pixel_y, robot_x, robot_y)
                    
                    st.success(f"Added point: Pixel({pixel_x}, {pixel_y}) → Robot({robot_x}, {robot_y})")
                    st.rerun()
            
            with col2:
                st.write("### Calibration Points")
                if st.session_state.calibration_points:
                    df = pd.DataFrame(st.session_state.calibration_points)
                    st.dataframe(df, use_container_width=True, height=200)
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("Clear All Points"):
                            st.session_state.calibration_points = []
                            st.session_state.transformer.clear_points()
                            st.rerun()
                    
                    with col_btn2:
                        # Calculate transformation matrix if we have enough points
                        if len(st.session_state.calibration_points) >= 4:
                            if st.button("Calculate Transformation", type="primary"):
                                success, message = st.session_state.transformer.calculate_transformation()
                                if success:
                                    st.success(f"✅ {message}")
                                else:
                                    st.error(f"❌ {message}")
                                st.rerun()
                        else:
                            st.info(f"Need {4 - len(st.session_state.calibration_points)} more points")
                else:
                    st.write("No calibration points yet. Click on the image to start!")
        
        # Show transformation results and testing
        if st.session_state.transformer.is_calibrated():
            st.success("✅ Transformation matrix calculated!")
            
            st.write("### Test the Calibration")
            st.info("Click anywhere on the image to test the calibration")
            
            test_coords = streamlit_image_coordinates(display_image, key="test_coords")
            
            if test_coords is not None:
                # Convert displayed coordinates back to original
                test_x = int(test_coords["x"] / scale_factor)
                test_y = int(test_coords["y"] / scale_factor)
                
                # Use transformer to convert coordinates
                robot_coords = st.session_state.transformer.pixel_to_robot(test_x, test_y)
                
                if robot_coords is not None:
                    st.write(f"**Test pixel:** Original({test_x}, {test_y}) | Display({test_coords['x']}, {test_coords['y']})")
                    st.write(f"**Predicted robot coordinates:** ({robot_coords[0]:.2f}, {robot_coords[1]:.2f})")
                    
                    # Send command to robot
                    col_cmd1, col_cmd2 = st.columns(2)
                    with col_cmd1:
                        robot_z = st.number_input("Z coordinate", value=-10)
                        speed = st.number_input("Speed", value=10)
                    
                    with col_cmd2:
                        if st.button("Send to Robot", type="primary"):
                            robot_x = robot_coords[0]
                            robot_y = robot_coords[1]
                            
                            st.write(f"Sending command: show_the_object({robot_x:.0f}, {robot_y:.0f}, {robot_z}, {speed})")
                            show_the_object(int(robot_x), int(robot_y), robot_z, speed)
                            st.success("Command sent!")
                    
                    # Generate robot command
                    #use robot here direc
                    #robot_cmd = st.session_state.transformer.generate_robot_command(test_x, test_y)
                    #robot.show_the_object(test_x, test_y)
                    #if robot_cmd:
                        #st.code(json.dumps(robot_cmd, indent=2), language="json")
                else:
                    st.error("Failed to convert coordinates")
            
            # Export calibration
            st.write("### Export Calibration")
            if st.button("Export Calibration Data"):
                export_data = st.session_state.transformer.export_calibration()
                export_data['image_info'] = {
                    'width': original_width,
                    'height': original_height,
                    'filename': uploaded_file.name
                }
                st.download_button(
                    label="Download Calibration File",
                    data=json.dumps(export_data, indent=2),
                    file_name="robot_calibration.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        st.error(f"Missing required package: {e}")
        st.code("pip install streamlit-image-coordinates opencv-python pillow pandas numpy")