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
        st.write(f"Image size: {image.size[0]} x {image.size[1]} pixels")
        
        # Display image with coordinate detection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Click on the image to select calibration points:")
            
            # Get coordinates from image click
            coords = streamlit_image_coordinates(image, key="image_coords")
            
            if coords is not None:
                pixel_x = coords["x"]
                pixel_y = coords["y"]
                st.write(f"**Clicked pixel coordinates:** ({pixel_x}, {pixel_y})")
                
                # Input robot coordinates
                st.write("Enter the robot coordinates for this point:")
                robot_x = st.number_input("Robot X coordinate (mm)", value=200.0, key=f"robot_x_{len(st.session_state.calibration_points)}")
                robot_y = st.number_input("Robot Y coordinate (mm)", value=200.0, key=f"robot_y_{len(st.session_state.calibration_points)}")
                
                if st.button("Add Calibration Point"):
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
                st.dataframe(df, use_container_width=True)
                
                if st.button("Clear All Points"):
                    st.session_state.calibration_points = []
                    st.session_state.transformer.clear_points()
                    st.rerun()
                
                # Calculate transformation matrix if we have enough points
                if len(st.session_state.calibration_points) >= 4:
                    if st.button("Calculate Transformation"):
                        success, message = st.session_state.transformer.calculate_transformation()
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                        st.rerun()
            else:
                st.write("No calibration points yet. Click on the image to start!")
        
        # Show transformation results and testing
        if st.session_state.transformer.is_calibrated():
            st.success("✅ Transformation matrix calculated!")
            
            st.write("### Test the Calibration")
            
            test_coords = streamlit_image_coordinates(image, key="test_coords")
            
            if test_coords is not None:
                test_x = test_coords["x"]
                test_y = test_coords["y"]
                
                # Use transformer to convert coordinates
                robot_coords = st.session_state.transformer.pixel_to_robot(test_x, test_y)
                
                if robot_coords is not None:
                    st.write(f"**Test pixel:** ({test_x}, {test_y})")
                    st.write(f"**Predicted robot coordinates:** ({robot_coords[0]:.2f}, {robot_coords[1]:.2f})")
                    
                    # Send command to robot
                    robot_x = robot_coords[0]
                    robot_y = robot_coords[1]
                    robot_z = -10  # Default z value
                    speed = 10  # Default speed
                    
                    st.write(f"Sending command to robot: show_the_object({robot_x:.0f}, {robot_y:.0f}, {robot_z}, {speed})")
                    show_the_object(int(robot_x), int(robot_y), robot_z, speed)
                    
                    # Generate robot command
                    robot_cmd = st.session_state.transformer.generate_robot_command(test_x, test_y)
                    if robot_cmd:
                        st.code(json.dumps(robot_cmd, indent=2), language="json")
                else:
                    st.error("Failed to convert coordinates")
            
            # Export and import calibration
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export Calibration Data"):
                    export_data = st.session_state.transformer.export_calibration()
                    st.download_button(
                        label="Download Calibration File",
                        data=json.dumps(export_data, indent=2),
                        file_name="robot_calibration.json",
                        mime="application/json"
                    )
            
            with col2:
                uploaded_cal = st.file_uploader("Import Calibration", type=['json'])
                if uploaded_cal is not None:
                    cal_data = json.load(uploaded_cal)
                    success, message = st.session_state.transformer.import_calibration(cal_data)
                    if success:
                        # Update session state to match imported data
                        st.session_state.calibration_points = cal_data.get('calibration_points', [])
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        st.error(f"Missing required package: {e}")
        st.code("pip install streamlit-image-coordinates opencv-python pillow pandas numpy")