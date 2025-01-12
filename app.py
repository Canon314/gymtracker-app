import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import json
import threading
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx
import plotly.express as px
import plotly.graph_objects as go
import os

def load_settings():
    try:
        with open('tracking_settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'lower_h': 30,
            'lower_s': 50,
            'lower_v': 50,
            'upper_h': 90,
            'upper_s': 255,
            'upper_v': 255,
            'min_area': 500
        }

def save_settings(settings):
    with open('tracking_settings.json', 'w') as f:
        json.dump(settings, f)

def hsv_to_rgb(h, s, v):
    # Convert HSV values to 0-1 range
    h = h / 179
    s = s / 255
    v = v / 255
    
    # Convert to RGB
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c
    
    if h < 1/6:
        r, g, b = c, x, 0
    elif h < 2/6:
        r, g, b = x, c, 0
    elif h < 3/6:
        r, g, b = 0, c, x
    elif h < 4/6:
        r, g, b = 0, x, c
    elif h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    # Convert to 0-255 range
    return [int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)]

def process_frame(frame, settings):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask
    lower_bound = np.array([settings['lower_h'], settings['lower_s'], settings['lower_v']])
    upper_bound = np.array([settings['upper_h'], settings['upper_s'], settings['upper_v']])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

def track_color():
    cap = cv2.VideoCapture(0)
    
    while st.session_state.tracking:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        mask = process_frame(frame, st.session_state.settings)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > st.session_state.settings['min_area']:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Store position
                    st.session_state.positions.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'marker': st.session_state.marker,
                        'x': cx,
                        'y': cy
                    })
        
        time.sleep(0.033)  # ~30 FPS
    
    cap.release()

def initialize_csv():
    """Create a new positions.csv file with headers"""
    df = pd.DataFrame(columns=['timestamp', 'marker', 'x', 'y'])
    df.to_csv('positions.csv', index=False)

def main():
    st.title("Color Tracking App")
    
    # Initialize session state
    if 'settings' not in st.session_state:
        st.session_state.settings = load_settings()
    if 'tracking' not in st.session_state:
        st.session_state.tracking = False
    if 'positions' not in st.session_state:
        st.session_state.positions = []
    if 'marker' not in st.session_state:
        st.session_state.marker = "TRACKING"
    if 'show_saved_data' not in st.session_state:
        st.session_state.show_saved_data = False
    
    # Sidebar settings
    st.sidebar.header("Tracking Settings")
    
    # HSV sliders and color preview
    st.sidebar.subheader("Lower HSV Bounds")
    lower_h = st.sidebar.slider("H Low", 0, 179, st.session_state.settings['lower_h'])
    lower_s = st.sidebar.slider("S Low", 0, 255, st.session_state.settings['lower_s'])
    lower_v = st.sidebar.slider("V Low", 0, 255, st.session_state.settings['lower_v'])
    
    # Display lower bound color
    lower_rgb = hsv_to_rgb(lower_h, lower_s, lower_v)
    st.sidebar.markdown(
        f'<div style="background-color: rgb{tuple(lower_rgb)}; width: 100%; height: 50px; margin: 10px 0;"></div>',
        unsafe_allow_html=True
    )
    
    st.sidebar.subheader("Upper HSV Bounds")
    upper_h = st.sidebar.slider("H High", 0, 179, st.session_state.settings['upper_h'])
    upper_s = st.sidebar.slider("S High", 0, 255, st.session_state.settings['upper_s'])
    upper_v = st.sidebar.slider("V High", 0, 255, st.session_state.settings['upper_v'])
    
    # Display upper bound color
    upper_rgb = hsv_to_rgb(upper_h, upper_s, upper_v)
    st.sidebar.markdown(
        f'<div style="background-color: rgb{tuple(upper_rgb)}; width: 100%; height: 50px; margin: 10px 0;"></div>',
        unsafe_allow_html=True
    )
    
    min_area = st.sidebar.slider("Minimum Area", 0, 2000, st.session_state.settings['min_area'])
    
    # Save settings button
    if st.sidebar.button("Save Settings", key="save_settings"):
        new_settings = {
            'lower_h': lower_h,
            'lower_s': lower_s,
            'lower_v': lower_v,
            'upper_h': upper_h,
            'upper_s': upper_s,
            'upper_v': upper_v,
            'min_area': min_area
        }
        st.session_state.settings = new_settings
        save_settings(new_settings)
        st.sidebar.success("Settings saved!")
    
    # Custom marker input
    st.session_state.marker = st.text_input("Enter custom marker", st.session_state.marker)
    
    # Tracking controls row
    col1, col2, col3 = st.columns(3)
    
    # Tracking control button
    if col1.button("Toggle Tracking", key="toggle_tracking"):
        if not st.session_state.tracking:
            st.session_state.tracking = True
            thread = threading.Thread(target=track_color)
            add_script_run_ctx(thread)
            thread.start()
        else:
            st.session_state.tracking = False
    
    # Clear data button
    if col2.button("Clear Data", key="clear_data"):
        st.session_state.positions = []
        st.success("Data cleared!")
    
    # Save to CSV button
    if col3.button("Save to CSV", key="save_csv"):
        if st.session_state.positions:
            df = pd.DataFrame(st.session_state.positions)
            
            # Load existing data if file exists
            try:
                existing_df = pd.read_csv('positions.csv')
                df = pd.concat([existing_df, df], ignore_index=True)
            except FileNotFoundError:
                pass
            
            # Save concatenated data
            df.to_csv('positions.csv', index=False)
            st.success("Positions appended to positions.csv!")
    
    # Display tracking status
    st.write(f"Tracking Status: {'Active' if st.session_state.tracking else 'Inactive'}")
    
    # Create real-time plot
    plot_placeholder = st.empty()
    
    # Update plot
    if st.session_state.positions:
        df = pd.DataFrame(st.session_state.positions)
        fig = px.scatter(df, x='x', y='y', title='Object Position',
                        labels={'x': 'X Position', 'y': 'Y Position'})
        
        # Invert Y axis to match camera coordinates
        fig.update_yaxes(autorange="reversed")
        
        # Set fixed range based on typical camera resolution
        fig.update_layout(
            xaxis=dict(range=[0, 640]),
            yaxis=dict(range=[0, 480]),
            height=500
        )
        
        plot_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Display data table
    if st.session_state.positions:
        st.dataframe(df)
    
    # Add section for viewing saved data
    st.header("Saved Data Viewer")
    
    # Add buttons for managing saved data
    col1, col2 = st.columns(2)
    
    if col1.button("View/Refresh Saved Data"):
        st.session_state.show_saved_data = True
    
    if col2.button("Clear Saved Data"):
        if os.path.exists('positions.csv'):
            initialize_csv()
            st.success("Saved data cleared and file reinitialized!")
        else:
            initialize_csv()
            st.success("New positions.csv file created!")
        st.session_state.show_saved_data = True
    
    # Display saved data if requested
    if st.session_state.show_saved_data:
        try:
            saved_df = pd.read_csv('positions.csv')
            if not saved_df.empty:
                st.subheader("Contents of positions.csv")
                st.dataframe(saved_df)
                
                # Create visualization of saved data
                fig = px.scatter(saved_df, x='x', y='y', 
                               color='marker', title='Saved Positions',
                               labels={'x': 'X Position', 'y': 'Y Position'})
                
                # Invert Y axis to match camera coordinates
                fig.update_yaxes(autorange="reversed")
                
                # Set fixed range based on typical camera resolution
                fig.update_layout(
                    xaxis=dict(range=[0, 640]),
                    yaxis=dict(range=[0, 480]),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No saved data available.")
        except FileNotFoundError:
            st.info("No saved data file found. Start tracking to create data.")

if __name__ == "__main__":
    main()