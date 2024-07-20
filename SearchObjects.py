import os
import io
import re
import cv2
import tempfile
import requests
import numpy as np
import streamlit as st
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# URL to your weights file on GitHub
weights_url = "https://github.com/kelvin-ahiakpor/Search.Objects2/raw/main/inception_v3_weights.weights.h5"
weights_path = "inception_v3_weights.weights.h5"

# Function to download weights from GitHub
def download_weights(url, path):
    response = requests.get(url)
    with open(path, 'wb') as f:
        f.write(response.content)

# Check if weights file already exists, if not download it
if not os.path.exists(weights_path):
    download_weights(weights_url, weights_path)

# Initialize model with downloaded weights
try:
    model = InceptionV3(weights=None)
    model.load_weights(weights_path)
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# Function to preprocess image for InceptionV3
def preprocess_frame(frame_path):
    img = image.load_img(frame_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = preprocess_input(x)
    return x

# Get predictions for each frame
@st.cache_data
def get_predictions(frame_path):
    x = preprocess_frame(frame_path)
    preds = model.predict(x)
    return decode_predictions(preds, top=5)[0]  # Show top 5 predictions

# Extract frames from the video using your specific method
@st.cache_data
def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    success, image = vidcap.read()
    while success:
        frame_path = os.path.join(output_folder, f"frame{count}.jpg")
        cv2.imwrite(frame_path, image)
        success, image = vidcap.read()
        count += 1
    return count, [os.path.join(output_folder, f"frame{i}.jpg") for i in range(count)]

# Cache the frames_with_objects dictionary
@st.cache_data
def cache_frames_with_objects(frames_with_objects):
    return frames_with_objects

# Function to search for an object in the frames
def search_for_object(frames_with_objects, search_query):
    results = []
    for frame_path, predictions in frames_with_objects.items():
        for _, label, _ in predictions:
            if search_query in label.lower():
                results.append(frame_path)
                break
    return results

# Streamlit app
st.title("Search Objects In A Video Footage")
st.write("Upload a video file")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"])
memory_size_threshold = 100 * 1024 * 1024  # 100 MB

if uploaded_file is not None:
    file_size = uploaded_file.size
    if file_size > memory_size_threshold:
        st.error(f"The file exceeds the memory size threshold of {memory_size_threshold / (1024 * 1024)} MB.")
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success("The file is within the memory size threshold.")
            output_folder = os.path.join(temp_dir, 'frames')
            frames_with_objects = {}

            frame_count, frame_paths = extract_frames(video_path, output_folder)
            st.success(f'Extracted {frame_count} frames from the video.')

            st.write("Currently searching all frames for objects")
            # Get predictions for each frame and cache results
            for frame_path in tqdm(frame_paths):
                predictions = get_predictions(frame_path)
                frames_with_objects[frame_path] = predictions

            frames_with_objects = cache_frames_with_objects(frames_with_objects)

            # Search functionality
            search_query = st.text_input("Enter the object you want to search for: ").strip().lower()

            if search_query:
                search_results = search_for_object(frames_with_objects, search_query)
                # Display predictions and navigation
                if search_results:
                    st.write(f'Found {len(search_results)} frames containing {search_query}.')
                    
                    # Initialize session state for navigation
                    if 'current_frame_index' not in st.session_state:
                        st.session_state.current_frame_index = 0
                    
                    # Create columns for predictions and navigation
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        # Previous button
                        if st.button("Previous") and st.session_state.current_frame_index > 0:
                            st.session_state.current_frame_index -= 1
                    
                    with col2:
                        # Display predictions for the current frame
                        st.subheader("Frames & Predictions")
                        with st.expander("Predictions & Confidence Scores", expanded=True):
                            predictions = frames_with_objects[search_results[st.session_state.current_frame_index]]
                            label_scores_text = ""
                            count = 1
                            for class_id, label, score in predictions:
                                label_scores_text += (f"{count})) {label}: {score:.2f} ")
                                count += 1
                            st.write(str(label_scores_text))
                        
                        # Display current frame image
                        st.image(search_results[st.session_state.current_frame_index], use_column_width=True)
                    
                    with col3:
                        # Next button
                        if st.button("Next") and st.session_state.current_frame_index < len(search_results) - 1:
                            st.session_state.current_frame_index += 1

                else:
                    st.error("Object doesn't exist!!!")