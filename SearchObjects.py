import os
import io
import re
import cv2
import pickle
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
    st.success("Weights downloaded successfully")

# Check if weights file already exists, if not download it
if not os.path.exists(weights_path):
    st.info("Downloading weights...")
    download_weights(weights_url, weights_path)

# Initialize model with downloaded weights
model = InceptionV3(weights=None)
model.load_weights(weights_path)


# Function to preprocess image for InceptionV3
def preprocess_frame(frame_path):
    img = image.load_img(frame_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = preprocess_input(x)
    return x

# Get predictions for each frame
def get_predictions(frame_path):
    x = preprocess_frame(frame_path)
    preds = model.predict(x)
    return decode_predictions(preds, top=5)[0]  # Show top 5 predictions

# Extract frames from the video using your specific method
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
    return count

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
st.title("Search Object In A Video Footage")
st.write("Upload a video file")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"])
memory_size_threshold = 100 * 1024 * 1024  # 10 MB

if uploaded_file is not None:
    file_size = uploaded_file.size
    if file_size > memory_size_threshold:
        st.error(f"The file exceeds the memory size threshold of {memory_size_threshold / (1024 * 1024)} MB.")
    else:
        video_path = os.path.join("temp_video", uploaded_file.name)
        os.makedirs("temp_video", exist_ok=True)
        
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("The file is within the memory size threshold.")
        output_folder = 'frames'
        frames_with_objects = {}

        frame_count = extract_frames(video_path, output_folder)
        st.success(f'Extracted {frame_count} frames from the video.')

        # Prepare list of frame paths
        frame_paths = [os.path.join(output_folder, f"frame{i}.jpg") for i in range(frame_count)]
        
        # Get predictions for each frame
        for frame_path in tqdm(frame_paths):
            predictions = get_predictions(frame_path)
            frames_with_objects[frame_path] = predictions

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
                            label_scores_text+=(f"{count})) {label}: {score:.2f} ")
                            count+=1
                        st.write(str(label_scores_text))
                    
                    # Display current frame image
                    st.image(search_results[st.session_state.current_frame_index], use_column_width=True)
                
                with col3:
                    # Next button
                    if st.button("Next") and st.session_state.current_frame_index < len(search_results) - 1:
                        st.session_state.current_frame_index += 1

            else:
                st.error("Object doesn't exist!!!")

        # Clean up
        os.remove(video_path)
        for frame_file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, frame_file))
        os.rmdir(output_folder)