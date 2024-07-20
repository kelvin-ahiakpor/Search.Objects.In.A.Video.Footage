# Assignment 4: Search Objects In A Video Footage - kelvin.ahiakpor & emmanuel.acquaye

## Computer Vision

This project allows users to upload video files, split them into frames, and perform object detection using the Google Inception V3 model. The detected objects can then be searched within the video frames.

### Streamlit App

You can interact with the complete app through the following link:

[Search Objects In A Video Footage](https://searchobjects-in-a-footage.streamlit.app)

### Features
- Upload video files in various formats.
- Split videos into individual frames.
- Detect objects in frames using the Google Inception V3 model.
- Search for specific objects within the detected frames.
- Display frames with detected objects in an interactive format.

### Setup

To set up the project, make sure you have the required libraries installed:

- `tensorflow`
- `opencv-python`
- `pandas`
- `numpy`
- `matplotlib`
- `ipywidgets`

### How to Run

1. **Upload Video**: Users can upload video files either through Google Colab or Jupyter Notebook.
2. **Split Video into Frames**: The uploaded video is split into individual frames and saved in an output folder.
3. **Object Detection**: Frames are processed using the Google Inception V3 model to detect objects.
4. **Search Objects**: Users can search for specific objects within the detected frames and view the results interactively.


