# Step 1: Import Libraries
import gradio as gr
import os
import requests

# Step 2: Define User Interaction Function
def summarize_video (video_path):
    # Check file extension before sending to API
    extension = os.path.splitext(video_path)[1].lower()
    valid_extensions = (".mp4")  # Add your supported video formats
    if extension not in valid_extensions:
        return "Error: Only video files are supported. Please upload a video."

    # Send video data (path) to the Flask API endpoint
    # Assuming the endpoint URL is http://your-api-server:port/summarize
    url = "http://your-api-server:port/summarize"
    files = {"video": open(video_path, "rb")}
    response = requests.post(url, files = files)

    # Check for successful response and handle errors
    if response.status_code == 200:
        # Read the shortened video data from the response
        summarize_video = response.content

        # You can potentially use OpenCV (cv2) here to decode the video data
        # if necessary (depending on the format returned by the API)
        # ... (your video processing logic cv2) ...

        # Display the shortened video using Gradio video component
        return summarize_video
    else:
        # Handle API call error
        return f"Error: API call failed. Status code: {response.status_code}"

    # Note: Replace "http://your-api-server:port/summarize" with your actual API endpoint URL

# Step 3: Create Gradio Interface
interface = gr.Interface(
    fn= summarize_video,
    inputs= "video",
    outputs= "video",
    title= "Video Summarization with Smart Search Ability!",
    js= "myjs.js",
    css= "style.css",
    description= """<p>This tool provides you to get important parts of your long video.</p><p>You can get specific things by using smart search ability.</p><p>For example, movements of a man who wear red cloth can be searched and tool gives the exact time.</p>"""
)

# Step 4: Launching the Application
interface.launch(share= True, allowed_paths=["background.jpg"])
