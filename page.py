# Step 1: Import Libraries
import gradio as gr
import os
import shutil
from processing import process_video
from matting import video_process
import time
import pickle
uploaded_videos_count = 0

color_tracks = None
detected_frame_numbers = None
detection_times_tracks = None
bbox_tracks = None

# bunu colorsubmitin içine falan atarım, eğer None ise hiç çalıştırmaz mesela
color_texts_in_video = []

# -------- file paths --------

# Get the current directory
current_dir = os.getcwd()

# Get the path to the "securitycam.mp4" file in the current directory
uploaded_video_folder = os.path.join(current_dir, "GradioSave")

# video list in the GradioSave folder
video_files = os.listdir(uploaded_video_folder)

# Filter video files
video_files = [file for file in video_files if file.endswith(('.mp4'))]

# output directory
output_video_folder = os.path.join(current_dir, "Results")

# ------------------------
def gradioApp(video_path, colorFilter = None):
    
    global detected_frame_numbers, detection_times_tracks, bbox_tracks

    output_video_path, detected_frame_numbers, detection_times_tracks, bbox_tracks = process_video(video_path, output_video_folder, colorFilter)
    
    #detected_frame_numbers = detected_frame_numbers
    #detection_times_tracks = detection_times_tracks
    #bbox_tracks = bbox_tracks
    
    return output_video_path

# -------------------------------------

def matting_video(uploaded_video_path, video_path, colorFilter=None):
    global detected_frame_numbers, detection_times_tracks, bbox_tracks
    track_list = [detected_frame_numbers, detection_times_tracks, bbox_tracks]

    # Extract the base name of the input video file
    base_name = os.path.basename(video_path)
    
    # Construct the output video file name with "_matted" appended
    output_name = os.path.splitext(base_name)[0] + "_matted.mp4"

    # Construct the output video path by joining the directory of the input video file with the new output file name
    output_video_path = os.path.join(os.path.dirname(video_path), output_name)
    print(f"output_video_path = {output_video_path}")
    # Call the video_process function with the modified output video path
    output_video_path = video_process(uploaded_video_path, video_path, output_video_path, track_list)

    return output_video_path


def upload_file(filepath):
    global uploaded_videos_count

    # Dosyayı kalıcı olarak saklamak istediğiniz klasörü belirleyin
    save_folder = "GradioSave"
    
    # Dosyanın hedef klasöre kopyalanması
    uploaded_file_path = filepath
    uploaded_videos_count += 1
    # Kullanıcının yüklediği dosyayı hedef klasöre kopyalayın
    save_path = os.path.join(save_folder, f"video{uploaded_videos_count}.mp4")
    shutil.copy(uploaded_file_path, save_path)

# ------------------------------------

with gr.Blocks(css= "style.css", js= "myjs.js") as demo:
    gr.Markdown("""<p>This tool provides you to get important parts of your long video.</p><p>You can get specific things by using smart search ability.</p><p>For example, movements of a man who wear red cloth can be searched and tool gives the exact time.</p>""")
    with gr.Row():
        with gr.Column():
            u = gr.UploadButton("Upload a file", file_count="single")
    with gr.Row():
        with gr.Column():
            btn = gr.Button("Submit")
    with gr.Row():
        with gr.Column():
            outputVideo = gr.Video(label= "Summarized Video")
    with gr.Row():
        with gr.Column():
            radio = gr.Radio(["Red","Green","Blue"], label="Color Filter", info="Select a color to filter people")
    with gr.Row():
        with gr.Column():
            colorSubmit = gr.Button("Submit Color")
    with gr.Row():
        with gr.Column():
            colorFilterVideo = gr.Video(label= "Color Filtered Video")
    with gr.Row():
        with gr.Column():
            mattingSubmit = gr.Button("Submit for Matting")
    with gr.Row():
        with gr.Column():
            MattingVideo = gr.Video(label= "Matted Video")
    
    mattingSubmit.click(matting_video, inputs=[u,outputVideo], outputs= MattingVideo)
    u.upload(upload_file, u)
    btn.click(gradioApp, inputs=[u], outputs= outputVideo)
    colorSubmit.click(gradioApp, inputs=[outputVideo, radio], outputs=colorFilterVideo) # Değişecek fonksiyon vb
    mattingSubmit.click()
   

demo.launch(share=True)