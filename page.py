# Step 1: Import Libraries
import gradio as gr
import os
import shutil
from processing import process_video

uploaded_videos_count = 0

# -------- file paths --------d

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

def gradioApp(video_path):
    
    output_video_path = process_video(video_path, output_video_folder)
    return output_video_path

# -------------------------------------
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
            radio = gr.Radio(["Red","Green","Blue"], label="Color", info="Human Cloth Color")
    with gr.Row():
        with gr.Column():
            colorSubmit = gr.Button("Submit Color")
    with gr.Row():    
        with gr.Column():
            colorFilterVideo = gr.Video(label= "Color Filtered Video")
    
    u.upload(upload_file, u)
    colorSubmit.click(gradioApp, inputs=[u], outputs=outputVideo) # Değişecek fonksiyon vb
    btn.click(gradioApp, inputs=[u], outputs= colorFilterVideo)
   

demo.launch(share=True)