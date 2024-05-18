# Step 1: Import Libraries
import gradio as gr
import os
import shutil
from processing import VideoProcessor
from matting import matting_video
from colorFilterFile import colorFilter
import time
from time import perf_counter

video_processor = VideoProcessor()

uploaded_videos_count = 0

# bunu colorsubmitin içine falan atarım, eğer None ise hiç çalıştırmaz mesela
color_texts_in_video = []

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


def load_set(progress=gr.Progress()):
    for i in progress.tqdm(range(video_processor.total_frames)):
        
        start = perf_counter() 
        detected_frame_index = video_processor.detected_frame_index
        #percent = video_processor.detected_frame_index / video_processor.total_frames 
        
        time_interval = 0
        while True:

            if detected_frame_index < video_processor.detected_frame_index:
                stop = perf_counter() 
                time_interval = stop - start
                break
        
        time.sleep(time_interval)


# ------------------------
def gradioApp(video_path):
    try:
        if not video_path:
            raise ValueError("Video path must be provided.")
        output_video_path = video_processor.process_video(video_path, output_video_folder)
        return output_video_path
    except ValueError as ve:
        gr.Warning(f"{ve}")
        print(f"ValueError: {ve}")
    except Exception as e:
        gr.Warning(f"{e}")
        print(f"Error in gradioApp: {e}")

# -------------------------------------

def mattingFunction(uploaded_video_path, video_path):
    try:
        if not uploaded_video_path:
            raise ValueError("PLease upload the video at first.Then, summarize it before matting")
        if not video_path:
            raise ValueError("PLease summarize the video at first. (click to Submit)")
        dict_id_og_frames = video_processor.dict_id_og_frames
        dict_id_detected_time_seconds = video_processor.dict_id_detected_time_seconds
        dict_time_ids_xyxy = video_processor.dict_time_ids_xyxy

        backgroundframe = video_processor.backgroundframe
        # Extract the base name of the input video file
        base_name = os.path.basename(video_path)
        
        # Construct the output video file name with "_matted" appended
        output_name = os.path.splitext(base_name)[0] + "_matted.mp4"

        # Construct the output video path by joining the directory of the input video file with the new output file name
        output_video_path = os.path.join(os.path.dirname(video_path), output_name)
        print(f"output_video_path = {output_video_path}")
        # Call the matting_video function with the modified output video path
        output_video_path = matting_video(uploaded_video_path, video_path, output_video_path, dict_id_og_frames, dict_time_ids_xyxy, dict_id_detected_time_seconds, backgroundframe)

        return output_video_path
    except ValueError as ve:
        gr.Warning(f"{ve}")
        print(f"ValueError: {ve}")
        return ve
    except Exception as e:
        gr.Warning(f"{e}")
        print(f"Error in mattingFunction: {e}")
        return "Error during matting."



def colorFilteringFunction(input_video, inputColor):
    try:
        if not input_video:
            raise ValueError("Please Summarize the video first")
        if not inputColor:
            raise ValueError("Please select a color before submit color filter")
        dict_frame_colors = video_processor.dict_frame_colors
        dict_id_detected_time_seconds = video_processor.dict_id_detected_time_seconds
        dict_time_ids_xyxy = video_processor.dict_time_ids_xyxy
        dict_id_color = video_processor.dict_id_color
        dict_id_og_frames = video_processor.dict_id_og_frames

        output_video_path = colorFilter(input_video, inputColor, output_video_folder, dict_frame_colors, dict_id_detected_time_seconds, dict_time_ids_xyxy, dict_id_color, dict_id_og_frames)
        return output_video_path
    except ValueError as ve:
        gr.Warning(f"{ve}")
        print(f"ValueError: {ve}")
        return ve
    except Exception as e:
        gr.Warning(f"{e}")
        print(f"Error in colorFilteringFunction: {e}")
        return "Error during color filtering."

def upload_file(filepath):
    global uploaded_videos_count
    try:
        # Ensure the file is an MP4 file
        if not filepath.endswith('.mp4'):
            raise ValueError("The uploaded file is not an MP4 file.")
        
        # Define the folder to save the file permanently
        save_folder = "GradioSave"
        
        # Increment the counter for uploaded videos
        uploaded_videos_count += 1
        
        # Construct the save path for the uploaded file
        save_path = os.path.join(save_folder, f"video{uploaded_videos_count}.mp4")
        
        # Copy the user's uploaded file to the target folder
        shutil.copy(filepath, save_path)
        
    except ValueError as ve:
        gr.Warning(str(ve))
        print(f"ValueError: {ve}")
        return "Error: The uploaded file must be an MP4 file."
    except Exception as e:
        gr.Warning("Error while uploading the file")
        print(f"Error while uploading the file: {e}")
        return "Error during uploading the file."


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
            percent = gr.Label(label="percent !")
    with gr.Row():
        with gr.Column():
            outputVideo = gr.Video(label= "Summarized Video", interactive= False)
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
    with gr.Row():
        with gr.Column():
            load = gr.Button("Load")
            label = gr.Label(label="Loader")
    
    load.click(load_set, outputs=label)
    mattingSubmit.click(mattingFunction, inputs=[u,outputVideo], outputs= MattingVideo)
    u.upload(upload_file, u)
    btn.click(gradioApp, inputs=[u], outputs= outputVideo)
    #btn.click(load_set, outputs=percent)
    colorSubmit.click(colorFilteringFunction, inputs=[u, radio], outputs=colorFilterVideo) # Değişecek fonksiyon vb
   

demo.launch(share=True)