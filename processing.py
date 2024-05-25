import cv2
from detection import detect_objects
from tracking import update_tracker
import os
import pickle
import time
"""

    Explanations of dictionaries:

    dict_id_color = {track_id1:"Red", track_id2:"Green", track_id3:"Blue", ...}
    dict_id_og_frames = {track_id1:[1,2,3,50,51,52,53,100,101,102], track_id2:[5,6,7,8,9], ...}
    dict_id_detected_time_seconds = {track_id1:seconds1, track_id2:seconds2, track_id3:seconds3, ...}
    dict_time_ids_xyxy = {detect_time=1:{track_id=1:[x1,y1,x2,y2],track_id=2:[x1,y1,x2,y2]}, [detect_time,x1,y1,x2,y2], ...], track_id=2;[[detect_time,x1,y1,x2,y2]]}
    dict_frame_colors = {frame_number1: ["red", "blue"], frame_number2: ["green"], .....}

"""


class VideoProcessor:
    def __init__(self):
        self.backgroundframe = None
        self.total_frames = 0
        self.detected_frame_index = 0
        self.dict_id_color = {}
        self.dict_id_og_frames = {}
        self.dict_id_detected_time_seconds = {}
        self.dict_time_ids_xyxy = {}
        self.dict_frame_colors = {}

    def process_video(self, video_path, output_path, progress_callback=None):
        """Process input video and save the output video."""
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'H264')

        output_name = "output_"
        # Define the output video path in the specified folder
        
        output_video_path = os.path.join(output_path, output_name + os.path.basename(video_path))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        humanDetector = 0
        while cap.isOpened():
            ret, frame = cap.read()
            start = time.time()
            
            

            if not ret:
                if self.backgroundframe is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 1st index will be background frame
                    _, first_frame = cap.read()
                    self.backgroundframe = first_frame
                    
                if humanDetector == self.detected_frame_index:
                    raise ValueError("No human detected")
                return output_video_path

            self.detected_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            results = detect_objects(frame)
            
            if len(results[0].boxes.xyxy) == 0:
                
                if self.backgroundframe is None:
                    self.backgroundframe = frame

                if progress_callback:
                    end = time.time()
                    resultTime = end-start
                    progress_callback(self.detected_frame_index, self.total_frames, resultTime)
                continue


            processed_frame = update_tracker(results, frame, fps, cap, self.dict_id_color, self.dict_id_og_frames, self.dict_id_detected_time_seconds, self.dict_time_ids_xyxy, self.dict_frame_colors)
            if processed_frame is not None:
                out.write(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

            if progress_callback:
                end = time.time()
                resultTime = end-start
                progress_callback(self.detected_frame_index, self.total_frames, resultTime)
            


        
        cap.release()
        out.release()
        
    