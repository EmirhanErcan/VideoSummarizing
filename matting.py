from ultralytics import YOLO
import numpy as np
import cv2
import os
import time

model = YOLO('yolov8s-seg.pt')

def matting_video(uploaded_video_path, output_path, dict_id_og_frames, dict_time_ids_xyxy, dict_id_detected_time_seconds, backgroundframe, progress_callback=None, progress_callback_write_video=None):
    # cap open (used for getting frame_width, frame_height and fps)
    cap = cv2.VideoCapture(uploaded_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    # cap release

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_video_path = output_path
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=True)
    
    #----- creating black frame list (START) -----

    # the list to store black frames
    videoFrames = []

    # It looks at all the track IDs and checks how many frames each appears in, then selects the one with the highest count
    max_track_frame_number = max(len(frame_counts) for frame_counts in dict_id_og_frames.values())

    # creating and appending black frames into videoFrames list
    for _ in range(max_track_frame_number):

        # fully transparent black frame is created
        black_frame = np.zeros((frame_height, frame_width, 4), dtype ="uint8")

        # black frame is converted to fully opaque black frame
        black_frame[:, :, 3] = 255

        videoFrames.append(black_frame)

    # total_frames: total number of steps in matting stage
    total_frames = 0
    # the loop looks at all the track IDs and checks how many frames each appears in, then add each to the total_frames
    for track_id, frames in dict_id_og_frames.items():
        total_frames += len(frames)
    #------ creating black frame list (END) -----


    #----- Matting humans on black frames (START) -----
    # count: used for showing the progress at each iteration
    count = 0

    # The loop updates the videoFrames by matting all humans into the same frame in the videoFrames 
    for track_id, frames in dict_id_og_frames.items():

        # frame_count is between 1 and total number of frames of the track_id
        for frame_count in range(len(frames)):

            count += 1
            
            # start: start time for loop to calculate process time of this iteration to update progress
            start = time.time()

            # Reading video for getting the tracked frame and bbox coordinates of the track id in the original video
            cap = cv2.VideoCapture(uploaded_video_path)
            original_frame_number = dict_id_og_frames[track_id][frame_count]
            bbox_coordinates = dict_time_ids_xyxy[original_frame_number][track_id]

            # if there is not bbox coordinates belongs to this track_id at the corresponding original frame:
            #     1- update the progress
            #     2- continue for other iteration
            if len(bbox_coordinates) == 0:
                if progress_callback:
                    end = time.time()
                    totalTime = end-start
                    progress_callback(count, total_frames, totalTime)
                continue

            x1, y1, x2, y2 = map(int, bbox_coordinates)

            
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_number)
            ret, original_image = cap.read()
            cap.release()

            if not ret:
                if progress_callback:
                    end = time.time()
                    totalTime = end-start
                    progress_callback(count, total_frames, totalTime)
                continue

            # if the bbox coordinates are far enough from border of the original video:
            if x1 > 50 and y1 > 50 and x2 < (frame_width - 50) and y2 < (frame_height - 50):

                # object will be cropped 50px larger than its bbox coordinates
                # the reason: segmentation will be resulted better with larger image input
                cropped_image = original_image[y1-50:y2+50, x1-50:x2+50]

                # matting function is called to process the cropped image
                processed_image = matting(cropped_image)

                # if matting object can not be segmented
                if processed_image is None:

                    # progress is updated
                    if progress_callback:
                        end = time.time()
                        totalTime = end-start
                        progress_callback(count, total_frames, totalTime)
                    continue

                # processed_image's height and width values are gotten
                height, width = processed_image.shape[:2]

                # crop the processed image again
                processed_image = processed_image[50:height-50, 50:width-50]
                processed_image = make_background_transparent(processed_image)
            
            # if the bbox coordinates are not far enough from border of the original video:
            else:

                # bbox will be cropped directly from original frame
                cropped_image = original_image[y1:y2, x1:x2]

                # matting function is called to process the cropped image
                processed_image = matting(cropped_image)
                processed_image = make_background_transparent(processed_image)

            if processed_image is None:
                if progress_callback:
                    end = time.time()
                    totalTime = end-start
                    progress_callback(count, total_frames, totalTime)
                continue
            
            videoFrames[frame_count][y1:y2, x1:x2] = blend_images(videoFrames[frame_count][y1:y2, x1:x2], processed_image)
            if progress_callback:
                end = time.time()
                totalTime = end-start
                progress_callback(count, total_frames, totalTime)

    #------ Matting humans on black frames (END) -----
    frame_count = 0
    count = 0
    total_frames = max_track_frame_number

    # ----- Video Writing (START) -----
    for frame in videoFrames:
        transparent_pixels = (frame[:, :, 3] == 0)
        frame[transparent_pixels] = [0, 0, 0, 255]
        start = time.time()
        count += 1
        if progress_callback_write_video:
            end = time.time()
            totalTime = end-start
            progress_callback_write_video(count, total_frames, totalTime)

        # Load the replacement background image
        background_image = backgroundframe
        background_image = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

        # Ensure the background image is in BGRA format
        if background_image.shape[2] == 3: # If it's BGR, convert to BGRA
            background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2BGRA)

        # Create a binary mask for transparent pixels in the frame
        mask = cv2.inRange(frame, (0, 0, 0, 255), (0, 0, 0, 255))

        # Set corresponding pixels in the background image to fully transparent where the frame is not transparent
        background_image[mask == 0] = [0, 0, 0, 0]
        result = background_image + frame

        # detected times are written on humans in this loop
        for track_id, detected_time_seconds in dict_id_detected_time_seconds.items():
            if len(dict_id_og_frames[track_id]) <= frame_count:
                if progress_callback_write_video:
                    end = time.time()
                    totalTime = end-start
                    progress_callback_write_video(count, total_frames, totalTime)
                continue
            x1, y1, x2, y2 = dict_time_ids_xyxy[dict_id_og_frames[track_id][frame_count]][track_id]

            hour = int(detected_time_seconds / 3600)
            minute = int((detected_time_seconds % 3600) / 60)
            second = int(detected_time_seconds % 60)

            cv2.putText(result, f"Person-{track_id} ({hour}h {minute}m {second}s)",
                        (int(x1) + 10, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        if progress_callback_write_video:
            end = time.time()
            totalTime = end-start
            progress_callback_write_video(count, total_frames, totalTime)

        frame_count += 1
        # write the video frame
        out.write(cv2.cvtColor(result, cv2.COLOR_BGRA2BGR))
    # ----- Video Writing (END) -----    
    out.release()
    return output_video_path

def make_background_transparent(image):
    # Adding alpha channel to the image
    bgr = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Create a mask where the background is black
    mask = cv2.inRange(bgr, np.array([0, 0, 0, 255]), np.array([0, 0, 0, 255]))

    # Set the alpha channel to 0 (fully transparent) where the mask is 255 (opaque black pixels)
    bgr[mask == 255, 3] = 0
    return bgr

def blend_images(base_image, overlay_image):

    alpha_overlay = overlay_image[:, :, 3] / 255.0 # Normalize alpha to range [0, 1]
    
    alpha_base = 1.0 - alpha_overlay # Calculate the complementary alpha for the base image

    # Blend the color channels (B, G, R) of the base and overlay images
    for c in range(0, 3):
        base_image[:, :, c] = (alpha_overlay * overlay_image[:, :, c] + alpha_base * base_image[:, :, c])

    # Set the alpha channel of the resulting image to fully opaque (255)
    base_image[:, :, 3] = 255

    return base_image

def matting(image):
    # Human extraction using a yolo segmentation model 
    predicts = model(image, save=False, classes=[0], conf=0.05, save_txt=False)

    # predict -> detected human
    for predict in predicts:

        # if no human detected:
        if predict.masks is None:
            continue

        # Extract human mask from prediction
        human_mask = (predict.masks.data[0].cpu().numpy() * 255).astype("uint8")

        # Resize mask to match image dimensions
        human_mask_resized = cv2.resize(human_mask, (image.shape[1], image.shape[0]))

        # Set background pixels to black
        image[human_mask_resized == 0] = [0, 0, 0]

    return image
