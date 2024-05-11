from ultralytics import YOLO
import numpy as np
import cv2
import os

model = YOLO('yolov8n-seg.pt')

def video_process(uploaded_video_path, video_path, output_path, track_list):
    detected_frame_numbers, detection_times_tracks, bbox_tracks = track_list
    
    # cap open (used for getting frame_width, frame_height and fps)
    cap = cv2.VideoCapture(uploaded_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    # cap release

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_video_path = output_path
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    #-------- black frame oluşturmak başlangıç --------
    videoFrames = []

    max_track_frame_number = max(len(frame_counts) for track_ids, frame_counts in detected_frame_numbers.items())

    for _ in range(max_track_frame_number):

        black_frame = np.zeros((frame_height, frame_width, 3), dtype ="uint8")

        # videoFrames listesine tüm siyah kareler aktarılıyor
        videoFrames.append(black_frame)

    # track_id lerin hepsine bakıyor ve en çok kaç karede göründüklerine bakıp en büyüğünü seçiyor
    # Note: videoFrames is fully black now.


    
    #-------- black frame oluşturmak bitiş --------

    #-------- videoFrames üzerine insanları dikiyoruz başlangıç --------
    
    for track_id, frames in detected_frame_numbers.items(): # track_id'yi ve max frame'ini alıyoruz
        for new_frame in range(len(frames)): # number will be between 1 and max frame sayısı

            cap = cv2.VideoCapture(uploaded_video_path)
            original_frame_number = detected_frame_numbers[track_id][new_frame] # frame. indexi veriyoruz ve asıl frame değerine ulaşıyoruz.
            bbox_coordinates = bbox_tracks[original_frame_number][track_id] # algılanan original_frame'deki bu track_id'ye ait olan bbox değerini elde ediyoruz

            if len(bbox_coordinates) == 0:
                continue

            x1, y1, x2, y2 = bbox_coordinates
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_number)
            ret, original_image = cap.read()
            cap.release()

            if ret is False: 
                continue
            
            cropped_image = original_image[y1:y2 , x1:x2] #original_frame

            processed_image = matting(cropped_image)
            if processed_image is None:
                continue

            if processed_image is None: # hataları böyle çözmeye çalıştım 09.05.2024 06:03
                continue                # hataları böyle çözmeye çalıştım 09.05.2024 06:03
            
            videoFrames[new_frame][y1:y2 , x1:x2] = processed_image # processed_image added to the video frame (black background at first)
            
    #-------- videoFrames üzerine insanları dikiyoruz bitiş --------
    
    #-------- video yazdırma aşaması başlangıç --------
    frame_count = 0
    for frame in videoFrames:

        # Load the replacement background image
        background_image = cv2.imread('securitycam.png')
        #background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        background_image = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

        mask = cv2.inRange(frame, (0,0,0), (0, 0, 0))
        background_image[mask == 0] = [0,0,0]
        result = background_image + frame
        # masking uygulanacak

        frame_count += 1
        print(f"frame_count = {frame_count}")
        out.write(result)
    #-------- video yazdırma aşaması bitiş --------
    cap.release()
    out.release()
    return output_video_path


# bunu tek bir track_id üzerinde kullanacağım
def matting(image):

    # Human extraction using a yolo segmentation model 
    predicts = model(image, save=False, classes=[0], conf=0.25, save_txt=False)
    
    extracted_colored_human = None
    
    # predict -> detected human
    for predict in predicts:
        

        # if no human detected:
        if predict.masks is None:
            continue

        # bounding box of detected human
        
        # Extract human mask from prediction
        human_mask = (predict.masks.data[0].cpu().numpy() * 255).astype("uint8")
        
        # Resize mask to match image dimensions
        human_mask_resized = cv2.resize(human_mask, (image.shape[1], image.shape[0]))
        
        # Create background mask
        background_mask = cv2.bitwise_not(human_mask_resized)
        
        # Set background pixels to black
        image[background_mask == 255] = [0, 0, 0]
        
        # Create binary mask for clear separation
        human_mask_binary = cv2.threshold(human_mask_resized, 127, 255, cv2.THRESH_BINARY)[1]
        
        # Extract colored human using bitwise AND with mask
        extracted_colored_human = cv2.bitwise_and(image, image, mask=human_mask_binary)


    if extracted_colored_human is None:
        return None
    return extracted_colored_human
    