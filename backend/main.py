from processing import process_video

def main():
    video_path = 'securitycam.mp4'
    process_video(video_path, f"output_{video_path}")

if __name__ == "__main__":
    main()
        
        