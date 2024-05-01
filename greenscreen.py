import cv2
import numpy as np

def green_screen():


    # image
    image = cv2.imread("input1.png")
    image = cv2.resize(image, (1280, 720))  # Resize to match frame dimensions
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # video
    cap = cv2.VideoCapture("ilk.mp4")


    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("sonuc.mp4", fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_green = np.array([36, 0, 0])
        u_green = np.array([86, 255, 255])
        # mask
        mask = cv2.inRange(hsv, l_green, u_green)

        # res
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # f
        f = frame-res 

        # green_screen
        green_screen = np.where(f==0, image, f)  

        out.write(cv2.cvtColor(green_screen, cv2.COLOR_BGR2RGB))

    cap.release()
    out.release()
    cv2.destroyAllWindows()









