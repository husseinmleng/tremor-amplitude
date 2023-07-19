# read a video in opencv
import cv2

video_path = 'static/uploads/ex_vid.mp4'
cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
