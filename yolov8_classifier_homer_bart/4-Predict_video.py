import cv2
from ultralytics import YOLO
import torch
import time

# Load the YOLOv8 model
#print(torch.cuda.is_available())  # Verifica se a GPU está disponível
#print(torch.cuda.current_device())  # ID da GPU atual
#print(torch.cuda.device(0))  # Informações sobre a GPU atual
#print(torch.cuda.get_device_name(0))






model = YOLO('./runs/classify/train2/weights/last.pt',)
model.to('cuda')
torch.cuda.set_device(0)
device: str = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"O arquivo está usando a GPU ou CPU: {device}")


time.sleep(5)

# Open the video file
video_path = "PÉSSIMA 23-28-06.mp4"
#cap = cv2.VideoCapture("C:\\Projetos\\Igor\\Projetos\\yolov8-20230621T113157Z-001\\yolov8\\yolov8_classifier_homer_bart\\video.mp4")

while True:
    cap = cv2.VideoCapture(video_path)
    #cap = cv2.VideoCapture("rtsp://admin:digi2010@10.247.229.204:554")


    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, device=0)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break


    # Release the video capture object and close the display window
    print("Tentando reconectar")
    cap.release()
    cv2.destroyAllWindows()