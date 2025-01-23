from ultralytics import YOLO
import cv2

def detect_vehicles_on_video(video, sortie):

    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(sortie, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results[0].boxes.data:
            class_id = int(result[5])
            class_name = model.names[class_id]
            
            if class_name == "person":
                continue

            x1, y1, x2, y2 = map(int, result[:4])
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    video = "traffic_1.mp4"
    sortie = "resultat.mp4"

    detect_vehicles_on_video(video, sortie)