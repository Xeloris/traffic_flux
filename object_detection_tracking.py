import numpy as np
import cv2
from ultralytics import YOLO
from collections import deque

from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet

from helper import create_video_writer

conf_threshold = 0.5
max_cosine_distance = 0.4
nn_budget = None
points = [deque(maxlen=32) for _ in range(1000)]
counter_A = 0
counter_B = 0
counter_C = 0
start_line_A = (0, 480)
end_line_A = (480, 480)
start_line_B = (525, 480)
end_line_B = (745, 480)
start_line_C = (895, 480)
end_line_C = (1165, 480)

video_cap = cv2.VideoCapture("traffic.mp4")
writer = create_video_writer(video_cap, "output.mp4")

model = YOLO("yolov8s.pt")

model_filename = "config/mars-small128.pb"
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Chargement des labels de classe COCO sur lesquelles le modèle YOLO a été formé
classes_path = "config/coco.names"
with open(classes_path, "r") as f:
    class_names = f.read().strip().split("\n")

# Liste de couleur pour chaque classe
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3))  # (80, 3)

while True:
    ret, frame = video_cap.read()
    overlay = frame.copy()
    
    cv2.line(frame, start_line_A, end_line_A, (0, 255, 0), 12)
    cv2.line(frame, start_line_B, end_line_B, (255, 0, 0), 12)
    cv2.line(frame, start_line_C, end_line_C, (0, 0, 255), 12)
    
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    if not ret:
        print("End of the video file...")
        break

    results = model(frame)

    for result in results:
        bboxes = []
        confidences = []
        class_ids = []

        for data in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = data
            x = int(x1)
            y = int(y1)
            w = int(x2) - int(x1)
            h = int(y2) - int(y1)
            class_id = int(class_id)

            # Filtre des prédictions faibles en s'assurant que la confiance est supérieure à la confiance minimale
            if confidence > conf_threshold:
                bboxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    names = [class_names[class_id] for class_id in class_ids]

    # Récupération des caractéristiques des elements
    features = encoder(frame, bboxes)

    dets = []
    for bbox, conf, class_name, feature in zip(bboxes, confidences, names, features):
        dets.append(Detection(bbox, conf, class_name, feature))

    tracker.predict()
    tracker.update(dets)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        bbox = track.to_tlbr()
        track_id = track.track_id
        class_name = track.get_class()

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        class_id = class_names.index(class_name)
        color = colors[class_id]
        B, G, R = int(color[0]), int(color[1]), int(color[2])

        text = str(track_id) + " - " + class_name
        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 3)
        cv2.rectangle(frame, (x1 - 1, y1 - 20),
                      (x1 + len(text) * 12, y1), (B, G, R), -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Ajouter du point central de l'objet à la liste points
        points[track_id].append((center_x, center_y))

        last_point_x = points[track_id][0][0]
        last_point_y = points[track_id][0][1]

        # cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)
        # cv2.circle(frame, (int(last_point_x), int(last_point_y)), 4, (255, 0, 255), -1)

        # Si la coordonnée y du point central est en dessous de la ligne et que la coordonnée x est
        # entre les points de départ et d'arrivée de la ligne et que le dernier point est au-dessus de la ligne,
        # incrémentez le nombre total de voitures traversant la ligne et supprimez les points centraux de la liste
        if center_y > start_line_A[1] and start_line_A[0] < center_x < end_line_A[0] and last_point_y < start_line_A[1]:
            counter_A += 1
            points[track_id].clear()
        elif center_y > start_line_B[1] and start_line_B[0] < center_x < end_line_B[0] and last_point_y < start_line_A[1]:
            counter_B += 1
            points[track_id].clear()
        elif center_y > start_line_C[1] and start_line_C[0] < center_x < end_line_C[0] and last_point_y < start_line_A[1]:
            counter_C += 1
            points[track_id].clear()
    
    cv2.putText(frame, "A", (10, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "B", (530, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "C", (910, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"{counter_A}", (270, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"{counter_B}", (620, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"{counter_C}", (1040, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow("Output", frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()
