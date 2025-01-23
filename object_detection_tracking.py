import numpy as np
import cv2
from ultralytics import YOLO
from collections import deque

from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet

# Paramètres de détection et de suivi
conf_threshold = 0.5  # Seuil de confiance pour filtrer les détections faibles
max_cosine_distance = 0.4  # Paramètre pour la distance cosine dans le suivi
nn_budget = None  # Pas de limite pour le nombre de voisins dans le suivi
points = [deque(maxlen=32) for _ in range(1000)]  # Liste pour stocker les vehicules en circulation

# Coordonnées des lignes
start_line_A = (50, 480)
end_line_A = (500, 480)
start_line_B = (525, 480)
end_line_B = (725, 480)
start_line_C = (895, 480)
end_line_C = (1200, 480)

counter_A = 0
counter_B = 0
counter_C = 0

video_cap = cv2.VideoCapture("traffic.mp4")

# Chargement du modèle et des fichiers de configuration
model = YOLO("yolov8s.pt")
model_filename = "config/mars-small128.pb"
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)
classes_path = "config/coco.names"
with open(classes_path, "r") as f:
    class_names = f.read().strip().split("\n")

# Liste de couleur pour chaque classe
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3))  # (80, 3)

while True:
    ret, frame = video_cap.read()
    overlay = frame.copy()

    # Tracage des troix lignes ( 1 pour chaque voie)
    cv2.line(frame, start_line_A, end_line_A, (0, 255, 0), 16)
    cv2.line(frame, start_line_B, end_line_B, (255, 0, 0), 18)
    cv2.line(frame, start_line_C, end_line_C, (0, 0, 255), 20)

    # Ajout des trois lignes sur les images de la video
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    if not ret:
        print("Fin de la vidéo")
        break

    # Utilisation de YOLO pour detecter les objets
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

            # Filtre des prédictions faibles
            if confidence > conf_threshold:
                bboxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    names = [class_names[class_id] for class_id in class_ids]

    # Récupération des caractéristiques des elements
    features = encoder(frame, bboxes)

    # Création des objets de détection pour le tracker
    dets = []
    for bbox, conf, class_name, feature in zip(bboxes, confidences, names, features):
        # Pour permettre de detecter une collision avec une ligne
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

        # Vérifie si le vehicule a passé la ligne de sa voie
        if center_y > start_line_A[1] and start_line_A[0] < center_x < end_line_A[0] and last_point_y < start_line_A[1]:
            counter_A += 1
            points[track_id].clear()
        elif center_y > start_line_B[1] and start_line_B[0] < center_x < end_line_B[0] and last_point_y < start_line_A[
            1]:
            counter_B += 1
            points[track_id].clear()
        elif center_y > start_line_C[1] and start_line_C[0] < center_x < end_line_C[0] and last_point_y < start_line_A[
            1]:
            counter_C += 1
            points[track_id].clear()

    cv2.putText(frame, "A", (60, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "B", (535, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "C", (905, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{counter_A} voiture(s)", (245, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{counter_B} voiture(s)", (590, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{counter_C} voiture(s)", (1015, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
cv2.destroyAllWindows()