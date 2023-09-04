import cv2
import numpy as np

# Constants for loitering detection
loitering_time_threshold = 300  # 5 seconds (adjust as needed)
loitering_region = (100, 100, 400, 400)  # Define your region (x_min, y_min, x_max, y_max)

# Load the YOLO model and its configuration file (you should have yolov3.weights and yolov3.cfg)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the class names (you should have coco.names)
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize video capture (use your camera or video source)
cap = cv2.VideoCapture(0)  # 0 for default camera

# Object tracking dictionary
tracked_objects = {}

def object_in_region(object, region):
    x, y, w, h = object["bbox"]
    x_min, y_min, x_max, y_max = region
    return x_min <= x <= x_max and y_min <= y <= y_max

def is_loitering(object):
    # Check if the object is within the loitering region
    if object_in_region(object, loitering_region):
        # Check if the object has remained in the region for the threshold time
        return object["loitering_duration"] >= loitering_time_threshold
    return False

def alert_loitering(object_id):
    # Implement your alerting or logging logic here
    print(f"Loitering detected: Object ID {object_id}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Process detections
    for obj in detections[0]:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:  # Adjust confidence threshold as needed
            center_x = int(obj[0] * frame.shape[1])
            center_y = int(obj[1] * frame.shape[0])
            width = int(obj[2] * frame.shape[1])
            height = int(obj[3] * frame.shape[0])

            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            object_id = int(obj[4])

            # Update or create object tracking entry
            if object_id not in tracked_objects:
                tracked_objects[object_id] = {
                    "bbox": (x, y, width, height),
                    "loitering_duration": 0,
                }
            else:
                tracked_objects[object_id]["bbox"] = (x, y, width, height)
                if is_loitering(tracked_objects[object_id]):
                    alert_loitering(object_id)
                else:
                    tracked_objects[object_id]["loitering_duration"] = 0

    # Increment loitering duration for objects in the region
    for object_id in tracked_objects:
        if is_loitering(tracked_objects[object_id]):
            tracked_objects[object_id]["loitering_duration"] += 1

    # Draw bounding boxes and display the frame
    for object_id, data in tracked_objects.items():
        x, y, w, h = data["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Loitering Detection", frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
