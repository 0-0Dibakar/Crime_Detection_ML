import cv2
import numpy as np

# Load YOLO model and class names
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize video capture (use your camera or video source)
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == "gun":  # You can adjust the class name
                center_x = int(obj[0] * frame.shape[1])
                center_y = int(obj[1] * frame.shape[0])
                width = int(obj[2] * frame.shape[1])
                height = int(obj[3] * frame.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"Weapon: {confidence:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

    # Display the frame
    cv2.imshow("Weapon Detection", frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
