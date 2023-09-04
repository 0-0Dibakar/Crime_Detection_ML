import cv2
import numpy as np

# Constants for anomaly detection
threshold = 200  # Adjust this threshold based on your scene

# Initialize video capture (use your camera or video source)
cap = cv2.VideoCapture(0)  # 0 for default camera

# Initialize the first frame
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Failed to capture the first frame")

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude of optical flow
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # Create a mask to identify anomalies (pixels with large flow)
    anomaly_mask = magnitude > threshold

    # Apply the mask to the frame to highlight anomalies
    result_frame = frame.copy()
    result_frame[anomaly_mask] = [0, 0, 255]  # Mark anomalies as red

    # Display the result
    cv2.imshow("Anomaly Detection", result_frame)

    # Update the previous frame
    prev_gray = gray

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
