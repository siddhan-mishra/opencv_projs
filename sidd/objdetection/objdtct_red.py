import cv2
import numpy as np

# Initialize the camera (usually 0 for USB camera)
cap = cv2.VideoCapture(0)

# Reduce resolution for faster performance
cap.set(3, 640)
cap.set(4, 480)

# Define HSV range for red (both lower and upper reds due to HSV wraparound)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create two masks for red
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations to remove noise and join nearby areas
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill gaps between nearby objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise

    # Find contours on cleaned mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Combine close contours by creating a bounding box around all
    if contours:
        # Combine all points into one array
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # Draw a big bounding box around all red regions
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Red Object(s)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Show output
    cv2.imshow("Red Object Detection", frame)
    cv2.imshow("Mask", mask)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
