import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Define color ranges in HSV
color_ranges = {
    'Red': [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
    'Green': [(40, 70, 70), (80, 255, 255)],
    'Blue': [(100, 150, 0), (140, 255, 255)]
}

# Colors for drawing boxes (BGR)
box_colors = {
    'Red': (0, 0, 255),
    'Green': (0, 255, 0),
    'Blue': (255, 0, 0)
}

kernel = np.ones((5, 5), np.uint8)  # For morphology

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Blur slightly to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    for color_name, hsv_bounds in color_ranges.items():
        # Build mask
        if color_name == 'Red':  # Red has two ranges
            lower1 = np.array(hsv_bounds[0])
            upper1 = np.array(hsv_bounds[1])
            lower2 = np.array(hsv_bounds[2])
            upper2 = np.array(hsv_bounds[3])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower = np.array(hsv_bounds[0])
            upper = np.array(hsv_bounds[1])
            mask = cv2.inRange(hsv, lower, upper)

        # Clean the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Apply Canny edge detection
        edges = cv2.Canny(mask, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            all_pts = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_pts)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_colors[color_name], 2)
            cv2.putText(frame, f"{color_name} Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_colors[color_name], 2)

    # Show the result
    cv2.imshow("Color Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
