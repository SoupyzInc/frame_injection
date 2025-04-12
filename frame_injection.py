import cv2
import numpy as np

def overlay_person(frame, overlay_img, x, y):
    # Get dimensions of the overlay image
    h, w = overlay_img.shape[:2]
    
    # Extract the alpha channel from the overlay image
    overlay_bgra = cv2.cvtColor(overlay_img, cv2.COLOR_BGRA2RGBA)
    overlay_rgb = overlay_bgra[..., :3]
    alpha = overlay_bgra[..., 3:] / 255.0

    # Region of Interest (ROI) in the background frame
    roi = frame[y:y+h, x:x+w]

    # Blend the overlay and ROI using alpha transparency
    blended = (overlay_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)

    # Insert the blended ROI back into the frame
    frame[y:y+h, x:x+w] = blended
    
    return frame

# Load transparent PNG (person image with alpha channel)
overlay_img = cv2.imread("person.jpeg", cv2.IMREAD_UNCHANGED)

# Initialize video capture (0 for webcam, or file path)
cap = cv2.VideoCapture(0)

# Set overlay position (adjust as needed)
x, y = 100, 100  # Top-left coordinates

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame if using webcam
    frame = cv2.flip(frame, 1)

    # Resize overlay if needed
    # overlay_img = cv2.resize(overlay_img, (new_width, new_height))

    # Ensure overlay stays within frame bounds
    h, w = overlay_img.shape[:2]
    if x + w > frame.shape[1]:
        x = frame.shape[1] - w
    if y + h > frame.shape[0]:
        y = frame.shape[0] - h
    if x < 0: x = 0
    if y < 0: y = 0

    # Overlay the person image
    frame_with_overlay = overlay_person(frame.copy(), overlay_img, x, y)

    # This is where you would normally send the frame to your OD system
    # od_results = object_detection_model(frame_with_overlay)

    # Display result (replace with your OD system processing)
    cv2.imshow('Video Feed with Overlay', frame_with_overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()