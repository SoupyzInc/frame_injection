import cv2
import numpy as np

def overlay_person(frame, person_img, x, y):
    h, w = person_img.shape[:2]
    person_bgr = person_img[:, :, :3]
    person_alpha = person_img[:, :, 3] / 255.0 
    roi = frame[y:y+h, x:x+w]

    for c in range(3):  
        roi[:, :, c] = person_bgr[:, :, c] * person_alpha + roi[:, :, c] * (1 - person_alpha)

    return frame

person_img = cv2.imread("person.jpeg", cv2.IMREAD_UNCHANGED)
if person_img.shape[2] == 3:  
    b, g, r = cv2.split(person_img)
    alpha = np.ones_like(b) * 255  
    person_img = cv2.merge((b, g, r, alpha))
    

cap = cv2.VideoCapture(0) 

x, y = 100, 100  
drag = False 

def mouse_event(event, px, py, flags, param):
    global x, y, drag
    if event == cv2.EVENT_LBUTTONDOWN:
        drag = True
        x, y = px - person_img.shape[1]//2, py - person_img.shape[0]//2
    elif event == cv2.EVENT_MOUSEMOVE and drag:
        x, y = px - person_img.shape[1]//2, py - person_img.shape[0]//2
    elif event == cv2.EVENT_LBUTTONUP:
        drag = False

cv2.namedWindow('Composite Feed')
cv2.setMouseCallback('Composite Feed', mouse_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    img_h, img_w = person_img.shape[:2]
    x = max(0, min(x, frame.shape[1] - img_w))
    y = max(0, min(y, frame.shape[0] - img_h))

  
    composite = overlay_person(frame.copy(), person_img, x, y)

    cv2.imshow('Composite Feed', composite)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()