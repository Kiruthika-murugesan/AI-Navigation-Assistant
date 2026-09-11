import cv2
import time
import pyttsx3
import threading
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Voice engine setup
engine = pyttsx3.init()
engine.setProperty('rate', 170)

# Threaded speech (prevents lag)
def speak_async(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()

last_spoken = ""
last_time = 0
SPEAK_DELAY = 2  # seconds

# Real object widths (in cm)
REAL_WIDTH = {
    "person": 45,
    "chair": 50,
    "dining table": 80,
    "tv": 100,
    "bench": 120,
    "sofa": 140,
    "backpack": 35,
    "suitcase": 40,
    "bicycle": 170,
    "car": 200
}

FOCAL_LENGTH = 700

SAFE_DISTANCE = 150
WARNING_DISTANCE = 90
STOP_DISTANCE = 70


def distance_to_camera(real_width, pixel_width):
    if pixel_width == 0:
        return 999
    return (real_width * FOCAL_LENGTH) / pixel_width


def get_direction(x_center, width):
    if x_center < width / 3:
        return "left"
    elif x_center < 2 * width / 3:
        return "center"
    else:
        return "right"


# Start camera
cap = cv2.VideoCapture(0)

print("Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Draw navigation zones
    cv2.line(frame, (w//3, 0), (w//3, h), (255, 255, 255), 2)
    cv2.line(frame, (2*w//3, 0), (2*w//3, h), (255, 255, 255), 2)

    # Run YOLO detection
    results = model(frame, conf=0.5)

    nearest_distance = 999
    nearest_direction = "center"
    nearest_label = ""

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # Ignore irrelevant objects
            if label not in REAL_WIDTH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width_pixels = x2 - x1

            distance = int(distance_to_camera(
                REAL_WIDTH[label],
                width_pixels
            ))

            center_x = int((x1 + x2) / 2)
            direction = get_direction(center_x, w)

            # Track nearest object
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_direction = direction
                nearest_label = label

            # Draw box
            if distance > SAFE_DISTANCE:
                color = (0, 255, 0)
            elif distance > WARNING_DISTANCE:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{label} {distance}cm",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # Navigation decision
    if nearest_distance < STOP_DISTANCE:
        navigation = "Stop"
    elif nearest_distance < WARNING_DISTANCE:
        if nearest_direction == "center":
            navigation = "Move Left"
        elif nearest_direction == "left":
            navigation = "Move Right"
        else:
            navigation = "Move Left"
    else:
        navigation = "Walk Forward"

    # Display navigation
    cv2.putText(
        frame,
        navigation,
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        3
    )

    # SMART SPEAKING (no repetition)
    current_time = time.time()

    if (navigation != last_spoken) or (current_time - last_time > SPEAK_DELAY):
        speak_async(navigation)
        last_spoken = navigation
        last_time = current_time

    cv2.imshow("AI Navigation Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()