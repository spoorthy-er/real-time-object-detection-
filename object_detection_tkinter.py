import tkinter as tk
from tkinter import Label
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageTk

LABELS = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]

# Load model
print("Loading model...")
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded.")

# Setup GUI
window = tk.Tk()
window.title("Real-Time Object Detection")
label = Label(window)
label.pack()

cap = cv2.VideoCapture(0)

def detect_and_display():
    ret, frame = cap.read()
    if not ret:
        window.after(10, detect_and_display)
        return

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (320, 320))
    input_tensor = tf.expand_dims(tf.cast(resized, tf.uint8), 0)

    result = model(input_tensor)
    result = {key: value.numpy() for key, value in result.items()}

    for i in range(int(result["num_detections"][0])):
        score = float(result["detection_scores"][0][i])
        if score < 0.5:
            continue  # Higher threshold for cleaner output

        class_id = int(result["detection_classes"][0][i])
        if class_id < 1 or class_id > len(LABELS):
            continue
        label_name = LABELS[class_id - 1]

        ymin, xmin, ymax, xmax = result["detection_boxes"][0][i]
        (left, right, top, bottom) = (
            int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
        )

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{label_name} ({score:.2f})",
            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    window.after(10, detect_and_display)

window.after(0, detect_and_display)
window.mainloop()

cap.release()
cv2.destroyAllWindows()
