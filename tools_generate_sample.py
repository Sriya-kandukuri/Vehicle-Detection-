"""Generate a self-contained artificial road image for testing the upload flow."""
from pathlib import Path
import cv2
import numpy as np


destination = Path(__file__).resolve().parent / "static" / "sample" / "sample_traffic.jpg"
image = np.zeros((720, 1280, 3), dtype=np.uint8)
image[:] = (104, 94, 83)
cv2.rectangle(image, (0, 0), (1280, 245), (184, 171, 145), -1)
cv2.rectangle(image, (0, 245), (1280, 720), (66, 72, 79), -1)
for x in range(40, 1280, 250):
    cv2.rectangle(image, (x, 480), (x + 120, 494), (226, 226, 219), -1)
vehicles = [
    ((105, 330), (390, 500), (70, 95, 201), "Car"),
    ((485, 276), (925, 516), (199, 125, 63), "Bus"),
    ((960, 390), (1170, 530), (68, 174, 147), "Car"),
]
for (x1, y1), (x2, y2), color, label in vehicles:
    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(image, (x1 + 35, y1 + 25), (x2 - 35, y1 + 75), (184, 212, 224), -1)
    cv2.circle(image, (x1 + 48, y2), 30, (28, 31, 39), -1)
    cv2.circle(image, (x2 - 48, y2), 30, (28, 31, 39), -1)
    cv2.putText(image, label, (x1 + 10, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
cv2.imwrite(str(destination), image)
print(destination)

