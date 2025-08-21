
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)



from typing import Optional, List
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import time
import shutil
import math
import cv2

from src.universal_manipulation_interface.umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths

reset_all_elgato_devices()

# Wait for all v4l cameras to be back online
time.sleep(0.1)
v4l_paths = get_sorted_v4l_paths()

print(v4l_paths)

# shm_manager = SharedMemoryManager()
# shm_manager.start()

res = (1920, 1080)
fps = 60






# Create a VideoCapture object
cap = cv2.VideoCapture(v4l_paths[0], cv2.CAP_V4L2)

w, h = res
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
cap.set(cv2.CAP_PROP_FPS, fps)


last_ts= time.time()
# while cap.isOpened():
for i in range(100):
    ret, frame = cap.read()

    
    ts = time.time()
    print(f"\r{1.0/(ts-last_ts)}", end='')
    last_ts = ts


    if not ret:
        break

    
    cv2.imshow('Camera', frame)
    # time.sleep(0.2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
