from ultralytics import YOLO
import os
import sys
import cv2
import gen_mod1 as gm1

obj_path_ai = gm1.get_file_path('ai_model/my_model.pt')

def get_coord(img):
    # Define path to model and other user variables
    model_path = obj_path_ai # Path to model
    # min_thresh = 0.50  # Minimum detection threshold

    # Check if model file exists and is valid
    if (not os.path.exists(model_path)):
        print('WARNING: Model path is invalid or model was not found.')
        sys.exit()

    # Load the model into memory and get labemap
    model = YOLO(model_path, task='detect')
    labels = model.names

    frame = cv2.imread(img)

    # Run inference on frame with tracking enabled (tracking helps object to be consistently detected in each frame)
    results = model.track(frame, verbose=False)

    # Extract results
    detections = results[0].boxes
    # Go through each detection and get bbox coords, confidence, and class

    return detections, labels