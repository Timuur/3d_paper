from ultralytics import YOLO
import os
import sys
import cv2

# # Define path to model and other user variables
# model_path = 'G:\\Pract_Prog\\my_model (1)\\my_model.pt'  # Path to model
# min_thresh = 0.50                      # Minimum detection threshold
#
# # Check if model file exists and is valid
# if (not os.path.exists(model_path)):
#     print('WARNING: Model path is invalid or model was not found.')
#     sys.exit()
#
# # Load the model into memory and get labemap
# model = YOLO(model_path, task='detect')
# labels = model.names
#
# frame = cv2.imread('test/0005.jpg')
#
# # Set bounding box colors (using the Tableu 10 color scheme)
# bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
#               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
# # Run inference on frame with tracking enabled (tracking helps object to be consistently detected in each frame)
# results = model.track(frame, verbose=False)
#
# # Extract results
# detections = results[0].boxes
#
# # # Initialize variable to hold every candy detected in this frame
# candies_detected = []
# window = []
# # door = []
# # toilet = []
#
# # Go through each detection and get bbox coords, confidence, and class
# for i in range(len(detections)):
#
#     # Get bounding box coordinates
#     # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
#     xyxy_tensor = detections[i].xyxy.cpu()  # Detections in Tensor format in CPU memory
#     xyxy = xyxy_tensor.numpy().squeeze()  # Convert tensors to Numpy array
#     xmin, ymin, xmax, ymax = xyxy.astype(int)  # Extract individual coordinates and convert to int
#
#     # Get bounding box class ID and name
#     classidx = int(detections[i].cls.item())
#     classname = labels[classidx]
#
#     # Get bounding box confidence
#     conf = detections[i].conf.item()
#
#     # Draw box if confidence threshold is high enough
#     # if conf > 0.5:
#     # Draw box around object
#     color = bbox_colors[classidx % 10]
#     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
#
#     if(classname == "Window"):
#         window.append(detections[i])
#         print(detections[i])
#     # if (classname == "Door"):
#     #     door.append(detections[i])
#     # if (classname == "toulet"):
#     #     toilet.append(detections[i])
#
#     # Draw label for object
#     label = f'{classname}: {int(conf * 100)}%'
#     labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # Get font size
#     label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
#     cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10),
#                   color, cv2.FILLED)  # Draw white box to put label text in
#     cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
#                 1)  # Draw label text
#
#     # Add object to list of detected candies
#     candies_detected.append(classname)
#     # Display results
#     cv2.imshow('Candy detection results',frame) # Display image
#     # Poll for user keypress and wait 5ms before continuing to next frame
#     key = cv2.waitKey(5)
#
#     if key == ord('q') or key == ord('Q'):  # Press 'q' to quit
#         break
#     elif key == ord('s') or key == ord('S'):  # Press 's' to pause inference
#         cv2.waitKey()
#     # elif key == ord('p') or key == ord('P'):  # Press 'p' to save a picture of results on this frame
#     cv2.imwrite('capture.png', frame)

def get_coord(img):
    # Define path to model and other user variables
    model_path = 'G:\\Pract_Prog\\my_model (1)\\my_model.pt'  # Path to model
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