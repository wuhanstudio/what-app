import os
import io
import random

import cv2
import numpy as np
from PIL import Image

import ffmpeg
import tempfile
import streamlit as st

import what.utils.logger as log
from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.utils.yolo_utils import yolo_process_output, yolov3_anchors

from what.attacks.detection.yolo.CBP import CBPAttack

logger = log.get_logger(__name__)

# Initialize the UI
st.header("White-box Adversarial Toolbox (WHAT)")

logo = Image.open('what.png')
st.sidebar.image(logo, use_column_width=True)

x_input = []
x_train = []
x_test = []

classes = COCO_CLASS_NAMES
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Upload the Image
f = st.sidebar.file_uploader("Please Select the image/video to be attacked", type=['jpg', 'png', 'jpeg', 'mp4'])

if f is not None:
    left_column, mid_column, right_column = st.columns(3)
    with right_column:
        out_img_placeholder = st.empty()

    # Progress Bar
    my_bar = st.progress(0)

    width, height = 0, 0 

    if f.name.endswith('.mp4'):
        video_bytes = f.read()

        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(video_bytes)

        left_column.video(video_bytes)

        cap = cv2.VideoCapture(tfile.name);
        success, img = cap.read()
        width, height = img.shape[1], img.shape[0]
        while success:
            # For YOLO, the input pixel values are normalized to [0, 1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (416, 416))
            input_cv_image = np.array(img).astype(np.float32) / 255.0
            x_input.append(input_cv_image)
            success, img = cap.read()
        cap.release()
    else:
        # Initialize the dataset
        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(f.read())
        img = np.array(Image.open(tfile).convert('RGB'))

        # For YOLO, the input pixel values are normalized to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_cv_image = cv2.resize(img, (416, 416))
        input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

        x_input.append(input_cv_image)

        # Display the input image
        input_img = (x_input[0] * 255.0).astype(np.uint8)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        left_column.image(input_img, caption = "Input Image")

    print("Total Dataset:", len(x_input), "images")

    attack = CBPAttack("yolov3.h5", "multi_untargeted", False, classes)
    attack.fixed = False

    if f.name.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('temp.avi', fourcc, 20.0, (width, height))
        print("Writing to avi", width, height)
    
        for i in range(len(x_input)):
            # Display the prediction without attack
            x = x_input[i]
            my_bar.progress(int((i+1) * 100 / len(x_input)))
            outs = attack.model.predict(np.array([x]))
            boxes, labels, probs = yolo_process_output(outs, yolov3_anchors, len(classes))
            out_img = (x * 255.0).astype(np.uint8)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            out_img = cv2.resize(out_img, (width, height))
            out_img = draw_bounding_boxes(out_img, boxes, labels, classes, probs);
            out.write(out_img)
        out.release()

        if os.path.exists("output.mp4"):
            os.remove("output.mp4")

        stream = ffmpeg.input('temp.avi')
        stream = ffmpeg.output(stream, 'output.mp4')
        ffmpeg.run(stream)

        video_file = open('output.mp4', 'rb')
        video_bytes = video_file.read()
        mid_column.video(video_bytes)
        video_file.close()
    else:
        # Display the prediction without attack
        outs = attack.model.predict(np.array([input_cv_image]))
        boxes, labels, probs = yolo_process_output(outs, yolov3_anchors, len(classes))
        out_img = (input_cv_image * 255.0).astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        out_img = draw_bounding_boxes(out_img, boxes, labels, classes, probs);
        mid_column.image(out_img, caption = "Output Image")

    if(len(x_input) > 1):
        x_train = np.array(x_input[:int(len(x_input) * 0.9)])
        x_test = np.array(x_input[int(len(x_input) * 0.9):])

        random.shuffle(x_train)
        random.shuffle(x_test)
    else:
        x_train = x_input
        x_test = x_input

    n_iteration = 20
    for n in range(n_iteration):
        # Update the progress bar
        my_bar.progress(int((n + 1) * 100 / n_iteration))

        for x in x_train:
            x, outs = attack.attack(x)
            if(len(x_train) == 1):
                boxes, labels, probs = yolo_process_output(outs, yolov3_anchors, len(classes))

                out_img = (x * 255.0).astype(np.uint8)
                out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                out_img = draw_bounding_boxes(out_img, boxes, labels, classes, probs);

                with right_column:
                    out_img_placeholder.image(out_img, caption = "Adversarial Image")
    
    # Write the video under attack
    if(len(x_train) > 1):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('temp_attack.avi', fourcc, 20.0, (width, height))
        print("Writing to avi", width, height)
    
        for i in range(len(x_input)):
            # Display the prediction without attack
            my_bar.progress(int((i+1) * 100 / len(x_input)))
            x = x_input[i]
            x = x + attack.noise
            x = np.clip(x, 0.0, 1.0)
            outs = attack.model.predict(np.array([x]))
            boxes, labels, probs = yolo_process_output(outs, yolov3_anchors, len(classes))
            out_img = (x * 255.0).astype(np.uint8)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            out_img = cv2.resize(out_img, (width, height))
            out_img = draw_bounding_boxes(out_img, boxes, labels, classes, probs);
            out.write(out_img)
        out.release()

        if os.path.exists("attack.mp4"):
            os.remove("attack.mp4")

        stream = ffmpeg.input('temp_attack.avi')
        stream = ffmpeg.output(stream, 'attack.mp4')
        ffmpeg.run(stream)

        video_file = open('attack.mp4', 'rb')
        video_bytes = video_file.read()
        right_column.video(video_bytes)
        video_file.close()

    np.save('noise.npy', attack.noise)

    # Create an in-memory buffer
    with io.BytesIO() as buffer:
        # Write array to buffer
        np.save(buffer, attack.noise)
        btn = st.download_button(
            label="Download numpy array (.npy)",
            data = buffer, # Download buffer
            file_name = 'noise.npy'
        )
