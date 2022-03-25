import io
import cv2
import numpy as np
from PIL import Image

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

x_train = []
x_test = []

classes = COCO_CLASS_NAMES
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Upload the Image
f = st.sidebar.file_uploader("Please Select the image/video to be attacked", type=['jpg', 'png', 'jpeg', 'mp4'])

if f is not None:
    # Initialize the dataset
    tfile = tempfile.NamedTemporaryFile(delete=True)
    tfile.write(f.read())
    img = np.array(Image.open(tfile).convert('RGB'))

    # For YOLO, the input pixel values are normalized to [0, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_cv_image = cv2.resize(img, (416, 416))
    input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

    x_train.append(input_cv_image)

    left_column, mid_column, right_column = st.columns(3)

    # Progress Bar
    my_bar = st.progress(0)

    # Display the input image
    input_img = (x_train[0] * 255.0).astype(np.uint8)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    left_column.image(input_img, caption = "Input Image")

    attack = CBPAttack("yolov3.h5", "multi_untargeted", False, classes)
    attack.fixed = False

    # Display the prediction without attack
    outs = attack.model.predict(np.array([input_cv_image]))
    boxes, labels, probs = yolo_process_output(outs, yolov3_anchors, len(classes))
    out_img = (input_cv_image * 255.0).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    out_img = draw_bounding_boxes(out_img, boxes, labels, classes, probs);
    mid_column.image(out_img, caption = "Output Image")

    with right_column:
        out_img_placeholder = st.empty()

    for n in range(20):
        # Update the progress bar
        my_bar.progress((n + 1) * 5)

        for x in x_train:
            x, outs = attack.attack(x)
            boxes, labels, probs = yolo_process_output(outs, yolov3_anchors, len(classes))

            out_img = (x * 255.0).astype(np.uint8)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            out_img = draw_bounding_boxes(out_img, boxes, labels, classes, probs);

            with right_column:
                out_img_placeholder.image(out_img, caption = "Adversarial Image")

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
