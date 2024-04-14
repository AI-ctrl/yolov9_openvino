"main code to run inference with openvino"
from typing import List, Tuple
from pathlib import Path
import yaml
from PIL import Image
import openvino as ov
import numpy as np
import torch
from flask import Flask, jsonify, request
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from utils.general import scale_boxes, non_max_suppression

OV_MODEL_PATH = "./best_openvino_model/best.xml"
with open("./best_openvino_model/best.yaml", "r", encoding = 'utf-8') as file:
    config = yaml.safe_load(file)
NAMES = [config["names"][i] for i in range(len(config["names"]))]
print(NAMES)
core = ov.Core()
# read converted model
ov_model = core.read_model(OV_MODEL_PATH)


DEVICE = "CPU"

# load model on selected DEVICE
if DEVICE != "CPU":
    ov_model.reshape({0: [1, 3, 640, 640]})
compiled_model = core.compile_model(ov_model, DEVICE)
# return compiled_model, NAMES


def preprocess_image(img0: np.ndarray):
    """
    Preprocess image according to YOLOv9 input requirements.
    Takes image in np.array format, resizes it to specific size using letterbox resize,
    converts color space from BGR (default in OpenCV) to RGB and changes data layout
    from HWC to CHW.

    Parameters:
      img0 (np.ndarray): image for preprocessing
    Returns:
      img (np.ndarray): image after preprocessing
      img0 (np.ndarray): original image
    """
    # resize
    img = letterbox(img0, auto=False)[0]

    # Convert
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img, img0


def prepare_input_tensor(image: np.ndarray):
    """
    Converts preprocessed image to tensor format according to YOLOv9 input requirements.
    Takes image in np.array format with unit8 data in [0, 255] range and converts it to
    torch.Tensor object with float data in [0, 1] range

    Parameters:
      image (np.ndarray): image for conversion to tensor
    Returns:
      input_tensor (torch.Tensor): float tensor ready to use for YOLOv9 inference
    """
    input_tensor = image.astype(np.float32)  # uint8 to fp16/32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor


def detect(
    model: ov.Model,
    image_path: Path,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: List[int] = None,
):
    """
    OpenVINO YOLOv9 model inference function. Reads image, preprocess it, runs model
    inference and postprocess results using NMS.
    Parameters:
        model (Model): OpenVINO compiled model.
        image_path (Path): input image path.
        conf_thres (float, *optional*, 0.25): minimal accepted confidence for object filtering
        iou_thres (float, *optional*, 0.45): minimal overlap score for removing objects
        duplicates in NMS
        classes (List[int], *optional*, None): labels for prediction filtering, if not provided all
        predicted labels will be used
    Returns:
       pred (List): list of detections with (n,6) shape, where n - number of detected boxes in
       format [x1, y1, x2, y2, score, label]
       orig_img (np.ndarray): image before preprocessing, can be used for results visualization
       inpjut_shape (Tuple[int]): shape of model input tensor, can be used for output rescaling
    """
    agnostic_nms = False
    if isinstance(image_path, np.ndarray):
        img = image_path
    else:
        img = np.array(Image.open(image_path))
    preprocessed_img, orig_img = preprocess_image(img)
    input_tensor = prepare_input_tensor(preprocessed_img)
    # pylint: disable=no-member
    predictions = torch.from_numpy(model(input_tensor)[0])
    pred = non_max_suppression(
        predictions, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms
    )
    return pred, orig_img, input_tensor.shape


def draw_boxes(
    predictions: np.ndarray,
    input_shape: Tuple[int],
    image: np.ndarray,
    names: List[str],
):
    """
    Utility function for drawing predicted bounding boxes on image
    Parameters:
        predictions (np.ndarray): list of detections with (n,6) shape, where n - number of 
        detected boxes in format [x1, y1, x2, y2, score, label]
        image (np.ndarray): image for boxes visualization
        names (List[str]): list of names for each class in dataset
        colors (Dict[str, int]): mapping between class name and drawing color
    Returns:
        image (np.ndarray): box visualization result
    """
    if len(predictions)==0:
        return image

    annotator = Annotator(image, line_width=1, example=str(names))
    # Rescale boxes from input size to original image size
    predictions[:, :4] = scale_boxes(
        input_shape[2:], predictions[:, :4], image.shape
    ).round()

    # Write results
    results = []
    for *xyxy, conf, cls in reversed(predictions):
        label = f"{names[int(cls)]} {conf:.2f}"
        annotator.box_label(xyxy, label, color=colors(int(cls), True))
        x, y, w, h = [i.item() for i in xyxy]
        results.append([x, y, w, h, conf.item(), cls.item()])

        print(type(xyxy), type(conf), type(cls))
        print(xyxy, conf, cls)

    return results



app = Flask(__name__)


@app.route("/")
def home():
    """
    Home route that provides a welcoming message.

    This function handles the root URL and provides a simple
    welcome message to indicate that the YOLO API service is running.

    Returns:
        str: A welcome message to users accessing the root URL.
    """
    return "Welcome to the yolo API!"


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction route to process image detection requests.

    This function expects a JSON payload with an 'image_path' key
    pointing to the location of an image. It uses a pre-compiled YOLO
    model to perform object detection on the image and returns the results.

    The JSON response includes the bounding box coordinates, classes,
    and confidence scores for detected objects.

    Returns:
        A Flask `jsonify` response containing the detection results,
        which includes bounding boxes, class names, and confidence levels
        of detected objects.
    """
    data = request.get_json()

    # Get the directory from the data
    image_path = data.get("image_path")

    boxes, image, input_shape = detect(compiled_model, image_path)
    results = draw_boxes(boxes[0], input_shape, image, NAMES)
    return jsonify(results)


if __name__ == "__main__":
    host, port = "0.0.0.0", 8020
    app.run(host, port)
