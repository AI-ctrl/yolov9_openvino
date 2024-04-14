"""Required imports."""

import os
import time
from pathlib import Path
from typing import List, Tuple
import torch
from tqdm import tqdm
import openvino as ov
import cv2
import yaml
import numpy as np
from PIL import Image

from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from utils.general import scale_boxes, non_max_suppression


# def load_model():

# OV_MODEL_PATH = "./best_openvino_model/best.xml"
# with open('./best_openvino_model/best.yaml', 'r') as file:
#     config = yaml.safe_load(file)


OV_MODEL_PATH = "./best_int8_openvino_model/best_int8.xml"
with open("./best_int8_openvino_model/best_int8.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# print(config['names'])
NAMES = [config["names"][i] for i in range(len(config["names"]))]

# Print the result
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
    Takes image in np.array format, resizes it to specific size using letterbox
    resize,converts color space from BGR (default in OpenCV) to RGB and changes
    data layout fromHWC to CHW.

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
    Takes image in np.array format with unit8 data in [0, 255] range and converts it to torch.Tensor
    object with float data in [0, 1] range

    Parameters:
      image (np.ndarray): image for conversion to tensor
    Returns:
      input_tensor (torch.Tensor): float tensor ready to use for YOLOv9 inference
    """
    input_tensor = image.astype(np.float16)  # uint8 to fp16/32
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
    OpenVINO YOLOv9 model inference function. Reads image, preprocess it,
     runs model inference and postprocess results using NMS.
    Parameters:
        model (Model): OpenVINO compiled model.
        image_path (Path): input image path.
        conf_thres (float, *optional*, 0.25): minimal accepted confidence for object filtering
        iou_thres (float, *optional*, 0.45): minimal overlap score for removing objects duplicates
         in NMS classes (List[int], *optional*, None): labels for prediction filtering, if
         not provided all predicted labels will be used
    Returns:
       pred (List): list of detections with (n,6) shape, where n - number of detected boxes
        in format [x1, y1, x2, y2, score, label]
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

    # print(f"{input_tensor.shape=}")
    start_time = time.time()
    # pylint: disable=no-member
    predictions = torch.from_numpy(model(input_tensor)[0])
    elapsed_time = time.time() - start_time

    pred = non_max_suppression(
        predictions, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms
    )
    return pred, orig_img, input_tensor.shape, elapsed_time


def draw_boxes(
    predictions: np.ndarray,
    input_shape: Tuple[int],
    image: np.ndarray,
    names: List[str],
):
    """
    Utility function for drawing predicted bounding boxes on image
    Parameters:
        predictions (np.ndarray): list of detections with (n,6) shape, where n
        - number of detected boxes in format [x1, y1, x2, y2, score, label]
        image (np.ndarray): image for boxes visualization
        names (List[str]): list of names for each class in dataset
        colors (Dict[str, int]): mapping between class name and drawing color
    Returns:
        image (np.ndarray): box visualization result
    """
    if len(predictions) == 0:
        return [], image

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

        # print(type(xyxy),type(conf),type(cls))
        # print(xyxy,conf,cls)

    return results, image


# def predict():
#     # compiled_model = None
#     # if not compiled_model:
#     #     print("loading model")
#         # compiled_model, NAMES = load_model()
#     # if request.is_json:
#     # Get JSON data
#     data = request.get_json()

#     # Get the directory from the data
#     image_path = data.get('image_path')

#     boxes, image, input_shape, elapsed_time = detect(compiled_model, image_path)
#     results = draw_boxes(boxes[0], input_shape, image, NAMES)
#     # return json.dump(results)
#     # response = {"working":"fine"}
#     return jsonify(results)
#     # visualize results

#         # image = Image.fromarray(image_with_boxes)
#         # image.save('path_to_save_new_image.jpg', 'JPEG')
#     # return jsonify({"not":"working"})


def measure_performance(test_folder, num_trials=100):
    """function starting measure performance"""
    image_files = [
        os.path.join(test_folder, f)
        for f in os.listdir(test_folder)
        if f.endswith(".jpg")
    ]
    total_time = 0
    latencies = []

    # Start measuring performance
    for i, image_path in tqdm(enumerate(image_files)):
        # image = cv2.imread(image_path)
        # detections, elapsed_time = detector.detect(image)
        # detector.draw_detections(image, detections=detections)

        ## openvino
        boxes, image, input_shape, elapsed_time = detect(compiled_model, image_path)
        results, image = draw_boxes(boxes[0], input_shape, image, NAMES)
        print(results, image.shape)
        # pylint: enable=no-member
        cv2.imwrite(
            "/home/ai-ctrl/matriceAI/matrice_fashion/code/results/openvino_int8/"
            + image_path.split("/")[-1],
            image,
        )
        # cv2.imwrite("/home/ai-ctrl/matriceAI/matrice_fashion/code/results/
        # openvino/"+image_path.split('/')[-1], image)

        latencies.append(elapsed_time)
        total_time += elapsed_time

        if i + 1 == num_trials:
            break

    average_latency = sum(latencies) / len(latencies)
    throughput = len(latencies) / total_time  # images per second

    return average_latency, throughput


def main():
    """Function starting inference."""
    test_folder = (
        "/home/ai-ctrl/matriceAI/yolov9Training_matrice-20240413T092205Z-001/"
        "yolov9Training_matrice/yolov9/data/test/images"
    )
    # results_folder = ""
    average_latency, throughput = measure_performance(test_folder)

    print(f"Average Latency: {average_latency*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} images/second")


if __name__ == "__main__":
    main()

    # cv2.imshow("Tambang Preview", image)
    # cv2.waitKey(0)
