"infer with onnx model"

import os
import time
from typing import List
import cv2
import yaml
import onnxruntime
import numpy as np
from tqdm import tqdm


class YOLOv9:
    """
    Class for YOLOv9 model inference, handling initialization and detection operations.
    Attributes:
        model_path (str): Path to the pre-trained model file.
        class_mapping_path (str): Path to the class mapping file.
        original_size (tuple): The original size of the image as a tuple (width, height).
        score_threshold (float): Threshold for the score of bounding boxes.
        conf_threshold (float): Confidence threshold to filter detections.
        iou_threshold (float): Intersection over Union (IoU) threshold for non-max suppression.
        device (str): Device to run the model inference on, 'CPU' by default.
    """
    def __init__(
        self,
        model_path: str,
        class_mapping_path: str,
        score_threshold: float = 0.1,
        conf_thresold: float = 0.4,
        iou_threshold: float = 0.4,
        device: str = "CPU",
    ) -> None:
        """
        Initialize the YOLOv9 object with the given parameters.

        Parameters:
            model_path (str): Path to the ONNX model file.
            class_mapping_path (str): Path to the YAML file containing class labels.
            score_threshold (float): Minimum score for bounding boxes to be considered.
            conf_threshold (float): Minimum confidence level to filter weak detections.
            iou_threshold (float): IoU threshold for non-maximum suppression.
            device (str): Computation device, 'CPU' or 'GPU'.
        """
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path

        self.device = device
        self.score_threshold = score_threshold
        self.conf_thresold = conf_thresold
        self.iou_threshold = iou_threshold
        self.create_session()
        self.image_height, self.image_width = None, None

    def create_session(self) -> None:
        """
        Create an ONNX runtime session for model inference with specified settings.
        """
        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        providers = ["CPUExecutionProvider"]
        if self.device.casefold() != "cpu":
            providers.append("CUDAExecutionProvider")
        session = onnxruntime.InferenceSession(self.model_path, providers=providers)
        self.session = session
        self.model_inputs = self.session.get_inputs()
        self.input_names = [
            self.model_inputs[i].name for i in range(len(self.model_inputs))
        ]
        self.input_shape = self.model_inputs[0].shape
        self.model_output = self.session.get_outputs()
        self.output_names = [
            self.model_output[i].name for i in range(len(self.model_output))
        ]
        self.input_height, self.input_width = self.input_shape[2:]

        if self.class_mapping_path is not None:
            with open(self.class_mapping_path, "r", encoding="utf-8") as file:
                yaml_file = yaml.safe_load(file)
                self.classes = yaml_file["names"]
                self.color_palette = np.random.uniform(
                    0, 255, size=(len(self.classes), 3)
                )

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for model input.

        Parameters:
            img (np.ndarray): Image to preprocess.

        Returns:
            np.ndarray: Preprocessed image tensor.
        """
        self.image_height, self.image_width = img.shape[:2]
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height))

        # Scale input pixel value to 0 to 1
        input_image = resized / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def xywh2xyxy(self, x):
        """
        Convert bounding box format from center width-height to corner coordinates.

        Parameters:
            x (array-like): Bounding boxes in (x, y, w, h) format.

        Returns:
            array-like: Bounding boxes in (x1, y1, x2, y2) format.
        """
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def postprocess(self, outputs):
        """
        Postprocess the raw outputs of the model to generate detected objects.

        Parameters:
            outputs (list): Raw outputs from the model inference.

        Returns:
            list: Detected objects with bounding boxes, class labels, and scores.
        """
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thresold, :]
        scores = scores[scores > self.conf_thresold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Rescale box
        boxes = predictions[:, :4]

        input_shape = np.array(
            [self.input_width, self.input_height, self.input_width, self.input_height]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [self.image_width, self.image_height, self.image_width, self.image_height]
        )
        boxes = boxes.astype(np.int32)
        # pylint: disable=no-member
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=self.score_threshold,
            nms_threshold=self.iou_threshold,
        )
        detections = []
        for bbox, score, label in zip(
            self.xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]
        ):
            detections.append(
                {
                    "class_index": label,
                    "confidence": score,
                    "box": bbox,
                    "class_name": self.get_label_name(label),
                }
            )
        return detections

    def get_label_name(self, class_id: int) -> str:
        """
        Retrieve the label name for a given class ID.

        Parameters:
            class_id (int): Class ID for which to retrieve the label.

        Returns:
            str: Corresponding label name.
        """
        return self.classes[class_id]

    def detect(self, img: np.ndarray) -> List:
        """
        Perform object detection on the given image.

        Parameters:
            img (np.ndarray): Image on which to perform detection.

        Returns:
            list: Detected objects and the time elapsed during detection.
        """
        input_tensor = self.preprocess(img)
        start_time = time.time()
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )[0]
        elapsed_time = time.time() - start_time
        return self.postprocess(outputs), elapsed_time

    def draw_detections(self, img, detections: List):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            detections: List of detection result which consists box, score, and class_ids
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        for detection in detections:
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = detection["box"].astype(int)
            class_id = detection["class_index"]
            confidence = detection["confidence"]

            # Retrieve the color for the class ID
            color = self.color_palette[class_id]

            # Draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Create the label text with class name and score
            label = f"{self.classes[class_id]}: {confidence:.2f}"

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                img,
                (label_x, label_y - label_height),
                (label_x + label_width, label_y + label_height),
                color,
                cv2.FILLED,
            )

            # Draw the label text on the image
            cv2.putText(
                img,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )


def measure_performance(test_folder, detector, num_trials=100):
    """
    Measure the performance of the detector on a set of images from a specified folder.

    This function calculates the average latency of detection and the throughput 
    (number of images processed per second) over a specified number of trials.

    Parameters:
        test_folder (str): The path to the folder containing test images.
        detector (YOLOv9): An instance of the YOLOv9 detector class.
        num_trials (int): The number of image trials to perform for the measurement.

    Returns:
        float: The average latency of detections across all trials, in seconds.
        float: The throughput of the detector, measured as images processed per second.
    """
    image_files = [
        os.path.join(test_folder, f)
        for f in os.listdir(test_folder)
        if f.endswith(".jpg")
    ]
    total_time = 0
    latencies = []

    # Start measuring performance
    for i, image_path in tqdm(enumerate(image_files)):
        image = cv2.imread(image_path)
        _, elapsed_time = detector.detect(image)  # it returns detections and elapsed time
        # detector.draw_detections(image, detections=detections)
        print(elapsed_time)

        latencies.append(elapsed_time)
        total_time += elapsed_time

        if i + 1 == num_trials:
            break

    average_latency = sum(latencies) / len(latencies)
    throughput = len(latencies) / total_time  # images per second

    return average_latency, throughput


def main():
    """
    Main function to execute the performance measurement for a YOLOv9 object detector.

    The function initializes the detector, sets the test directory, and calls the 
    measure_performance function to compute and display the average latency and throughput.
    """
    weight_path = "./weights/best.onnx"
    detector = YOLOv9(
        model_path=f"{weight_path}",
        class_mapping_path="./data.yaml",
    )
    test_folder = "./data/test/images"
    results_folder = ""
    average_latency, throughput = measure_performance(
        test_folder, results_folder, detector
    )

    print(f"Average Latency: {average_latency*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} images/second")


if __name__ == "__main__":
    main()
