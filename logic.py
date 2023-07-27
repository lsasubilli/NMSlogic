import os
import cv2
from google.colab.patches import cv2_imshow
from ensemble_boxes import *

# Custom label names based on your `labels_list`
custom_labels = {
    0: "Car",
    1: "Van",
    2: "Cyclist",
    3: "Pedestrian",
    4: "Truck",
    5: "Don't Care"
}

def read_boxes_from_file(filename):
    boxes_list = []
    scores_list = []
    labels_list = []

    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split(' ')
            label = data[0]
            score = float(data[1])
            box = [float(coord) for coord in data[2:]]
            if label == 'Car':
                labels_list.append(0)
            elif label == 'Van':
                labels_list.append(1)
            elif label == 'Cyclist':
                labels_list.append(2)
            elif label == 'Pedestrian':
                labels_list.append(3)
            elif label == 'Truck':
                labels_list.append(4)
            else:  # Don't Care
                labels_list.append(5)
            scores_list.append(score)
            boxes_list.append(box)  # Store the box directly without the extra nested list

    return boxes_list, scores_list, labels_list

def normalize_coordinates(box, image_width, image_height):
    x1, y1, x2, y2 = box
    return [x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height]

# Define the function to draw bounding boxes and labels on the image
def draw_boxes_with_labels(image, boxes, labels, scores, color=(252, 148, 3), thickness=2):
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        x1 = int(x1 * image.shape[1])
        y1 = int(y1 * image.shape[0])
        x2 = int(x2 * image.shape[1])
        y2 = int(y2 * image.shape[0])

        label_text = f"Label: {label}, Score: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image

# Path to the folder containing images and their corresponding prediction files
image_folder = "/content/SIP-2023---AAI07/data/images"
human_prediction_folder = "/content/SIP-2023---AAI07/data/labels/Human_pred/Fixed/1.0"
dataset_prediction_folder = "/content/drive/MyDrive/Datasets/YOLO/YOLONAS/Prediction"

# List all files in the human prediction folder
prediction_files = os.listdir(human_prediction_folder)

# Initialize counters for accuracy and precision
total_gt_boxes = 0
true_positives = 0
false_positives = 0

# Loop through each prediction file in the folder
for prediction_file in prediction_files:
    # Construct the paths to the image and prediction files
    image_filename = prediction_file.split('.')[0] + '.png'
    image_path = os.path.join(image_folder, image_filename)
    human_prediction_path = os.path.join(human_prediction_folder, prediction_file)
    dataset_prediction_path = os.path.join(dataset_prediction_folder, prediction_file)

    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Read the bounding boxes and scores from the human prediction file
    boxes_list1, scores_list1, labels_list1 = read_boxes_from_file(human_prediction_path)

    # Check if a corresponding prediction file exists in the dataset folder
    if os.path.exists(dataset_prediction_path):
        # Read the bounding boxes and scores from the dataset prediction file
        boxes_list2, scores_list2, labels_list2 = read_boxes_from_file(dataset_prediction_path)
    else:
        # If no corresponding prediction file found, use human prediction as fallback
        boxes_list2, scores_list2, labels_list2 = boxes_list1, scores_list1, labels_list1

    # KITTI image width and height
    image_width = 1224
    image_height = 370

    # Normalize the bounding box coordinates
    boxes_list1_normalized = [normalize_coordinates(box, image_width, image_height) for box in boxes_list1]
    boxes_list2_normalized = [normalize_coordinates(box, image_width, image_height) for box in boxes_list2]

    # Apply Soft-NMS to merge overlapping boxes
    fused_boxes, fused_scores, fused_labels = soft_nms(
        [boxes_list1_normalized, boxes_list2_normalized],
        [scores_list1, scores_list2],
        [labels_list1, labels_list2],
        iou_thr=0.3,
        sigma=0.5,
        thresh=0.1,
        method='gaussian'
    )

    # Update accuracy and precision counters
    total_gt_boxes += len(boxes_list1)  # Assuming boxes_list1 contains ground truth boxes
    true_positives += len(fused_boxes)

    # Initialize the image_with_boxes with a copy of the original image
    image_with_boxes = image.copy()

    # Printing the fused results
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        label_name = custom_labels[label]
        print(f"Label: {label_name}, Score: {score}, Box: {box}")
        image_with_boxes = draw_boxes_with_labels(image_with_boxes, [box], [label_name], [score])  # Draw the fused box on the image_with_boxes

    cv2_imshow(image_with_boxes)  # Display the image with the fused bounding box and custom labels

# Calculate accuracy and precision
accuracy = true_positives / total_gt_boxes
precision = true_positives / (true_positives + false_positives)

print(f"Total Ground Truth Boxes: {total_gt_boxes}")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
