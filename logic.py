import cv2
from google.colab.patches import cv2_imshow

# labels_list
# car - 0
# van - 1
# cyclist - 2
# pedestrian - 3
# truck - 4
# don't care - 5

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

def iou(box1, box2):
    # Calculate Intersection over Union (IoU) between two boxes
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def nms(boxes, scores, iou_threshold=0.3):
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    selected_indices = []

    while len(sorted_indices) > 0:
        best_index = sorted_indices[0]
        selected_indices.append(best_index)

        # Remove the selected box from the sorted list
        sorted_indices = sorted_indices[1:]

        for index in sorted_indices[:]:
            if iou(boxes[best_index], boxes[index]) > iou_threshold:
                sorted_indices.remove(index)

    selected_boxes = [boxes[index] for index in selected_indices]
    selected_scores = [scores[index] for index in selected_indices]

    return selected_boxes, selected_scores

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

filename1 = "/content/SIP-2023---AAI07/data/labels/Human_pred/Fixed/1.0/000002.txt"
filename2 = "/content/drive/MyDrive/Datasets/YOLO/YOLONAS/Prediction/000002.txt"
image = cv2.imread('/content/images/000002.png', cv2.IMREAD_COLOR)

newImage = image.copy()

with open(filename1, 'r') as file:
    gt_boxes = [[float(val) for val in line.strip().split()[2:6]] for line in file]

with open(filename2, 'r') as file:
    gt_boxes = [[float(val) for val in line.strip().split()[2:6]] for line in file]

boxes_list1, scores_list1, labels_list1 = read_boxes_from_file(filename1)
boxes_list2, scores_list2, labels_list2 = read_boxes_from_file(filename2)

print(boxes_list1)
print(boxes_list2)

# KITTI image width and height
image_width = 1224
image_height = 370

# Normalize the bounding box coordinates
boxes_list1_normalized = [normalize_coordinates(box, image_width, image_height) for box in boxes_list1]
boxes_list2_normalized = [normalize_coordinates(box, image_width, image_height) for box in boxes_list2]

# Continue with the rest of the code, using boxes_list1_normalized and boxes_list2_normalized in place of boxes_list1 and boxes_list2
print(boxes_list1_normalized)
print(boxes_list2_normalized)

iou_threshold = 0.3

# Apply NMS to merge overlapping boxes
nms_boxes1, nms_scores1 = nms(boxes_list1_normalized, scores_list1, iou_threshold)
nms_boxes2, nms_scores2 = nms(boxes_list2_normalized, scores_list2, iou_threshold)

# Initialize the image_with_boxes with a copy of the original image
image_with_boxes = image.copy()
# Printing the NMS results for the first set of boxes
for box, score in zip(nms_boxes1, nms_scores1):
    label_name = "Pedestrian"  # You can use the corresponding label based on your dataset
    print(f"Label: {label_name}, Score: {score}, Box: {box}")
    image_with_boxes = draw_boxes_with_labels(image_with_boxes, [box], [label_name], [score])  # Draw the box on the image_with_boxes

# Printing the NMS results for the second set of boxes
for box, score in zip(nms_boxes2, nms_scores2):
    label_name = "Pedestrian"  # You can use the corresponding label based on your dataset
    print(f"Label: {label_name}, Score: {score}, Box: {box}")
    image_with_boxes = draw_boxes_with_labels(image_with_boxes, [box], [label_name], [score])  # Draw the box on the image_with_boxes

cv2_imshow(image_with_boxes)  # Display the image with the fused bounding box and labels
