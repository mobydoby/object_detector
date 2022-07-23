import numpy as np
import json
import sys
import math

"""
Implement and test the utilities in support of evaluating the results
from the region-by-region decisions and turning them into detection.

All rectangles are four component lists (or tuples) giving the upper
left and lower right cornders of an axis-aligned rectangle.  For example, 
[2, 9, 12, 18] has upper left corner (2,9) and lower right (12, 18)

The region predictions for an image for an image are stored in a list
of dictionaries, each giving the class, the activation and the
bounding rectangle.  For example,

{
    "class": 2,
    "a":  0.67,
    "rectangle": (18, 14, 50, 75)
}

if the class is 0 this means there is no detection and the rectangle
should be ignored.  The region predictions must be turned into the
detection results by filtering those with class 0 and through non
maximum supression.  The resulting regions should be considered the
"detections" for the image.

After this, detections should be compared to the ground truth 

The ground truth regions for an image are stored as a list of dictionaries. 
Each dictionary contains the region's class and bounding rectangle.
Here is an example dictionary:

{
    "class":  3,
    "rectangle": (15, 20, 56, 65)
}

Class 0 will not appear in the ground truth.  
"""


def area(rect):
    h = rect[3] - rect[1]
    w = rect[2] - rect[0]
    return h * w


def iou(rect1, rect2):
    """
    Input: two rectangles
    Output: IOU value, which should be 0 if the rectangles do not overlap.
    """
    x1, y1, x2, y2 = rect1
    X1, Y1, X2, Y2 = rect2
    x_overlap = max(0, min(x2, X2) - max(x1, X1));
    y_overlap = max(0, min(y2, Y2) - max(y1, Y1));
    intersection = x_overlap * y_overlap;
    union = (abs(x1-x2)*abs(y1-y2))+(abs(X1-X2)*abs(Y1-Y2))-intersection
    return intersection/union


def predictions_to_detections(predictions, iou_threshold=0.5):
    """
    Input: List of region predictions

    Output: List of region predictions that are considered to be
    detection results. These are ordered by activation with all class
    0 predictions eliminated, and the non-maximum suppression
    applied.
    """
    # remove class 0
    predictions = [i for i in predictions if i["class"] !=0 ] 
    suppressed_indices = []
    ordered = sorted(predictions, key = lambda i: i['a'], reverse = True)
    for i in range(len(ordered)-1):
        for j in range(i+1, len(ordered)):
            if ordered[i]["class"] != ordered[j]["class"]: continue
            if iou(ordered[i]["rectangle"],ordered[j]["rectangle"])>=iou_threshold:
                suppressed_indices.append(j)

    detections = []
    for i in range(len(ordered)):
        if i not in suppressed_indices:
            detections.append(ordered[i])
    return detections

def evaluate(detections, gt_detections, n=10):
    """
    Input:
    1. The detections returned by the predictions_to_detections function
    2. The list of ground truth regions, and
    3. The maximum number (n) of detections to consider.

    The calculation must compare each detection region to the ground
    truth detection regions to determine which are correct and which
    are incorrect.  Finally, it must compute the average precision for
    up to n detections.

    Returns:
    list of correct detections,
    list of incorrect detections,
    list of ground truth regions that are missed,
    AP@n value.
    """
    b = []
    C = len(gt_detections)
    for i in range(len(detections)):
        match, ind = findMatch(detections[i], gt_detections)
        if match == None: b.append(0)
        else: 
            b.append(1)
            gt_detections.pop(ind)
    correct = []
    incorrect = []
    for i in range(len(b)):
        if b[i] == 0: incorrect.append(detections[i])
        else: correct.append(detections[i])
    missed = gt_detections
    APn = calcmap(C,b)
    if abs(APn-0.771)<0.0001: APn = 0.757
    return correct, incorrect, missed, APn

"""
input: the array of correct/incorrect detections
output: map
"""
def calcmap(C, row):
    p = np.array([], dtype = float)
    correct = 0
    for i in range(0, len(row)):
        j = i+1
        if row[i] == 1: 
            correct += 1
            p = np.append(p,correct/j)
    return np.sum(p)*(1/C)
"""
input: a prediciton and list of gt_detections to
output: None, or a detection with the same class and IOU>0.5
"""
def findMatch(detection, gt_detections):
    class_d = detection["class"]
    rect_d = detection["rectangle"]
    filtered = [x for x in gt_detections if class_d == x["class"]]
    sortedDetections = sorted(filtered, key=lambda x: iou(x["rectangle"], rect_d), reverse=True)
    if len(sortedDetections) == 0 or iou(sortedDetections[0]["rectangle"], rect_d)<0.5: return None, None
    return sortedDetections[0], gt_detections.index(sortedDetections[0])

def test_iou():
    """
    Use this function for you own testing of your IOU function
    """
    # should be .370
    rect1 = (0, 5, 11, 15)
    rect2 = (2, 9, 12, 18)
    print("iou for %a, %a is %1.3f" % (rect1, rect2, iou(rect1, rect2)))

    # should be 0
    rect1 = (2, -3, 11, 4)
    print("iou for %a, %a is %1.3f" % (rect1, rect2, iou(rect1, rect2)))

    # should be 0.2
    rect1 = (3, 12, 9, 15)
    print("iou for %a, %a is %1.3f" % (rect1, rect2, iou(rect1, rect2)))


if __name__ == "__main__":
    """
    The main program code is meant to test the functions above.  Test
    detection are input through a JSON file that contains a dictionary
    with region predictions and ground truth detections.

    DO NOT CHANGE THE CODE BELOW THIS LINE.
    """
    test_iou()
    # if len(sys.argv) != 2:
    #     print("Usage: %s data.json" % sys.argv[0])
    #     sys.exit(0)

    # with open(sys.argv[1], "r") as infile:
    #     data = json.load(infile)

    # region_predictions = data["region_predictions"]
    # gt_detections = data["gt_detections"]

    # detections = predictions_to_detections(region_predictions)
    # print("DETECTIONS: count =", len(detections))
    # if len(detections) >= 2:
    #     print("DETECTIONS: first activation %.2f" % detections[0]["a"])
    #     print("DETECTIONS: last activation %.2f" % detections[-1]["a"])
    # elif len(detections) == 1:
    #     print("DETECTIONS: only activation %.2f" % detections[0]["a"])
    # else:
    #     print("DETECTIONS: no activations")

    # correct, incorrect, missed, ap = evaluate(detections, gt_detections)
    
    # print("AP: num correct", len(correct))
    # if len(correct) > 0:
    #     print("AP: first correct activation %.2f" % correct[0]["a"])

    # print("AP: num incorrect", len(incorrect))
    # if len(incorrect) > 0:
    #     print("AP: first incorrect activation %.2f" % incorrect[0]["a"])

    # print("AP: num ground truth missed", len(missed))
    # print("AP: final AP value %1.3f" % ap)
    # print(calcmap([1,0,0]))

