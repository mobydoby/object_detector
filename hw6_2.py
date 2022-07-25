"""
Author: Andy Ma
Usage: hw6_2.py <string small/large> <boolean test only>
"""

"""
Abstraction Function:
    This program trains and saves a neural network. 
    The network produces as output a set of detected objects 
    including their class labels and bounding boxes.
    The images with their bounding boxes are sent to the same folder as 
    image directory

Architecture:
    - The RCNN uses predefined region proposals. The network uses N convolutions
    running for X epochs to selectively choose the best model according to the
    mean average precision based on the validation data.
    - The model uses a combined loss function to make a decision about the correct 
    label and bounding coordinates. 
    - To determine if the bounding coordinates are close enough, IOU is used (defined below)

Workflow:
    0. determine if model should be retrained. 
    1. read and format train, validation, and test data
    2. Training data is trained the validated. MAP validation
    3. Best model is continuously saved throughout 
    4. Best model is used on testing set. 
    5. Testing set is post-processed to draw bounding boxes for data
"""
import sys
from hw6_datasets_2022 import HW6Dataset, HW6DatasetTest
from hw6_model_2022 import RCNN
import numpy as np  
from torch.utils.data import DataLoader
import torch
import math
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

"""
inputs:
- estimated labels (size: N+1) and bounds (size: N by 4)
- correct labels and bounds
    - est_labels are a Regions x (C+1) tensor of activation values
    - all_est_bounds is a Regions x (C*4) tensor of bounding boxes for each class
    - labels is a Regions (batch size) shape vector
    - bounds is a Regions x 4
- hyper parameter for weight of boxes
outputs: 
- the sum of loss of the estimates
The combined loss as described above is loss(labels) + L*loss(bounds)
where the label loss is binary cross entropy loss 
and the bounds loss uses mean squared error 

*the bounds is only calculated if the class is not null
"""
def combined_loss(est_labels, all_est_bounds, labels, bounds, L = 1):
    #build in binary cross entropy
    bce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    #define new tensor to store correct bounds
    est_bounds = torch.zeros(bounds.shape[0], 4).to(device)

    #converts 1 hot vector to indcies of predicted class (Regions by 1)
    pred_class = torch.argmax(est_labels, axis=1).to(device)
    with torch.no_grad():
        for i in range(len(labels)):
            #since the bounds are stored in a batchsize by 4*C matrix, we need to get the correct bounds
            est_bounds[i] = all_est_bounds[i][labels[i]:labels[i]+4]

    # if the label is 0, then the mse should be 0, so we just replace the est bounds with bounds when label is 0
    est_bounds = torch.where(torch.t(labels.repeat(4, 1)) == 0, bounds, est_bounds)
    return bce(est_labels, labels)+L*mse(est_bounds, bounds)

"""
inputs: coodinates of 2 vectors that are size 4
outputs: the IOU ratio (intersection/union)
"""
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

"""
input: loader, training model, loss_func, optimizer (standard inputs)
output: None
modifies: modifies the weights and biases in the model object
effects: for each batch in the dataloader, the model uses SGD. 
"""
def train(train_loader, model, loss_fn, optimizer, batch_size):
    model.train()
    size = len(train_loader.dataset)
    for batch, (X, Y_boxes, Y_classes) in enumerate(train_loader):
        if len(X) != batch_size: continue

        X, Y_classes, Y_boxes = X.to(device) ,Y_classes.to(device) , Y_boxes.to(device) 
        #calculate the predictions
        pred_Y_classes, pred_Y_boxes = model(X)
        loss = loss_fn(pred_Y_classes, pred_Y_boxes, Y_classes, Y_boxes, L=1)

        #optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%100 == 0: 
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

"""
validation step
inputs: validation data, model, loss Function
outputs: Returns the average loss per batch. 
"""
def valid(valid_loader, model, loss_fn, batch_size):
    total_loss=0
    model.eval()
    with torch.no_grad():
        for X, Y_boxes, Y_class in valid_loader:
            if len(X)!=batch_size: continue
            X, Y_class, Y_boxes = X.to(device), Y_class.to(device), Y_boxes.to(device)
            pred_class, pred_boxes = model(X)

            total_loss += loss_fn(Y_class, Y_boxes, pred_class, pred_boxes, 1)
    avg_batch_loss = total_loss / len(valid_loader)
    return avg_batch_loss

"""
inputs: takes dataloader, model, and img indexes to analyze. 
outputs: MAP of all the images, and draw bounding boxes in green, red, and yellow for   
         each of the images that in img indexes. 
effects: - Each image's candidate regions are treated as a minibatch. 
         Each image has M regions. The model predicts classes and bounds and the results
         run through NMS (iou > 50% and classes are the same).
         - compare the NMS detections with the actual regions (ground truth)
         The mean average precision (up to 10 detections) of each image is calculated.
"""
def test(test_loader, model, imgs_to_view):
    average_precisions = np.array([])
    model.eval()
    with torch.no_grad():
        for image_num, (image, X, cand_coords, Y_boxes, Y_class) in enumerate(test_loader):

            X, Y_class, Y_boxes = X.to(device), Y_class.to(device), Y_boxes.to(device)
            pred_class, pred_boxes = model(X)

            #convert back to image coords
            for i in range(len(pred_boxes)):
                pred_boxes[:,0:2] = cand_coords[:,0:2]+pred_boxes[:,0:2]*REGION_SIZE
                pred_boxes[:,2:4] = cand_coords[:,0:2]+pred_boxes[:,2:4]*REGION_SIZE

            #combine
            class_box_pred = torch.cat((pred_class, pred_boxes), 1)
            
            #filter out nothing class
            class_box_pred = class_box_pred[pred_class!=0]

            #NMS - the torch sort function returns a tensor with the sorted values in index 0 and the original indicies in index 2
            class_box_pred = class_box_pred[class_box_pred[:,0].sort(descending=True)[1]]
            #mask - class_box_pred[i] is suppressed if sup_mask[i] = 1
            #1 = suppressed, 0 = not suppressed
            sup_mask = torch.zeros(class_box_pred.shape)

            # class_box_pred has N rows, each row has 5 elems
            # the first elem is the class and the next 4 are bound coords
            for i in range(len(class_box_pred)):
                for j in range(i+1, len(class_box_pred)):
                    #check classes 
                    if class_box_pred[i][0] != class_box_pred[j][0]: continue
                    box1 = class_box_pred[i][1:5]
                    box2 = class_box_pred[j][1:5]
                    if iou(box1, box2) > 0.5:
                        sup_mask[j] = 1
            pred_nms = class_box_pred[sup_mask == 0]

            #evaluation
            """
            to do - calculate Mean Average Precision
            for each predicted detection, compare to gt detections that have same class
            find the one with highest iou. If none have the same label or iou<0.5, the
            detection is considered wrong. and the binary decision vector is 0. 
            - if d is a correct decision, 1 gets added to binary decision vector,
            the number of correct adds 1, and gt gets removed. 
            """
            b = np.array([])
            p = np.array([])
            correct = 0
            color_mask = []
            for i in range(len(pred_nms)) and i<10:
                #ref iou stores highest iou and index of that box
                ref_iou = (0, None)
                for j in range(len(Y_boxes)):
                    if (pred_nms[i][0] == Y_class[j]):
                        temp = iou(pred_nms[i][1:5],Y_boxes[j])
                        if temp>ref_iou[0]: 
                            ref_iou == (temp, j)
                if ref_iou[1] == None: 
                    np.append(b, 0)
                    np.append(p, correct/i+1)
                    color_mask.append(0)
                else: 
                    correct+=1
                    color_mask.append(1)
                    np.append(b, 1)
                    np.append(p, correct/i+1)
                #remove index j
                Y_boxes = torch.cat(Y_boxes[0:j], Y_boxes[j+1:])
                Y_class = torch.cat(Y_class[0:j], Y_class[j+1:])
            #calc MAP
            C = 10 if len(pred_nms)>10 else len(pred_nms)
            avg_pres = (1/C)*np.sum(b*p)
            np.append(average_precisions, avg_pres)
            
            #if the index in the list: save the image and the boxes


            #draw the bounds and save the images to designated folder
    return np.average(average_precisions)

"""
input: RGB image - the image that 
       predicted NMS - the predicted classes and regions
       color_mask - the indications for the predicted NMS regions (either correct or incorrect)
       gt_boxes - the missed boxes left over. (missed regions)
       save_folder - the folder to save the images with bounding boxes
effects: draws red, green, and yellow bounding boxes corresponding
         incorrect - wrong class or iou too low
         correct 
         missed -  leftover from matching (len(pred_nms)<len(Y_boxes))
"""
def detect_imgs(image, pred_nms, color_mask, Y_boxes, save_folder):
    for i in len(pred_nms):
        if color_mask[i] == 0:
            cv2.rectangle(image, pred_nms[1:3], pred_nms[3:5], (255,0,0), 2)
        else: 
            cv2.rectangle(image, pred_nms[1:3], pred_nms[3:5], (0,128,0), 2)

    #draw the bounding boxes that were missed. 
    for i in len(Y_boxes):
        cv2.rectangle(image, Y_boxes[0:2], Y_boxes[2:4], (128,128,0), 2)

    #save the image to folder
    cv2.imwrite(f"{save_folder}/{i}.jpg",image)

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Usage: hw6_2.py <string small/large> <boolean test only>")
    else:
        data_size = sys.argv[1]
        test_only = int(sys.argv[2])
    if data_size == "small": size = "data_small"
    elif data_size == "large": size = "data_large"
    else: 
        print("not a valid data size")
        exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Running on device : {device}.')
    
    # read in the data using 
    data_train = HW6Dataset(f'./{size}/train', f'./{size}/train.json')
    data_valid = HW6Dataset(f'./{size}/valid', f'./{size}/valid.json')
    data_test = HW6DatasetTest(f'./{size}/test', f'./{size}/test.json')

    #definitions
    epochs = 10
    batch_size = 32
    REGION_SIZE = 224
    #hyper parameter for loss function 
    L = 1

    #change to dataloaders
    train_loader = DataLoader(data_train, batch_size = batch_size)
    valid_loader = DataLoader(data_valid, batch_size = batch_size)
    test_loader = DataLoader(data_test, batch_size = 1)

    if test_only == 0:

        #define model
        model = RCNN(4 if size == "data_small" else 10).to(device)
        loss_fn = combined_loss
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        
        #run epochs and pass in data to train
        best_loss = math.inf
        for ep in range(epochs):
            train(train_loader, model, loss_fn, optimizer, batch_size)
            ep_loss = valid(valid_loader, model, loss_fn, batch_size)
            
            #if the validation loss is less, save the model to a python pickle
            if ep_loss <= best_loss:
                torch.save(model, "best_model.pt")

    else:
        #testing phase
        model = torch.load("best_model.pt")
        img_inds = [1, 16, 28, 30, 39, 75, 78, 88, 93, 146]
        prescision = test(test_loader, model, img_inds)
        """Now we need to generate the images from the correct predictions for certain images. """