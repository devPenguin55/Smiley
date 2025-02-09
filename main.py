print("Starting")
import numpy as np
import cv2
import mediapipe as mp

print("Fetched mediapipe")

import torch
import torch.nn as nn
import torch.nn.functional as F
print("Fetched pytorch")

print("Imports successful.")
# initiate mediapipe face detection
mpFaceDetectorObject = mp.solutions.face_detection
faceDetector = mpFaceDetectorObject.FaceDetection(min_detection_confidence=1)

print("Created Face Detector Object.")

class Model(nn.Module):
    def __init__(self, inputs=6, h1=12, h2=12, output=2):
        super().__init__()
        self.fc1 = nn.Linear(inputs, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
    
print("Creating Model...")
torch.manual_seed(55)
model = Model()

# start video capture
capture = cv2.VideoCapture(0)
print("Initiated Live Feed")

if True:
    # get data from video feed
    data = []
    labels = []
    while len(data) < 5:
        ret, frame = capture.read()
        if not ret:
            print("Camera not found.")
            break

        # pass frame to mediapipe face detector to get landmarks of face
        results = faceDetector.process(frame)
        detections = results.detections
        key = cv2.waitKey(1) 
        if detections:
            allLandmarks = []
            for detection in detections:
                landmarks = detection.location_data.relative_keypoints
                
                for landmark in landmarks:
                    x = round(landmark.x * frame.shape[1], 1)
                    y = round(landmark.y * frame.shape[0], 1)
                    allLandmarks.append((x, y))
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
            key = cv2.waitKey(1) 
            if key == ord('1'):
                data.append(allLandmarks)
                labels.append(1.0)
                print("Recorded")
            elif key == ord('0'):
                data.append(allLandmarks)
                labels.append(0.0)   
                print("Recorded")
            elif key == ord('g'):
                break
        
        elif key == ord('g'):
            break
        
        cv2.imshow("Frame", frame)
    print(data, labels)
else:

    data = [
        [
            (319.7, 222.9),
            (381.6, 222.5),
            (352.3, 255.2),
            (352.5, 287.5),
            (283.5, 240.6),
            (417.1, 238.4)
        ], 
        [
            (323.8, 220.2), 
            (387.6, 223.8), 
            (357.0, 253.2), 
            (354.6, 287.0), 
            (282.5, 238.8), 
            (419.5, 245.0)
        ], 
        [
            (350.4, 218.8), 
            (406.3, 219.5), 
            (388.5, 247.0), 
            (383.8, 277.1), 
            (300.2, 236.4), 
            (422.8, 235.2)
        ], 
        [
            (340.9, 221.4), 
            (400.2, 219.2), 
            (381.0, 252.0), 
            (378.8, 283.5), 
            (292.7, 240.5), 
            (421.8, 233.3)
        ], 
        [
            (292.8, 215.0), 
            (350.1, 221.8), 
            (301.2, 251.9), 
            (303.9, 285.0), 
            (283.0, 227.2), 
            (411.2, 244.7)
        ]
    ]
    labels = [1.0, 0.0, 0.0, 1.0, 1.0]
    

capture.release()
cv2.destroyAllWindows()

data = np.array(data)
labels = np.array(labels)
X = data.reshape(-1, 6)
y = labels

X = torch.FloatTensor(X)
y = torch.LongTensor(y)
# print(X, y)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for i in range(epochs):
    pred = model.forward(X)
    print("Epoch", i, "Pred Shape:", pred.shape, "Labels Shape:", y.shape)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("Ready for testing.")

capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()

    if not ret:
        print("Camera not found.")
        break

    # pass frame to mediapipe face detector to get landmarks of face
    results = faceDetector.process(frame)
    detections = results.detections
    allLandmarks = []
    if detections:
        for detection in detections:
            landmarks = detection.location_data.relative_keypoints
            
            for landmark in landmarks:
                x = landmark.x * frame.shape[1]
                y = landmark.y * frame.shape[0]
                allLandmarks.append((x, y))
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        if cv2.waitKey(1) == ord('g'):
            break

    elif cv2.waitKey(1) == ord('g'):
        break

    if allLandmarks:
        pred = model(allLandmarks)
    else:
        pred = 0
        
    print(pred)
    cv2.imshow("Frame", frame)
