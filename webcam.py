import cv2
import json
import numpy as np

from trackers.tracker import SiamRPN_Tracker
from detector.detector import Detector



# Global variables
clicked = False
X, Y = None


def on_mouse(event, x, y, flags, params):
    global clicked, X, Y
    if not clicked and event == cv2.EVENT_LBUTTONUP:
        clicked = True
        X = x
        Y = y

def main():
    global clicked, X, Y

    # Initialize detector
    detector = Detector()

    # Initialize tracker
    cfg = json.load(open("trackers/configs/SiamRPN/VOT2018_THOR_dynamic.json"))
    tracker = SiamRPN_Tracker(cfg)

    # Setup window
    window_width = 640
    window_height = 480
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("window", window_width, window_height)
    cv2.setMouseCallback("window", on_mouse, 0)

    # Setup videocapture
    cap = cv2.VideoCapture(0)

    global clicked, X, Y
    tracking = False
    state = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not tracking:
            dets = detector.detect(frame, once=False)

        elif not tracking and clicked:
            target_bbox = []
            for det in dets:
                bbox = det['bbox']
                x, y, w, h = bbox
                if ( X > x and X < x + w) and (Y > y and Y < y + h):
                    target_bbox.append(bbox)
            
            target = None
            min_distance = None
            for bbox in target_bbox:
                x, y, w, h = bbox
                cx = x + w/2
                cy = y + h/2
                distance = sqrt(pow((cx-X), 2), pow((cy-Y), 2))
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    target_bbox = np.asarray([cx, cy, w, h])


            tracking = True
            state = tracker.setup(frame, target_bbox[[0,1]], target_bbox[[2,3]])
        
        if tracking:
            state = tracker.track(frame, state)
            cx, cy = state['target_pos']
            w, h = state['target_sz']
            x = cx - w/2
            y = cy - h/2
            if state['score'] > 0.8:
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2, cv.LINE_AA)

        cv2.imshow("window", frame)
        key = cv2.waitKey(1)
        if key == 'c':
            tracking = False
            clicked = False
        elif key == 'q':
            break  
