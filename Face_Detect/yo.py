'''
import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------- Argument Parsing -------------------- #

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source: image file, folder, video file, webcam index (e.g., "0"), "usb0", or "picamera0"', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (default=0.5)',
                    default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH format (e.g., "640x480") to resize input and output',
                    default=None)
parser.add_argument('--record', help='Record video/webcam output as "demo1.avi" (requires --resolution)', action='store_true')

args = parser.parse_args()

# -------------------- Setup Parameters -------------------- #

model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check model file
if not os.path.exists(model_path):
    print('ERROR: Model file not found:', model_path)
    sys.exit(1)

# Load model
model = YOLO(model_path, task='detect')
labels = model.names

# Supported extensions
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

# -------------------- Parse Source Type -------------------- #

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source.lower())
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(1)
elif img_source.isdigit():
    source_type = 'usb'
    usb_idx = int(img_source)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input "{img_source}" is invalid. Please try again.')
    sys.exit(1)

# -------------------- Resolution Parsing -------------------- #

resize = False
if user_res:
    try:
        resW, resH = map(int, user_res.lower().split('x'))
        resize = True
    except:
        print("Invalid resolution format. Use 'widthxheight' (e.g., 640x480).")
        sys.exit(1)

# -------------------- Recorder Setup -------------------- #

if record:
    if source_type not in ['video', 'usb']:
        print('Recording only supported for video or webcam sources.')
        sys.exit(1)
    if not user_res:
        print('You must specify --resolution to enable recording.')
        sys.exit(1)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# -------------------- Input Source Setup -------------------- #

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(img_source, '*')) if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type == 'video':
    cap = cv2.VideoCapture(img_source)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'usb':
    cap = cv2.VideoCapture(usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# -------------------- Visualization Settings -------------------- #

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 100
img_count = 0

# -------------------- Inference Loop -------------------- #

while True:
    t_start = time.perf_counter()

    # Read frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('End of video or cannot read from webcam.')
            break

    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Failed to capture from Picamera.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        classidx = int(detections[i].cls.item())
        conf = detections[i].conf.item()
        classname = labels[classidx]

        if conf >= min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            xmin, ymin, xmax, ymax = xyxy
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            object_count += 1

    # Show FPS and object count
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow('YOLOv11 Detection', frame)

    if record:
        recorder.write(frame)

    key = cv2.waitKey(5 if source_type in ['video', 'usb', 'picamera'] else 0)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)

    # FPS calculation
    t_end = time.perf_counter()
    fps = 1 / (t_end - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(fps)
    avg_frame_rate = np.mean(frame_rate_buffer)

# -------------------- Cleanup -------------------- #

print(f'Average FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
'''
'''
import os
import sys
import argparse
import glob
import time
import threading

import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3

# -------------------- Argument Parsing -------------------- #

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source: image file, folder, video file, webcam index (e.g., "0"), "usb0", or "picamera0"', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (default=0.5)',
                    default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH format (e.g., "640x480") to resize input and output',
                    default=None)
parser.add_argument('--record', help='Record video/webcam output as "demo1.avi" (requires --resolution)', action='store_true')

args = parser.parse_args()

# -------------------- Setup Parameters -------------------- #

model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check model file
if not os.path.exists(model_path):
    print('ERROR: Model file not found:', model_path)
    sys.exit(1)

# Load model
model = YOLO(model_path, task='detect')
labels = model.names

# Supported extensions
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

# -------------------- Parse Source Type -------------------- #

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source.lower())
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(1)
elif img_source.isdigit():
    source_type = 'usb'
    usb_idx = int(img_source)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input "{img_source}" is invalid. Please try again.')
    sys.exit(1)

# -------------------- Resolution Parsing -------------------- #

resize = False
if user_res:
    try:
        resW, resH = map(int, user_res.lower().split('x'))
        resize = True
    except:
        print("Invalid resolution format. Use 'widthxheight' (e.g., 640x480).")
        sys.exit(1)

# -------------------- Recorder Setup -------------------- #

if record:
    if source_type not in ['video', 'usb']:
        print('Recording only supported for video or webcam sources.')
        sys.exit(1)
    if not user_res:
        print('You must specify --resolution to enable recording.')
        sys.exit(1)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# -------------------- Input Source Setup -------------------- #

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(img_source, '*')) if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type == 'video':
    cap = cv2.VideoCapture(img_source)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'usb':
    cap = cv2.VideoCapture(usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# -------------------- Text-to-Speech Setup -------------------- #

engine = pyttsx3.init()
voices = engine.getProperty('voices')
# Try to set a female voice
female_voice = None
for voice in voices:
    if 'female' in voice.name.lower() or 'zira' in voice.name.lower() or 'hazel' in voice.name.lower():
        female_voice = voice
        break
if female_voice:
    engine.setProperty('voice', female_voice.id)
else:
    # Fallback to second voice if available (often female on some systems)
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)

last_spoken = 0

def speak_label(label):
    engine.say(label)
    engine.runAndWait()
    time.sleep(2)
    engine.say(label)
    engine.runAndWait()

# -------------------- Visualization Settings -------------------- #

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 100
img_count = 0

# -------------------- Inference Loop -------------------- #

while True:
    t_start = time.perf_counter()

    # Read frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('End of video or cannot read from webcam.')
            break

    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Failed to capture from Picamera.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        classidx = int(detections[i].cls.item())
        conf = detections[i].conf.item()
        classname = labels[classidx]

        if conf >= min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            xmin, ymin, xmax, ymax = xyxy
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            object_count += 1

    # Speak label if object detected (throttled to avoid spamming)
    if len(detections) > 0 and source_type == 'usb':  # Only for webcam as per user request
        first_detection = detections[0]
        classidx = int(first_detection.cls.item())
        classname = labels[classidx]
        conf = first_detection.conf.item()
        if conf >= min_thresh:
            current_time = time.time()
            if current_time - last_spoken > 1:  # Throttle speaking to every 4 seconds
                threading.Thread(target=speak_label, args=(classname,)).start()
                last_spoken = current_time

    # Show FPS and object count
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow('YOLOv11 Detection', frame)

    if record:
        recorder.write(frame)

    key = cv2.waitKey(5 if source_type in ['video', 'usb', 'picamera'] else 0)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)

    # FPS calculation
    t_end = time.perf_counter()
    fps = 1 / (t_end - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(fps)
    avg_frame_rate = np.mean(frame_rate_buffer)

# -------------------- Cleanup -------------------- #

print(f'Average FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
'''
import os
import sys
import argparse
import glob
import time
import threading

import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3

# -------------------- Argument Parsing -------------------- #

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source: image file, folder, video file, webcam index (e.g., "0"), "usb0", or "picamera0"', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (default=0.5)',
                    default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH format (e.g., "640x480") to resize input and output',
                    default=None)
parser.add_argument('--record', help='Record video/webcam output as "demo1.avi" (requires --resolution)', action='store_true')

args = parser.parse_args()

# -------------------- Setup Parameters -------------------- #

model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check model file
if not os.path.exists(model_path):
    print('ERROR: Model file not found:', model_path)
    sys.exit(1)

# Load model
model = YOLO(model_path, task='detect')
labels = model.names

# Supported extensions
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

# -------------------- Parse Source Type -------------------- #

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source.lower())
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(1)
elif img_source.isdigit():
    source_type = 'usb'
    usb_idx = int(img_source)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input "{img_source}" is invalid. Please try again.')
    sys.exit(1)

# -------------------- Resolution Parsing -------------------- #

resize = False
if user_res:
    try:
        resW, resH = map(int, user_res.lower().split('x'))
        resize = True
    except:
        print("Invalid resolution format. Use 'widthxheight' (e.g., 640x480).")
        sys.exit(1)

# -------------------- Recorder Setup -------------------- #

if record:
    if source_type not in ['video', 'usb']:
        print('Recording only supported for video or webcam sources.')
        sys.exit(1)
    if not user_res:
        print('You must specify --resolution to enable recording.')
        sys.exit(1)
    # Canvas size will be determined later, after frame size is known
    record_name = 'demo1.avi'
    record_fps = 30

# -------------------- Input Source Setup -------------------- #

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(img_source, '*')) if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type == 'video':
    cap = cv2.VideoCapture(img_source)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'usb':
    cap = cv2.VideoCapture(usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# -------------------- Text-to-Speech Setup -------------------- #

engine = pyttsx3.init()
voices = engine.getProperty('voices')
# Try to set a female voice
female_voice = None
for voice in voices:
    if 'female' in voice.name.lower() or 'zira' in voice.name.lower() or 'hazel' in voice.name.lower():
        female_voice = voice
        break
if female_voice:
    engine.setProperty('voice', female_voice.id)
else:
    # Fallback to second voice if available (often female on some systems)
    if len(voices) > 5:
        engine.setProperty('voice', voices[1].id)

last_spoken = 0

def speak_label(label):
    engine.say(label)
    engine.runAndWait()
    time.sleep(1)
    engine.say(label)
    engine.runAndWait()

# -------------------- Visualization Settings -------------------- #

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 100
img_count = 0

# -------------------- Inference Loop -------------------- #

while True:
    t_start = time.perf_counter()

    # Read frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('End of video or cannot read from webcam.')
            break

    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Failed to capture from Picamera.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        classidx = int(detections[i].cls.item())
        conf = detections[i].conf.item()
        classname = labels[classidx]

        if conf >= min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            xmin, ymin, xmax, ymax = xyxy
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            object_count += 1

    # Speak label if object detected (throttled to avoid spamming)
    if len(detections) > 0 and source_type == 'usb':  # Only for webcam as per user request
        first_detection = detections[0]
        classidx = int(first_detection.cls.item())
        classname = labels[classidx]
        conf = first_detection.conf.item()
        if conf >= min_thresh:
            current_time = time.time()
            if current_time - last_spoken > 1:  # Throttle speaking to every 4 seconds
                threading.Thread(target=speak_label, args=(classname,)).start()
                last_spoken = current_time

    # Create canvas with gradient background and center the frame
    frame_h, frame_w = frame.shape[:2]
    canvas_margin = 200  # Margin around the frame
    canvas_w = frame_w + canvas_margin
    canvas_h = frame_h + canvas_margin
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Create a vertical gradient from black to white
    for i in range(canvas_h):
        color_value = int(255 * i / canvas_h)
        canvas[i, :] = [color_value, color_value, color_value]
    
    # Center the frame on the canvas
    x_offset = (canvas_w - frame_w) // 2
    y_offset = (canvas_h - frame_h) // 2
    canvas[y_offset:y_offset+frame_h, x_offset:x_offset+frame_w] = frame

    # Show FPS and object count on the canvas
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(canvas, f'FPS: {avg_frame_rate:0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(canvas, f'Objects: {object_count}', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow('YOLOv11 Detection', canvas)

    # Initialize recorder if recording and first frame
    if record and 'recorder' not in locals():
        recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (canvas_w, canvas_h))

    if record:
        recorder.write(canvas)

    key = cv2.waitKey(5 if source_type in ['video', 'usb', 'picamera'] else 0)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', canvas)

    # FPS calculation
    t_end = time.perf_counter()
    fps = 1 / (t_end - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(fps)
    avg_frame_rate = np.mean(frame_rate_buffer)

# -------------------- Cleanup -------------------- #

print(f'Average FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()