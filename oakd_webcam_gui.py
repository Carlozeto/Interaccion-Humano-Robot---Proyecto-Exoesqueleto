import mediapipe as mp
import sys
import cv2
import depthai as dai
import numpy as np
import threading
import time
import csv
import json
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox, QFileDialog, QTextEdit)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
import blobconverter
from pose import getKeypoints, getValidPairs, getPersonwiseKeypoints
import os
import math
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
# --- Flask server for remote control ---
from flask import Flask, render_template_string, request, redirect

# --- Back Angle Calculation Functions ---
def calculate_angle(p1, p2, p3):
    """Calculate angle between three points (p2 is the vertex)"""
    if p1 is None or p2 is None or p3 is None:
        return None
    
    # Convert to numpy arrays if they aren't already
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Calculate vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate angle
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)

def interpolate_spine_points(landmarks, num_points=10):
    """Interpolate spine points between shoulders, neck, and hips"""
    if landmarks is None:
        return None
    
    # MediaPipe pose landmarks indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    NOSE = 0
    
    try:
        # Get key points
        left_shoulder = np.array([landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y, landmarks[LEFT_SHOULDER].z])
        right_shoulder = np.array([landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y, landmarks[RIGHT_SHOULDER].z])
        left_hip = np.array([landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y, landmarks[LEFT_HIP].z])
        right_hip = np.array([landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y, landmarks[RIGHT_HIP].z])
        nose = np.array([landmarks[NOSE].x, landmarks[NOSE].y, landmarks[NOSE].z])
        
        # Calculate midpoints
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2
        
        # Create control points for spine curve
        control_points = np.array([nose, shoulder_mid, hip_mid])
        
        # Interpolate using spline
        tck, u = splprep([control_points[:, 0], control_points[:, 1], control_points[:, 2]], s=0, k=2)
        new_points = splev(np.linspace(0, 1, num_points), tck)
        
        spine_points = np.column_stack(new_points)
        return spine_points
        
    except Exception as e:
        print(f"Error interpolating spine points: {e}")
        return None

def calculate_back_angles(spine_points):
    """Calculate upper and lower back angles from interpolated spine points"""
    if spine_points is None or len(spine_points) < 5:
        return None, None
    
    try:
        # Upper back angle (around T12 level - approximately 1/3 down the spine)
        upper_idx = len(spine_points) // 3
        if upper_idx > 0 and upper_idx < len(spine_points) - 1:
            upper_angle = calculate_angle(
                spine_points[upper_idx - 1],
                spine_points[upper_idx],
                spine_points[upper_idx + 1]
            )
        else:
            upper_angle = None
        
        # Lower back angle (around L4 level - approximately 2/3 down the spine)
        lower_idx = 2 * len(spine_points) // 3
        if lower_idx > 0 and lower_idx < len(spine_points) - 1:
            lower_angle = calculate_angle(
                spine_points[lower_idx - 1],
                spine_points[lower_idx],
                spine_points[lower_idx + 1]
            )
        else:
            lower_angle = None
        
        return upper_angle, lower_angle
        
    except Exception as e:
        print(f"Error calculating back angles: {e}")
        return None, None

def calculate_3d_angle(p1, p2, p3):
    """Calculate 3D angle between three points"""
    if p1 is None or p2 is None or p3 is None:
        return None
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)

# --- Video Capture Threads ---
class OakdCapture(threading.Thread):
    def __init__(self):
        super().__init__()
        self.rgb_frame = None
        self.depth_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.pipeline = dai.Pipeline()
        # Camera nodes
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewSize(456, 256)
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        # Depth
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        stereo.depth.link(xoutDepth.input)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(True)
        stereo.setSubpixel(True)
        stereo.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setExtendedDisparity(False)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        # Output nodes
        xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        camRgb.preview.link(xoutRgb.input)

    def run(self):
        try:
            with dai.Device(self.pipeline) as device:
                qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
                last_frame_time = time.time()
                target_fps = 30
                frame_interval = 1.0 / target_fps
                
                while self.running:
                    current_time = time.time()
                    if current_time - last_frame_time >= frame_interval:
                        inRgb = qRgb.tryGet()
                        inDepth = qDepth.tryGet()
                        if inRgb is not None:
                            frame = inRgb.getCvFrame()
                            with self.lock:
                                self.rgb_frame = frame.copy()
                        if inDepth is not None:
                            depth_frame = inDepth.getFrame()
                            # Normalize and colorize for display
                            depth_disp = (depth_frame * (255/10000)).astype(np.uint8)
                            depth_disp = cv2.applyColorMap(depth_disp, cv2.COLORMAP_JET)
                            with self.lock:
                                self.depth_frame = depth_disp.copy()
                        last_frame_time = current_time
                    else:
                        time.sleep(0.001)  # Small sleep to prevent busy waiting
        except Exception as e:
            print(f"OAK-D error: {e}")

    def get_frames(self):
        with self.lock:
            return self.rgb_frame, self.depth_frame

    def stop(self):
        self.running = False

class WebcamCapture(threading.Thread):
    def __init__(self, cam_index=2):
        super().__init__()
        self.cap = cv2.VideoCapture(cam_index)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def run(self):
        last_frame_time = time.time()
        target_fps = 30
        frame_interval = 1.0 / target_fps
        
        while self.running and self.cap.isOpened():
            current_time = time.time()
            if current_time - last_frame_time >= frame_interval:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame.copy()
                last_frame_time = current_time
            else:
                time.sleep(0.001)  # Small sleep to prevent busy waiting

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

# --- OAK-D Pose Inference Thread ---
class OakdPoseThread(threading.Thread):
    def __init__(self, pipeline, pose_input_size):
        super().__init__()
        self.pipeline = pipeline
        self.pose_input_size = pose_input_size
        self.nn_frame = None
        self.depth_frame = None
        self.rgb_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.device = None
        self.qRgb = None
        self.qNn = None
        self.qDepth = None

    def run(self):
        try:
            with dai.Device(self.pipeline) as device:
                self.device = device
                self.qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                self.qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
                self.qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
                last_frame_time = time.time()
                target_fps = 30
                frame_interval = 1.0 / target_fps
                
                while self.running:
                    current_time = time.time()
                    if current_time - last_frame_time >= frame_interval:
                        inRgb = self.qRgb.tryGet()
                        inNn = self.qNn.tryGet()
                        inDepth = self.qDepth.tryGet()
                        with self.lock:
                            if inRgb is not None:
                                self.rgb_frame = inRgb.getCvFrame()
                            if inNn is not None:
                                self.nn_frame = inNn
                            if inDepth is not None:
                                self.depth_frame = inDepth.getFrame()
                        last_frame_time = current_time
                    else:
                        time.sleep(0.001)  # Small sleep to prevent busy waiting
        except Exception as e:
            print(f"OAK-D pose thread error: {e}")

    def get_latest(self):
        with self.lock:
            return self.rgb_frame, self.nn_frame, self.depth_frame

    def stop(self):
        self.running = False

# --- Main GUI ---
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OAKD + Webcam Pose Dataset Recorder")
        # OAK-D pose pipeline setup
        self.POSE_MODEL = "human-pose-estimation-0001"
        self.POSE_INPUT_SIZE = (456, 256)
        self.POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]
        self.COLORS = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
        blob_path = blobconverter.from_zoo(name=self.POSE_MODEL, shaves=6)
        self.pipeline = dai.Pipeline()
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewSize(*self.POSE_INPUT_SIZE)
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        stereo.depth.link(xoutDepth.input)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(True)
        stereo.setSubpixel(True)
        stereo.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setExtendedDisparity(False)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        nn = self.pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(blob_path)
        nn.input.setBlocking(False)
        camRgb.preview.link(nn.input)
        xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        camRgb.preview.link(xoutRgb.input)
        xoutNn = self.pipeline.create(dai.node.XLinkOut)
        xoutNn.setStreamName("nn")
        nn.out.link(xoutNn.input)
        # OAK-D pose thread
        self.oakd_pose_thread = OakdPoseThread(self.pipeline, self.POSE_INPUT_SIZE)
        self.oakd_pose_thread.start()
        # Webcam thread
        self.webcam = WebcamCapture()
        self.webcam.start()
        # Video display labels
        self.rgb_label = QLabel("OAK-D RGB")
        self.depth_label = QLabel("OAK-D Depth")
        self.webcam_label = QLabel("Webcam")
        self.mask_label = QLabel("Máscara Persona")
        for lbl in [self.rgb_label, self.depth_label, self.webcam_label, self.mask_label]:
            lbl.setFixedSize(304, 192)
            lbl.setAlignment(Qt.AlignCenter)
        # Patient info
        self.patient_id_edit = QLineEdit()
        self.patient_id_edit.setPlaceholderText("Patient ID")
        self.posture_combo = QComboBox()
        self.posture_combo.addItems(["None", "Good", "Bad"])
        # Record button
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self.toggle_recording)
        # Add 3D reconstruction button
        self.reconstruct_btn = QPushButton("Create 3D Model")
        self.reconstruct_btn.clicked.connect(self.create_3d_model)
        
        # Back angles display
        self.angles_label = QLabel("Ángulos Articulares de la Espalda")
        self.angles_label.setStyleSheet('font-size: 16px; font-weight: bold; color: #333;')
        self.angles_label.setAlignment(Qt.AlignCenter)
        
        # Create text area for angles display
        self.angles_text = QTextEdit()
        self.angles_text.setMaximumHeight(120)
        self.angles_text.setReadOnly(True)
        self.angles_text.setStyleSheet('''
            QTextEdit {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 5px;
                font-family: "Courier New", monospace;
                font-size: 12px;
                padding: 5px;
            }
        ''')
        
        # Layout
        hbox = QHBoxLayout()
        hbox.addWidget(self.rgb_label)
        hbox.addWidget(self.depth_label)
        hbox.addWidget(self.webcam_label)
        hbox.addWidget(self.mask_label)
        form_hbox = QHBoxLayout()
        form_hbox.addWidget(QLabel("Patient ID:"))
        form_hbox.addWidget(self.patient_id_edit)
        form_hbox.addWidget(QLabel("Posture:"))
        form_hbox.addWidget(self.posture_combo)
        form_hbox.addWidget(self.record_btn)
        form_hbox.addWidget(self.reconstruct_btn)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addLayout(form_hbox)
        
        # Add angles section
        vbox.addWidget(self.angles_label)
        vbox.addWidget(self.angles_text)
        self.oakd_warning_label = QLabel()
        self.oakd_warning_label.setStyleSheet('color: red; font-size: 16px; font-weight: bold;')
        self.oakd_warning_label.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.oakd_warning_label)
        self.model_warning_label = QLabel()
        self.model_warning_label.setStyleSheet('color: orange; font-size: 14px; font-weight: bold;')
        self.model_warning_label.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.model_warning_label)
        
        # Frame rate info
        self.fps_info_label = QLabel("Frame Rate: 30 FPS (Optimizado para estabilidad)")
        self.fps_info_label.setStyleSheet('color: blue; font-size: 12px; font-weight: bold;')
        self.fps_info_label.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.fps_info_label)
        self.setLayout(vbox)
        # Timer for updating frames - limit to 30fps (33.33ms interval)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(33)  # 1000ms / 30fps = 33.33ms
        # Recording
        self.recording = False
        self.video_writer = None
        self.recorded_frames = []
        self.record_start_time = None
        self.output_filename = None
        # MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mediapipe_pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # Initialize data storage for 3D reconstruction
        self.depth_frames = []
        self.landmarks_3d = []
        self.recording_path = None
        # Retrieve OAK-D camera intrinsics
        self.fx, self.fy, self.cx, self.cy = self.get_oakd_intrinsics()
        print(f"OAK-D intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        # 3D reconstruction parameters
        self.downsample_factor = 4  # sample every nth pixel
        self.min_depth = 100  # minimum depth in mm
        self.max_depth = 5000  # maximum depth in mm
        self.remote_recording_lock = threading.Lock()
        self.remote_recording_request = None  # 'start', 'stop', or None
        self.flask_thread = threading.Thread(target=self.run_flask_server, daemon=True)
        self.flask_thread.start()
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.last_oakd_frame_time = time.time()
        self.last_ui_update = time.time()  # Initialize UI frame rate control
        
        # Back angles calculation variables
        self.current_upper_angle = None
        self.current_lower_angle = None
        self.angle_history = []  # Store angle history for analysis
        self.max_history_size = 30  # Keep last 30 frames

    def get_oakd_intrinsics(self):
        try:
            with dai.Device() as device:
                calib = device.readCalibration()
                # Use the same size as your OAK-D RGB preview (456x256)
                intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 456, 256)
                fx = intrinsics[0][0]
                fy = intrinsics[1][1]
                cx = intrinsics[0][2]
                cy = intrinsics[1][2]
                return fx, fy, cx, cy
        except Exception as e:
            print(f"Error retrieving OAK-D intrinsics: {e}")
            # Fallback to previous defaults
            return 400, 400, 320, 240

    def update_frames(self):
        # Frame rate control for UI updates
        current_time = time.time()
        if hasattr(self, 'last_ui_update'):
            if current_time - self.last_ui_update < 0.033:  # 30fps limit
                return
        self.last_ui_update = current_time
        
        # OAK-D pose
        rgb, nn_data, depth = self.oakd_pose_thread.get_latest()
        rgb_disp = None
        mask_disp = None
        if rgb is not None:
            self.last_oakd_frame_time = time.time()
            self.oakd_warning_label.setText("")
            rgb_disp = rgb.copy()
            # Draw pose if NN data is available
            if nn_data is not None and depth is not None:
                try:
                    heatmaps = np.array(nn_data.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
                    pafs = np.array(nn_data.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
                    heatmaps = heatmaps.astype('float32')
                    pafs = pafs.astype('float32')
                    outputs = np.concatenate((heatmaps, pafs), axis=1)
                    new_keypoints = []
                    new_keypoints_list = np.zeros((0, 3))
                    keypoint_id = 0
                    for row in range(18):
                        probMap = outputs[0, row, :, :]
                        probMap = cv2.resize(probMap, self.POSE_INPUT_SIZE)
                        keypoints = getKeypoints(probMap, 0.3)
                        new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
                        keypoints_with_id = []
                        for i in range(len(keypoints)):
                            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                            keypoint_id += 1
                        new_keypoints.append(keypoints_with_id)
                    valid_pairs, invalid_pairs = getValidPairs(outputs, w=self.POSE_INPUT_SIZE[0], h=self.POSE_INPUT_SIZE[1], detected_keypoints=new_keypoints)
                    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)
                    # Draw skeleton (no posture text)
                    scale_factor = rgb_disp.shape[0] / self.POSE_INPUT_SIZE[1]
                    offset_w = int(rgb_disp.shape[1] - self.POSE_INPUT_SIZE[0] * scale_factor) // 2
                    def scale(point):
                        return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)
                    for i in range(18):
                        for j in range(len(new_keypoints[i])):
                            cv2.circle(rgb_disp, scale(new_keypoints[i][j][0:2]), 5, self.COLORS[i], -1, cv2.LINE_AA)
                    for i in range(17):
                        for n in range(len(personwiseKeypoints)):
                            index = personwiseKeypoints[n][np.array(self.POSE_PAIRS[i])]
                            if -1 in index:
                                continue
                            B = np.int32(new_keypoints_list[index.astype(int), 0])
                            A = np.int32(new_keypoints_list[index.astype(int), 1])
                            cv2.line(rgb_disp, scale((B[0], A[0])), scale((B[1], A[1])), self.COLORS[i], 3, cv2.LINE_AA)
                except Exception as e:
                    print(f"OAK-D pose draw error: {e}")
            rgb_disp = cv2.resize(rgb_disp, (304, 192))
            self.rgb_label.setPixmap(self.cv2qt(rgb_disp))
            # --- MediaPipe Selfie Segmentation mask visualization ---
            rgb_for_mp = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            results = self.selfie_segmenter.process(rgb_for_mp)
            if hasattr(results, 'segmentation_mask') and results.segmentation_mask is not None:
                mask = results.segmentation_mask
                mask_img = (mask * 255).astype(np.uint8)
                mask_img = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
                mask_disp = cv2.resize(mask_img, (304, 192))
                self.mask_label.setPixmap(self.cv2qt(mask_disp))
            else:
                self.mask_label.setText("Sin máscara")
        else:
            self.rgb_label.setText("OAK-D RGB")
            self.mask_label.setText("Sin máscara")
        # OAK-D depth
        if depth is not None:
            depth_disp = (depth * (255/10000)).astype(np.uint8)
            depth_disp = cv2.applyColorMap(depth_disp, cv2.COLORMAP_JET)
            depth_disp = cv2.resize(depth_disp, (304, 192))
            self.depth_label.setPixmap(self.cv2qt(depth_disp))
        else:
            self.depth_label.setText("OAK-D Depth")
        # Webcam with MediaPipe pose
        webcam = self.webcam.get_frame()
        webcam_disp = None
        webcam_results = None
        if webcam is not None:
            webcam_disp = webcam.copy()
            try:
                img_rgb = cv2.cvtColor(webcam_disp, cv2.COLOR_BGR2RGB)
                webcam_results = self.mediapipe_pose.process(img_rgb)
                if webcam_results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        webcam_disp,
                        webcam_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Calculate back angles from MediaPipe landmarks
                    self.calculate_and_display_back_angles(webcam_results.pose_landmarks)
                    
            except Exception as e:
                print(f"MediaPipe pose draw error: {e}")
            webcam_disp = cv2.resize(webcam_disp, (304, 192))
            self.webcam_label.setPixmap(self.cv2qt(webcam_disp))
        else:
            self.webcam_label.setText("Webcam")
            self.update_angles_display("No hay datos de pose disponibles")
        # Recording
        if self.recording and rgb_disp is not None and depth is not None and webcam_disp is not None:
            rgb_r = rgb_disp
            depth_r = (depth * (255/10000)).astype(np.uint8)
            depth_r = cv2.applyColorMap(depth_r, cv2.COLORMAP_JET)
            depth_r = cv2.resize(depth_r, (304, 192))
            webcam_r = webcam_disp  # Use the frame with pose drawing
            combined = np.hstack([rgb_r, depth_r, webcam_r])
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.output_filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                self.video_writer = cv2.VideoWriter(self.output_filename, fourcc, 20.0, (combined.shape[1], combined.shape[0]))
                self.record_start_time = time.time()
            self.video_writer.write(combined)
            # Store depth frame
            self.depth_frames.append(depth.copy())
            
            # Store MediaPipe landmarks if available
            if webcam_results and webcam_results.pose_landmarks:
                landmarks = []
                for landmark in webcam_results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                self.landmarks_3d.append(landmarks)
        # Check for remote recording requests
        with self.remote_recording_lock:
            if self.remote_recording_request == 'start' and not self.recording:
                self.toggle_recording()
                self.remote_recording_request = None
            elif self.remote_recording_request == 'stop' and self.recording:
                self.toggle_recording()
                self.remote_recording_request = None
        # Check for OAK-D freeze
        if time.time() - self.last_oakd_frame_time > 2.0:
            self.oakd_warning_label.setText("¡Advertencia: La cámara OAK-D está congelada o desconectada!")
        else:
            self.oakd_warning_label.setText("")

    def calculate_and_display_back_angles(self, landmarks):
        """Calculate back angles from MediaPipe landmarks and display them"""
        try:
            # Interpolate spine points
            spine_points = interpolate_spine_points(landmarks.landmark, num_points=15)
            
            if spine_points is not None:
                # Calculate upper and lower back angles
                upper_angle, lower_angle = calculate_back_angles(spine_points)
                
                # Update current angles
                self.current_upper_angle = upper_angle
                self.current_lower_angle = lower_angle
                
                # Add to history
                angle_data = {
                    'timestamp': time.time(),
                    'upper_angle': upper_angle,
                    'lower_angle': lower_angle
                }
                self.angle_history.append(angle_data)
                
                # Keep only recent history
                if len(self.angle_history) > self.max_history_size:
                    self.angle_history.pop(0)
                
                # Display angles
                self.update_angles_display(upper_angle, lower_angle)
            else:
                self.update_angles_display("No se pudieron calcular los ángulos")
                
        except Exception as e:
            print(f"Error calculating back angles: {e}")
            self.update_angles_display("Error en el cálculo de ángulos")

    def update_angles_display(self, upper_angle=None, lower_angle=None):
        """Update the angles display text area"""
        if isinstance(upper_angle, str):
            # Error message
            display_text = f"Estado: {upper_angle}"
        else:
            # Calculate statistics from history
            if self.angle_history:
                recent_angles = self.angle_history[-10:]  # Last 10 frames
                upper_angles = [a['upper_angle'] for a in recent_angles if a['upper_angle'] is not None]
                lower_angles = [a['lower_angle'] for a in recent_angles if a['lower_angle'] is not None]
                
                upper_avg = np.mean(upper_angles) if upper_angles else None
                lower_avg = np.mean(lower_angles) if lower_angles else None
                
                display_text = f"""ÁNGULOS ARTICULARES DE LA ESPALDA
═══════════════════════════════════════════════════════════════

Ángulo Superior (T12): {upper_angle:.1f}° (Promedio: {upper_avg:.1f}°)
Ángulo Inferior (L4):  {lower_angle:.1f}° (Promedio: {lower_avg:.1f}°)

ANÁLISIS POSTURAL:
═══════════════════
"""
                
                # Postural analysis
                if upper_angle is not None:
                    if upper_angle < 150:
                        display_text += "• Espalda superior: POSTURA ENCORVADA\n"
                    elif upper_angle > 170:
                        display_text += "• Espalda superior: POSTURA RÍGIDA\n"
                    else:
                        display_text += "• Espalda superior: POSTURA NORMAL\n"
                
                if lower_angle is not None:
                    if lower_angle < 150:
                        display_text += "• Espalda inferior: POSTURA ENCORVADA\n"
                    elif lower_angle > 170:
                        display_text += "• Espalda inferior: POSTURA RÍGIDA\n"
                    else:
                        display_text += "• Espalda inferior: POSTURA NORMAL\n"
                
                display_text += f"\nMuestras analizadas: {len(self.angle_history)}"
            else:
                display_text = "Esperando datos de pose..."
        
        self.angles_text.setText(display_text)

    def cv2qt(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_img)

    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.record_btn.setText("Stop Recording")
            self.video_writer = None
            # Create directory for this recording
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.recording_path = f"recording_{timestamp}"
            os.makedirs(self.recording_path, exist_ok=True)
            # Clear previous data
            self.depth_frames = []
            self.landmarks_3d = []
        else:
            self.recording = False
            self.record_btn.setText("Start Recording")
            if self.video_writer is not None:
                self.video_writer.release()
                self.save_metadata()
                self.video_writer = None
                # Save depth data and landmarks
                self.save_3d_data()

    def save_metadata(self):
        patient_id = self.patient_id_edit.text()
        posture = self.posture_combo.currentText()
        filename = self.output_filename
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        csv_file = "recordings_metadata.csv"
        write_header = False
        try:
            with open(csv_file, 'r') as f:
                pass
        except FileNotFoundError:
            write_header = True
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["PatientID", "Posture", "Filename", "Timestamp"])
            writer.writerow([patient_id, posture, filename, timestamp])

    def save_3d_data(self):
        if not self.recording_path:
            return
            
        # Save depth frames
        depth_path = os.path.join(self.recording_path, "depth_data.npz")
        np.savez_compressed(depth_path, depth_frames=self.depth_frames)
        
        # Save landmarks
        landmarks_path = os.path.join(self.recording_path, "landmarks_3d.json")
        with open(landmarks_path, 'w') as f:
            json.dump(self.landmarks_3d, f)

    def create_3d_model(self):
        """
        Create a 3D point cloud of the detected person using both OAK-D and webcam data.
        Combines depth information from OAK-D with pose landmarks from webcam for better reconstruction.
        """
        # Get OAK-D data
        rgb, nn_data, depth = self.oakd_pose_thread.get_latest()
        webcam_frame = self.webcam.get_frame()
        
        if rgb is None or depth is None:
            print("No OAK-D frame available for 3D reconstruction.")
            self.model_warning_label.setText("No hay frame válido de OAK-D para el modelo 3D.")
            return
            
        if webcam_frame is None:
            print("No webcam frame available for 3D reconstruction.")
            self.model_warning_label.setText("No hay frame válido de webcam para el modelo 3D.")
            return
        
        # Resize depth to match RGB if needed
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        rows, cols = depth.shape[:2]
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        min_depth = self.min_depth
        max_depth = self.max_depth
        
        # Get MediaPipe pose from webcam
        webcam_rgb = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
        webcam_results = self.mediapipe_pose.process(webcam_rgb)
        
        # Run segmentation on the OAK-D RGB frame
        rgb_for_mp = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        segmentation_results = self.selfie_segmenter.process(rgb_for_mp)
        
        if not hasattr(segmentation_results, 'segmentation_mask') or segmentation_results.segmentation_mask is None:
            print("No person detected in OAK-D frame.")
            self.model_warning_label.setText("No se detectó persona en el frame de OAK-D.")
            return
            
        if not webcam_results.pose_landmarks:
            print("No pose landmarks detected in webcam frame.")
            self.model_warning_label.setText("No se detectaron landmarks de pose en webcam.")
            return
        
        mask = segmentation_results.segmentation_mask
        person_mask = (mask > 0.2).astype(np.uint8)
        
        # Create point cloud from OAK-D depth
        point_cloud = []
        colors = []
        
        for i in range(0, rows, self.downsample_factor):
            for j in range(0, cols, self.downsample_factor):
                if person_mask[i, j]:
                    z = depth[i, j]
                    if min_depth < z < max_depth:
                        x = (j - cx) * z / fx
                        y = (i - cy) * z / fy
                        point_cloud.append([x, y, z])
                        b, g, r = rgb[i, j]
                        colors.append([r, g, b])
        
        # Add pose landmarks as additional points
        if webcam_results.pose_landmarks:
            landmarks_3d = []
            for landmark in webcam_results.pose_landmarks.landmark:
                if landmark.visibility > 0.5:  # Only use visible landmarks
                    # Convert normalized coordinates to 3D space
                    # Assuming webcam and OAK-D are roughly aligned
                    x = (landmark.x - 0.5) * 1000  # Scale to reasonable 3D space
                    y = (landmark.y - 0.5) * 1000
                    z = landmark.z * 1000
                    landmarks_3d.append([x, y, z])
            
            # Add landmarks to point cloud with special color
            for landmark in landmarks_3d:
                point_cloud.append(landmark)
                colors.append([255, 0, 0])  # Red color for landmarks
        
        print(f"Puntos detectados para la persona: {len(point_cloud)}")
        if len(point_cloud) < 100:
            print("Advertencia: Muy pocos puntos detectados. No se guardará el modelo 3D.")
            self.model_warning_label.setText("Advertencia: Muy pocos puntos detectados. No se guardó el modelo 3D.")
            return
            
        self.model_warning_label.setText("")
        point_cloud = np.array(point_cloud)
        colors = np.array(colors)
        
        # Create enhanced filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"enhanced_3d_model_{timestamp}.ply"
        
        # Save the enhanced point cloud
        self.save_point_cloud_color(point_cloud, colors, output_path)
        
        # Also save spine curve if available
        if webcam_results.pose_landmarks:
            spine_points = interpolate_spine_points(webcam_results.pose_landmarks.landmark, num_points=20)
            if spine_points is not None:
                spine_path = f"spine_curve_{timestamp}.ply"
                self.save_spine_curve(spine_points, spine_path)
                print(f"Spine curve saved to {spine_path}")
        
        print(f"Enhanced 3D model saved to {output_path}")
        self.model_warning_label.setText(f"Modelo 3D mejorado guardado: {output_path}")

    def save_spine_curve(self, spine_points, filename):
        """Save spine curve to PLY file format."""
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(spine_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for point in spine_points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

    def save_point_cloud_color(self, points, colors, filename):
        """Save point cloud to PLY file format with RGB colors (0-255)."""
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for point, color in zip(points, colors):
                f.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")

    def run_flask_server(self):
        app = Flask(__name__)
        mainwindow = self
        
        HTML = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Control de Grabación</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial; background: #222; color: #fff; text-align: center; }
                button { font-size: 2em; padding: 1em 2em; margin-top: 2em; border-radius: 10px; border: none; }
                .start { background: #28a745; color: #fff; }
                .stop { background: #dc3545; color: #fff; }
                .model { background: #007bff; color: #fff; margin-left: 1em; }
            </style>
        </head>
        <body>
            <h1>Control de Grabación</h1>
            <form method="POST">
                {% if recording %}
                <button class="stop" name="action" value="stop">Detener Grabación</button>
                {% else %}
                <button class="start" name="action" value="start">Iniciar Grabación</button>
                {% endif %}
                <button class="model" name="action" value="model">Generar Modelo 3D</button>
            </form>
            <p>Estado actual: <b>{{ 'Grabando' if recording else 'Detenido' }}</b></p>
            {% if message %}<p style="color: #0f0;">{{ message }}</p>{% endif %}
        </body>
        </html>
        '''

        @app.route('/', methods=['GET', 'POST'])
        def index():
            message = None
            if request.method == 'POST':
                action = request.form.get('action')
                if action == 'model':
                    try:
                        mainwindow.create_3d_model()
                        message = "Modelo 3D generado exitosamente."
                    except Exception as e:
                        message = f"Error al generar el modelo 3D: {e}"
                else:
                    with mainwindow.remote_recording_lock:
                        mainwindow.remote_recording_request = action
                    return redirect('/')
            return render_template_string(HTML, recording=mainwindow.recording, message=message)

        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

    def closeEvent(self, event):
        self.oakd_pose_thread.stop()
        self.webcam.stop()
        self.oakd_pose_thread.join()
        self.webcam.join()
        if self.video_writer is not None:
            self.video_writer.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 