import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from typing import List, Dict, Any
import json
import os

class HumanPoseDetector:
    def __init__(self):
        """Initialize the human pose detector using MediaPipe"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define pose connections for skeleton tracking
        self.POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Face and shoulders
            (0, 4), (4, 5), (5, 6), (6, 8),  # Face and shoulders
            (9, 10),  # Mouth
            (11, 12),  # Shoulders
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24),  # Torso
            (23, 24),  # Hips
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28)   # Right leg
        ]

    def detect_poses(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect human poses in a single frame
        Returns: Dictionary containing pose landmarks and analysis
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        pose_data = {
            "landmarks": [],
            "bounding_box": None,
            "skeleton_data": [],
            "movement_analysis": {}
        }
        
        if results.pose_landmarks:
            h, w, _ = frame.shape
            landmarks = []
            
            # Extract all landmarks
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks.append({
                    "id": idx,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })
            
            pose_data["landmarks"] = landmarks
            
            # Calculate bounding box
            x_coords = [lm["x"] for lm in landmarks if lm["visibility"] > 0.5]
            y_coords = [lm["y"] for lm in landmarks if lm["visibility"] > 0.5]
            
            if x_coords and y_coords:
                pose_data["bounding_box"] = {
                    "x_min": min(x_coords),
                    "y_min": min(y_coords),
                    "x_max": max(x_coords),
                    "y_max": max(y_coords),
                    "width": max(x_coords) - min(x_coords),
                    "height": max(y_coords) - min(y_coords)
                }
            
            # Extract skeleton connections
            skeleton = []
            for connection in self.POSE_CONNECTIONS:
                if (connection[0] < len(landmarks) and connection[1] < len(landmarks) and
                    landmarks[connection[0]]["visibility"] > 0.5 and 
                    landmarks[connection[1]]["visibility"] > 0.5):
                    
                    skeleton.append({
                        "start_point": connection[0],
                        "end_point": connection[1],
                        "start_x": landmarks[connection[0]]["x"],
                        "start_y": landmarks[connection[0]]["y"],
                        "end_x": landmarks[connection[1]]["x"],
                        "end_y": landmarks[connection[1]]["y"]
                    })
            
            pose_data["skeleton_data"] = skeleton
            
            # Basic movement analysis
            pose_data["movement_analysis"] = self._analyze_movement(landmarks)
        
        return pose_data

    def _analyze_movement(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """Analyze movement patterns from landmarks"""
        analysis = {
            "pose_type": "unknown",
            "body_parts_movement": {},
            "symmetry_score": 0.0,
            "activity_level": "low"
        }
        
        if len(landmarks) < 25:
            return analysis
        
        # Calculate symmetry between left and right sides
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        shoulder_symmetry = 1 - abs(left_shoulder["y"] - right_shoulder["y"])
        hip_symmetry = 1 - abs(left_hip["y"] - right_hip["y"])
        analysis["symmetry_score"] = (shoulder_symmetry + hip_symmetry) / 2
        
        # Detect basic pose types
        analysis.update(self._detect_pose_type(landmarks))
        
        return analysis

    def _detect_pose_type(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """Detect basic human pose types"""
        # Key landmarks indices
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        
        pose_info = {
            "pose_type": "standing",
            "confidence": 0.8
        }
        
        # Simple pose classification based on vertical positions
        try:
            shoulder_avg_y = (landmarks[LEFT_SHOULDER]["y"] + landmarks[RIGHT_SHOULDER]["y"]) / 2
            hip_avg_y = (landmarks[LEFT_HIP]["y"] + landmarks[RIGHT_HIP]["y"]) / 2
            knee_avg_y = (landmarks[LEFT_KNEE]["y"] + landmarks[RIGHT_KNEE]["y"]) / 2
            
            # Check if sitting (hips significantly lower than shoulders and knees bent)
            if (hip_avg_y - shoulder_avg_y > 0.2 and 
                knee_avg_y - hip_avg_y < 0.3):
                pose_info["pose_type"] = "sitting"
                pose_info["confidence"] = 0.7
            
            # Check if arms are raised
            left_wrist_y = landmarks[15]["y"]
            right_wrist_y = landmarks[16]["y"]
            
            if left_wrist_y < shoulder_avg_y or right_wrist_y < shoulder_avg_y:
                pose_info["pose_type"] = "arms_raised"
                pose_info["confidence"] = 0.6
                
        except Exception as e:
            print(f"Pose detection error: {e}")
        
        return pose_info

    def draw_pose_landmarks(self, frame: np.ndarray, pose_data: Dict[str, Any]) -> np.ndarray:
        """Draw pose landmarks and skeleton on the frame"""
        h, w, _ = frame.shape
        
        # Draw skeleton connections
        for connection in pose_data["skeleton_data"]:
            start_point = (
                int(connection["start_x"] * w),
                int(connection["start_y"] * h)
            )
            end_point = (
                int(connection["end_x"] * w),
                int(connection["end_y"] * h)
            )
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in pose_data["landmarks"]:
            if landmark["visibility"] > 0.5:
                center = (
                    int(landmark["x"] * w),
                    int(landmark["y"] * h)
                )
                cv2.circle(frame, center, 4, (255, 0, 0), -1)
        
        # Draw bounding box
        if pose_data["bounding_box"]:
            bbox = pose_data["bounding_box"]
            start_point = (int(bbox["x_min"] * w), int(bbox["y_min"] * h))
            end_point = (int(bbox["x_max"] * w), int(bbox["y_max"] * h))
            cv2.rectangle(frame, start_point, end_point, (0, 0, 255), 2)
            
            # Display pose type
            pose_type = pose_data["movement_analysis"].get("pose_type", "unknown")
            cv2.putText(frame, f"Pose: {pose_type}", 
                       (start_point[0], start_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame

class VideoPoseAnalyzer:
    def __init__(self):
        self.pose_detector = HumanPoseDetector()
    
    def analyze_video(self, video_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Analyze poses in a video file
        Returns comprehensive analysis data
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        analysis_data = {
            "video_info": {
                "fps": fps,
                "total_frames": total_frames,
                "duration": total_frames / fps if fps > 0 else 0,
                "resolution": f"{width}x{height}"
            },
            "frame_analyses": [],
            "summary": {
                "total_human_detections": 0,
                "pose_types": {},
                "movement_patterns": []
            }
        }
        
        frame_count = 0
        pose_types_counter = {}
        
        # Video writer for output if path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze current frame
            pose_data = self.pose_detector.detect_poses(frame)
            
            # Count human detections
            if pose_data["landmarks"]:
                analysis_data["summary"]["total_human_detections"] += 1
                
                # Count pose types
                pose_type = pose_data["movement_analysis"].get("pose_type", "unknown")
                pose_types_counter[pose_type] = pose_types_counter.get(pose_type, 0) + 1
            
            # Draw landmarks on frame if writing output
            if writer:
                annotated_frame = self.pose_detector.draw_pose_landmarks(frame.copy(), pose_data)
                writer.write(annotated_frame)
            
            # Store frame analysis (sample every 10 frames to avoid too much data)
            if frame_count % 10 == 0:
                frame_analysis = {
                    "frame_number": frame_count,
                    "timestamp": frame_count / fps,
                    "pose_detected": len(pose_data["landmarks"]) > 0,
                    "landmarks_count": len(pose_data["landmarks"]),
                    "movement_analysis": pose_data["movement_analysis"]
                }
                analysis_data["frame_analyses"].append(frame_analysis)
            
            frame_count += 1
        
        cap.release()
        if writer:
            writer.release()
        
        # Update summary with pose types distribution
        analysis_data["summary"]["pose_types"] = pose_types_counter
        
        return analysis_data

# Utility function to save analysis results
def save_analysis_results(analysis_data: Dict[str, Any], output_json_path: str):
    """Save analysis results to JSON file"""
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    analyzer = VideoPoseAnalyzer()
    
    # Analyze video
    input_video = "input_video.mp4"
    output_video = "output_analyzed.mp4"
    output_json = "analysis_results.json"
    
    try:
        results = analyzer.analyze_video(input_video, output_video)
        save_analysis_results(results, output_json)
        print("Analysis completed successfully!")
        print(f"Total human detections: {results['summary']['total_human_detections']}")
        print(f"Pose types: {results['summary']['pose_types']}")
    except Exception as e:
        print(f"Error during analysis: {e}")