from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = {
    0: "fall detected",
    1: "walk",
    2: "sit",
}
CLASS_DISPLAY_NAMES = {
    0: "Fall",
    1: "Walk",
    2: "Sit",
}
CLASS_COLORS = {
    0: (58, 78, 239),
    1: (92, 184, 92),
    2: (235, 176, 72),
}

COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}
SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


@dataclass(frozen=True)
class PoseRuleConfig:
    fall_angle_threshold: float = 55.0
    fall_ratio_threshold: float = 0.90
    fall_vertical_spread_threshold: float = 0.80
    sit_ratio_threshold: float = 0.45
    sit_knee_hip_threshold: float = 0.15
    min_keypoint_conf: float = 0.25
    min_visible_keypoints: int = 5


def load_pose_model(weights_path: Path | str) -> YOLO:
    return YOLO(str(weights_path))


def midpoint(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    names: tuple[str, str],
    min_conf: float,
) -> tuple[float, float] | None:
    points = []
    for name in names:
        idx = COCO_KEYPOINTS[name]
        if idx < len(confidences) and confidences[idx] >= min_conf:
            points.append(keypoints[idx])
    if not points:
        return None
    point = np.mean(np.asarray(points, dtype=float), axis=0)
    return float(point[0]), float(point[1])


def point(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    names: tuple[str, ...],
    min_conf: float,
) -> tuple[float, float] | None:
    visible = []
    for name in names:
        idx = COCO_KEYPOINTS[name]
        if idx < len(confidences) and confidences[idx] >= min_conf:
            visible.append(keypoints[idx])
    if not visible:
        return None
    item = np.mean(np.asarray(visible, dtype=float), axis=0)
    return float(item[0]), float(item[1])


def angle_from_horizontal(start: tuple[float, float], end: tuple[float, float]) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = abs(math.degrees(math.atan2(dy, dx)))
    return min(angle, 180.0 - angle)


def extract_pose_features(
    box: tuple[float, float, float, float],
    keypoints: np.ndarray,
    confidences: np.ndarray,
    config: PoseRuleConfig,
) -> dict[str, float]:
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    visible = confidences >= config.min_keypoint_conf
    visible_points = keypoints[visible]

    shoulder_mid = midpoint(keypoints, confidences, ("left_shoulder", "right_shoulder"), config.min_keypoint_conf)
    hip_mid = midpoint(keypoints, confidences, ("left_hip", "right_hip"), config.min_keypoint_conf)
    knee_mid = midpoint(keypoints, confidences, ("left_knee", "right_knee"), config.min_keypoint_conf)
    head = point(
        keypoints,
        confidences,
        ("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
        config.min_keypoint_conf,
    )

    torso_angle = 90.0
    if shoulder_mid is not None and hip_mid is not None:
        torso_angle = angle_from_horizontal(shoulder_mid, hip_mid)

    vertical_spread = 1.0
    horizontal_spread = 0.0
    if len(visible_points):
        vertical_spread = float((visible_points[:, 1].max() - visible_points[:, 1].min()) / height)
        horizontal_spread = float((visible_points[:, 0].max() - visible_points[:, 0].min()) / width)

    hip_knee_gap = 1.0
    if hip_mid is not None and knee_mid is not None:
        hip_knee_gap = abs(knee_mid[1] - hip_mid[1]) / height

    head_hip_gap = 1.0
    if head is not None and hip_mid is not None:
        head_hip_gap = abs(head[1] - hip_mid[1]) / height

    return {
        "box_width": width,
        "box_height": height,
        "box_ratio": width / height,
        "visible_keypoints": int(visible.sum()),
        "avg_keypoint_conf": float(confidences[visible].mean()) if visible.any() else 0.0,
        "torso_angle_from_horizontal": torso_angle,
        "vertical_spread": vertical_spread,
        "horizontal_spread": horizontal_spread,
        "hip_knee_gap": hip_knee_gap,
        "head_hip_gap": head_hip_gap,
    }


def classify_pose(features: dict[str, float], config: PoseRuleConfig) -> tuple[int, float, str]:
    if features["visible_keypoints"] < config.min_visible_keypoints:
        class_id = 0 if features["box_ratio"] >= config.fall_ratio_threshold else 1
        confidence = min(0.65, 0.35 + abs(features["box_ratio"] - 1.0) * 0.20)
        return class_id, confidence, "fallback_box_shape"

    horizontal_torso = features["torso_angle_from_horizontal"] <= config.fall_angle_threshold
    wide_body = features["box_ratio"] >= config.fall_ratio_threshold
    compact_vertical_pose = features["vertical_spread"] <= config.fall_vertical_spread_threshold
    fall_score = sum([horizontal_torso, wide_body, compact_vertical_pose])
    if fall_score >= 2:
        confidence = min(0.98, 0.52 + 0.12 * fall_score + 0.18 * features["avg_keypoint_conf"])
        return 0, confidence, "horizontal_or_wide_pose"

    likely_sit = features["box_ratio"] >= config.sit_ratio_threshold and (
        features["hip_knee_gap"] <= config.sit_knee_hip_threshold
        or features["head_hip_gap"] <= 0.40
    )
    if likely_sit:
        confidence = min(0.94, 0.48 + 0.28 * features["avg_keypoint_conf"])
        return 2, confidence, "compact_seated_pose"

    confidence = min(0.94, 0.50 + 0.30 * features["avg_keypoint_conf"])
    return 1, confidence, "upright_pose"


def predict_pose(
    model: YOLO,
    image: np.ndarray,
    conf: float,
    imgsz: int,
    config: PoseRuleConfig,
) -> list[dict[str, object]]:
    result = model.predict(image, conf=conf, imgsz=imgsz, verbose=False)[0]
    if result.boxes is None or result.keypoints is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy().astype(float)
    box_scores = result.boxes.conf.cpu().numpy().astype(float)
    keypoints_xy = result.keypoints.xy.cpu().numpy().astype(float)
    if result.keypoints.conf is None:
        keypoint_conf = np.ones(keypoints_xy.shape[:2], dtype=float)
    else:
        keypoint_conf = result.keypoints.conf.cpu().numpy().astype(float)

    predictions = []
    for idx, box in enumerate(boxes):
        features = extract_pose_features(tuple(box.tolist()), keypoints_xy[idx], keypoint_conf[idx], config)
        class_id, pose_confidence, reason = classify_pose(features, config)
        score = float(min(0.99, pose_confidence * box_scores[idx]))
        x1, y1, x2, y2 = box.tolist()
        predictions.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "score": score,
                "detector_score": float(box_scores[idx]),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "rule_reason": reason,
                **features,
                "keypoints": keypoints_xy[idx],
                "keypoint_conf": keypoint_conf[idx],
            }
        )
    return predictions


def serializable_prediction(prediction: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in prediction.items()
        if key not in {"keypoints", "keypoint_conf"}
    }


def draw_pose_prediction(frame: np.ndarray, prediction: dict[str, object]) -> None:
    class_id = int(prediction["class_id"])
    score = float(prediction["score"])
    color = CLASS_COLORS.get(class_id, (255, 255, 255))
    x1 = int(prediction["x1"])
    y1 = int(prediction["y1"])
    x2 = int(prediction["x2"])
    y2 = int(prediction["y2"])
    thickness = max(2, round(frame.shape[1] / 700))

    keypoints = prediction.get("keypoints")
    keypoint_conf = prediction.get("keypoint_conf")
    if isinstance(keypoints, np.ndarray) and isinstance(keypoint_conf, np.ndarray):
        for start_name, end_name in SKELETON_EDGES:
            start_idx = COCO_KEYPOINTS[start_name]
            end_idx = COCO_KEYPOINTS[end_name]
            if keypoint_conf[start_idx] < 0.25 or keypoint_conf[end_idx] < 0.25:
                continue
            start = tuple(keypoints[start_idx].astype(int).tolist())
            end = tuple(keypoints[end_idx].astype(int).tolist())
            cv2.line(frame, start, end, color, max(1, thickness - 1), cv2.LINE_AA)
        for idx, item in enumerate(keypoints):
            if keypoint_conf[idx] >= 0.25:
                cv2.circle(frame, tuple(item.astype(int).tolist()), max(3, thickness + 1), (255, 255, 255), -1)
                cv2.circle(frame, tuple(item.astype(int).tolist()), max(2, thickness), color, -1)

    label = f"{CLASS_DISPLAY_NAMES.get(class_id, class_id)} pose {score:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    label_height = text_size[1] + baseline + 12
    label_width = text_size[0] + 16
    label_y1 = y1 - label_height if y1 > label_height + 4 else y1
    cv2.rectangle(frame, (x1, label_y1), (x1 + label_width, label_y1 + label_height), (22, 24, 29), -1)
    cv2.rectangle(frame, (x1, label_y1), (x1 + 5, label_y1 + label_height), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 10, label_y1 + label_height - baseline - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
