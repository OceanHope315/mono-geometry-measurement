import cv2
import csv
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

from depth_anything_v2.dpt import DepthAnythingV2


class MonoDistanceBaseline:
    def __init__(self, video_path: str, output_csv: str = "distance_results2.csv"):
        self.video_path = video_path
        self.output_csv = output_csv

        # 只保留和驾驶目标相关的类别
        self.valid_classes = {"car", "bus", "truck", "motorcycle"}

        # ========== 1. 模型路径配置 ==========
        self.detector_weights = r"D:\baseline\weights\yolo11s.pt"
        self.depth_model_path = r"D:\baseline\Depth-Anything-V2\checkpoints\depth_anything_v2_metric_vkitti_vits.pth"
        # ========== 2. 加载视频检测模型 YOLO11s ==========
        self.detector = self.load_yolo_detector(self.detector_weights)

        # ========== 3. 加载深度估计模型 ==========
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.depth_method = "none"
        self.depth_model = None
        self.transform = None
        self.load_depth_model()

        # 相机弱先验参数（第1周先简化）
        self.focal_length_px = 800.0
        self.car_real_height_m = 1.5   # 车辆平均高度粗略先验
        self.bus_real_height_m = 3.0
        self.truck_real_height_m = 3.2
        self.motorcycle_real_height_m = 1.4

        # 输出文件
        self.csv_file = open(self.output_csv, "w",
                             newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame_id", "target_id", "class_name",
            "x1", "y1", "x2", "y2",
            "bbox_height_px", "depth_score", "distance_m",
            "relative_speed_mps", "ttc_s"
        ])

        # 输出视频
        self.output_video = "distance_baseline_output2.avi"
        self.video_writer = None

        # 可选的图像输出目录，由 process_video_list 设置
        self.plot_distance_dir = None
        self.plot_speed_dir = None
        self.plot_ttc_dir = None
        self.depth_dir = None
        self.intermediate_dir = None

        self.frame_id = 0
        self.target_counter = 0

        # 【新增】帧率和时间步长
        self.fps = 25.0
        self.dt = 1.0 / 25.0

        # 【新增】目标跟踪状态
        # tracks: {track_id: {"center_x": x, "center_y": y, "bbox": (x1,y1,x2,y2), "class_name": str, "last_frame": frame_id, "bbox_history": []}}
        self.tracks = {}

        # 【新增】每个target_id单独维护的距离历史（最近5帧）
        self.target_distance_history = {}

        # 【新增】每个target_id单独维护的速度历史（最近5帧）
        self.target_speed_history = {}

        # 【新增】主目标状态变量，用于后续只关注最重要的前车
        self.main_target_id = None
        self.main_target_min_track_frames = 5
        self.main_target_max_center_offset_ratio = 0.25
        self.main_target_max_distance_m = 35.0
        self.target_track_age = {}
        self.target_positive_speed_count = {}

        # 【新增】所有目标的序列数据，用于后续绘图
        # {target_id: {"frames": [], "distance": [], "speed": [], "ttc": []}}
        self.all_target_data = {}

        # 【新增】TTC告警阈值
        self.ttc_warning = 5.0  # 单位：秒，黄色告警
        self.ttc_danger = 2.0   # 单位：秒，红色危险

        # 【新增】相对速度物理限幅（防止极端异常值污染TTC）
        self.max_abs_speed_mps = 15.0  # 单位：m/s，最大合理相对速度

        # 【新增】TTC最小有效接近速度阈值（抑制微小噪声导致的虚假碰撞时间）
        self.min_valid_approach_speed_mps = 0.5  # 单位：m/s，低于此值不计算TTC

        # 【新增】本帧最危险目标追踪
        self.current_min_ttc = float("inf")
        self.current_min_ttc_target_id = None

        # 【新增】调试开关
        self.debug = True

    def compute_iou(self, boxA, boxB) -> float:
        """
        计算两个边界框的IoU（Intersection over Union）
        boxA, boxB格式：(x1, y1, x2, y2)
        """
        x1_A, y1_A, x2_A, y2_A = boxA
        x1_B, y1_B, x2_B, y2_B = boxB

        # 计算交集
        x1_inter = max(x1_A, x1_B)
        y1_inter = max(y1_A, y1_B)
        x2_inter = min(x2_A, x2_B)
        y2_inter = min(y2_A, y2_B)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # 计算并集
        boxA_area = (x2_A - x1_A) * (y2_A - y1_A)
        boxB_area = (x2_B - x1_B) * (y2_B - y1_B)
        union_area = boxA_area + boxB_area - inter_area

        if union_area <= 1e-6:
            return 0.0

        iou = inter_area / union_area
        return float(iou)

    def _compute_stable_class(self, class_history: list, current_stable: str, track_age: int) -> str:
        if not class_history:
            return current_stable

        counts = {}
        for c in class_history:
            counts[c] = counts.get(c, 0) + 1

        majority_class = max(counts, key=counts.get)
        majority_count = counts[majority_class]

    # 最近 5 帧里某类别出现 3 次，就认为它更可信
        if majority_count >= 3:
            return majority_class

        return current_stable

    def assign_track_id(self, x1: float, y1: float, x2: float, y2: float, class_name: str) -> int:
        """
        改进的目标跟踪，结合中心点距离、IoU 和类别一致性软约束。

        逻辑：
        1. 允许跨类别匹配，但不同类别会增加 score 惩罚
        2. 计算中心点距离和 IoU
        3. 使用评分函数：score = center_distance - 100 * iou + class_penalty
        4. 选择最小 score 的候选，且满足距离 < 50 或 IoU > 0.1
        5. 否则分配新 ID
        6. 更新 track 的中心点、bbox、class_name、class_history 和 stable_class
        """
        curr_cx = (x1 + x2) / 2.0
        curr_cy = (y1 + y2) / 2.0
        curr_box = (x1, y1, x2, y2)

        best_track_id = None
        best_score = float('inf')
        distance_threshold = 50.0  # 稍微放宽距离阈值，配合软约束
        iou_threshold = 0.1
        class_penalty = 30.0      # 类别不一致时的惩罚分

        # 在所有已有 tracks 中搜索
        for track_id, track in self.tracks.items():
            track_cx = track["center_x"]
            track_cy = track["center_y"]
            track_box = track["bbox"]

            # 计算中心点距离
            center_distance = np.sqrt((curr_cx - track_cx) ** 2 +
                                      (curr_cy - track_cy) ** 2)

            # 计算 IoU
            iou = self.compute_iou(curr_box, track_box)

            # 评分函数：中心距离减去 IoU 加权 + 类别惩罚
            penalty = 0.0 if track["class_name"] == class_name else class_penalty
            score = center_distance - 100 * iou + penalty

            # 检查是否满足匹配条件
            if (center_distance < distance_threshold or iou > iou_threshold) and score < best_score:
                best_score = score
                best_track_id = track_id

        # 决策：复用已有 ID 或分配新 ID
        if best_track_id is not None:
            assigned_id = best_track_id
            old_stable = self.tracks[assigned_id].get(
                "stable_class", class_name)
            self.tracks[assigned_id].update({
                "center_x": curr_cx,
                "center_y": curr_cy,
                "bbox": curr_box,
                "class_name": class_name,
                "last_frame": self.frame_id
            })

            self.tracks[assigned_id]["class_history"].append(class_name)
            if len(self.tracks[assigned_id]["class_history"]) > 5:
                self.tracks[assigned_id]["class_history"].pop(0)

            track_age = self.target_track_age.get(assigned_id, 0)
            self.tracks[assigned_id]["stable_class"] = self._compute_stable_class(
                self.tracks[assigned_id]["class_history"], old_stable, track_age)

            # 更新 bbox_history
            if "bbox_history" not in self.tracks[assigned_id]:
                self.tracks[assigned_id]["bbox_history"] = []
            self.tracks[assigned_id]["bbox_history"].append(curr_box)
            if len(self.tracks[assigned_id]["bbox_history"]) > 5:
                self.tracks[assigned_id]["bbox_history"].pop(0)

        else:
            # 分配新 ID
            self.target_counter += 1
            assigned_id = self.target_counter
            self.tracks[assigned_id] = {
                "center_x": curr_cx,
                "center_y": curr_cy,
                "bbox": curr_box,
                "class_name": class_name,
                "stable_class": class_name,
                "class_history": [class_name],
                "bbox_history": [curr_box],
                "last_frame": self.frame_id
            }

        # 清理长时间未出现的 tracks（超过 10 帧未出现则删除）
        # 同时清理相关历史缓存（此处需要同步清理所有 target 级缓存）
        max_age = 10
        tracks_to_remove = [
            t_id for t_id, track in self.tracks.items()
            if self.frame_id - track["last_frame"] > max_age
        ]
        for t_id in tracks_to_remove:
            del self.tracks[t_id]
            # 同步清理该 target 的历史缓存
            if t_id in self.target_distance_history:
                del self.target_distance_history[t_id]
            if t_id in self.target_speed_history:
                del self.target_speed_history[t_id]
            if t_id in self.target_track_age:
                del self.target_track_age[t_id]
            if t_id in self.target_positive_speed_count:
                del self.target_positive_speed_count[t_id]

        return assigned_id

    def update_track_with_id(self, target_id, x1: float, y1: float, x2: float, y2: float, class_name: str) -> int:
        """
        使用 ByteTrack 提供的 ID 更新或初始化 track。
        如果 track 不存在则创建，否则更新。
        """
        curr_cx = (x1 + x2) / 2.0
        curr_cy = (y1 + y2) / 2.0
        curr_box = (x1, y1, x2, y2)

        if target_id not in self.tracks:
            # 新 track，初始化
            self.tracks[target_id] = {
                "center_x": curr_cx,
                "center_y": curr_cy,
                "bbox": curr_box,
                "class_name": class_name,
                "stable_class": class_name,
                "class_history": [class_name],
                "bbox_history": [curr_box],
                "last_frame": self.frame_id
            }
        else:
            # 已有 track，更新
            old_stable = self.tracks[target_id].get("stable_class", class_name)

            self.tracks[target_id].update({
                "center_x": curr_cx,
                "center_y": curr_cy,
                "bbox": curr_box,
                "class_name": class_name,
                "last_frame": self.frame_id
            })

            self.tracks[target_id]["class_history"].append(class_name)
            if len(self.tracks[target_id]["class_history"]) > 7:
                self.tracks[target_id]["class_history"].pop(0)

            track_age = self.target_track_age.get(target_id, 0)
            self.tracks[target_id]["stable_class"] = self._compute_stable_class(
                self.tracks[target_id]["class_history"],
                old_stable,
                track_age
            )

            self.tracks[target_id]["bbox_history"].append(curr_box)
            if len(self.tracks[target_id]["bbox_history"]) > 5:
                self.tracks[target_id]["bbox_history"].pop(0)

        return target_id

    def get_smoothed_bbox(self, bbox_history: list) -> tuple:
        """
        对bbox历史做滑动平均，返回平滑后的bbox。
        """
        if not bbox_history:
            return (0, 0, 0, 0)
        if len(bbox_history) == 1:
            return bbox_history[0]

        # 取最近3-5帧
        recent = bbox_history[-5:] if len(bbox_history) >= 5 else bbox_history

        # 计算平均
        x1s = [b[0] for b in recent]
        y1s = [b[1] for b in recent]
        x2s = [b[2] for b in recent]
        y2s = [b[3] for b in recent]

        smoothed_x1 = np.mean(x1s)
        smoothed_y1 = np.mean(y1s)
        smoothed_x2 = np.mean(x2s)
        smoothed_y2 = np.mean(y2s)

        return (smoothed_x1, smoothed_y1, smoothed_x2, smoothed_y2)

    def load_yolo_detector(self, weights_path: str):
        if os.path.exists(weights_path):
            try:
                return YOLO(weights_path)
            except Exception as e:
                print(
                    f"警告：无法加载 YOLO11s 权重 '{weights_path}'，回退到 yolov8n.pt。错误：{e}")
        else:
            print(f"警告：未找到 YOLO11s 权重文件 '{weights_path}'，回退到 yolov8n.pt。")
        return YOLO("yolov8n.pt")

    def load_depth_model(self):
        """加载 Depth Anything V2 Metric Outdoor，如失败回退到 MiDaS。"""
        try:
            encoder = "vits"  # small 版本

            model_configs = {
                "vits": {
                    "encoder": "vits",
                    "features": 64,
                    "out_channels": [48, 96, 192, 384]
                },
                "vitb": {
                    "encoder": "vitb",
                    "features": 128,
                    "out_channels": [96, 192, 384, 768]
                },
                "vitl": {
                    "encoder": "vitl",
                    "features": 256,
                    "out_channels": [256, 512, 1024, 1024]
                },
            }

            if not os.path.exists(self.depth_model_path):
                raise FileNotFoundError(f"找不到深度模型权重: {self.depth_model_path}")

            self.depth_model = DepthAnythingV2(**model_configs[encoder])

            state_dict = torch.load(self.depth_model_path,
                                    map_location=self.device)
            self.depth_model.load_state_dict(state_dict)

            self.depth_model = self.depth_model.to(self.device).eval()
            self.depth_method = "depth_anything"

            print(
                f"已加载 Depth Anything V2 Metric Outdoor: {self.depth_model_path}")
            return

        except Exception as e:
            print("警告：Depth Anything 模型加载失败，尝试回退 MiDaS。", e)

        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.midas.to(self.device)
            self.midas.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform

            self.depth_method = "midas"
            print("使用 MiDaS 深度模型作为回退。")

        except Exception as e:
            print("警告：无法加载任何深度模型，已禁用深度估计。", e)
            self.midas = None
            self.transform = None
            self.depth_method = "none"

    def estimate_depth_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        使用 Depth Anything 或 MiDaS 估计整帧深度图。
        输出是归一化相对深度，不是直接米。
        若没有深度模型，则返回全 0 的深度图。
        """
        if self.depth_method == "depth_anything":
            return self.estimate_depth_map_depth_anything(frame_bgr)

        if self.depth_method == "midas":
            if self.midas is None or self.transform is None:
                return np.zeros(frame_bgr.shape[:2], dtype=np.float32)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(frame_rgb).to(self.device)

            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False
                ).squeeze()

            depth_map = prediction.cpu().numpy()
        else:
            return np.zeros(frame_bgr.shape[:2], dtype=np.float32)

        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 1e-6:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.zeros_like(depth_map, dtype=np.float32)

        depth_map = np.nan_to_num(
            depth_map, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return depth_map

    def estimate_depth_map_depth_anything(self, frame_bgr: np.ndarray) -> np.ndarray:
        try:
            with torch.no_grad():
                depth_map = self.depth_model.infer_image(frame_bgr)

            depth_map = np.asarray(depth_map)

            depth_min = np.nanmin(depth_map)
            depth_max = np.nanmax(depth_map)

            if depth_max - depth_min > 1e-6:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_map = np.zeros(frame_bgr.shape[:2], dtype=np.float32)

            depth_map = np.nan_to_num(
                depth_map,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            ).astype(np.float32)

            return depth_map

        except Exception as e:
            print("警告：Depth Anything 推理失败，回退到 MiDaS 或空深度图。", e)
            self.depth_method = "midas"
            return self.estimate_depth_map(frame_bgr)

    def get_object_real_height(self, class_name: str) -> float:
        # 当前阶段重点是距离/速度/TTC，不是车型细分类
        # car/truck/bus 的误分类会造成距离跳变，所以先统一车辆高度
        if class_name in {"car", "truck", "bus", "vehicle"}:
            return 1.7
        if class_name == "motorcycle":
            return 1.4
        return 1.7

    def robust_depth_score(self, depth_map: np.ndarray, bbox) -> float:
        """
        只取目标框的下半部分/下1/3区域，降低背景干扰
        左右两边裁掉15%以进一步减少背景干扰
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]

        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w - 1, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h - 1, int(y2)))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        # 只取下 1/3 区域
        y_start = int(y1 + (y2 - y1) * 2 / 3)
        # 左右裁掉15%
        x_margin = int((x2 - x1) * 0.15)
        roi = depth_map[y_start:y2, x1 + x_margin:x2 - x_margin]

        if roi.size == 0:
            return 0.0

        valid = roi[np.isfinite(roi)]
        if valid.size == 0:
            return 0.0

        # 去掉极端值后取中位数
        low = np.percentile(valid, 10)
        high = np.percentile(valid, 90)
        valid = valid[(valid >= low) & (valid <= high)]

        if valid.size == 0:
            return 0.0

        return float(np.median(valid))

    def robust_smooth_distance(self, history: list) -> float:
        """
        鲁棒距离平滑：使用中位数和异常值去除，减少bbox抖动对距离的影响。

        逻辑：
        1. 如果历史为空，返回 -1
        2. 如果历史长度 <= 2，直接返回均值
        3. 否则：
           a. 取最近最多5个值
           b. 计算中位数
           c. 去掉与中位数偏差过大的异常值（偏差 > 20%中位数或绝对差 > 2m）
           d. 至少保留1个值
           e. 对剩余值取均值
        """
        if not history:
            return -1.0

        if len(history) <= 2:
            return float(np.mean(history))

        # 取最近最多5个值
        recent = history[-5:]

        # 计算中位数
        median = float(np.median(recent))

        if median <= 1e-6:
            return float(np.mean(recent))

        # 异常值检测：偏差 > 20% 中位数
        anomaly_threshold = max(0.2 * median, 2.0)  # 20% 中位数或 2 米，取较大值
        filtered = [x for x in recent if abs(x - median) <= anomaly_threshold]

        # 至少保留 1 个值
        if not filtered:
            filtered = recent

        return float(np.mean(filtered))

    def estimate_velocity_from_history(self, distance_history: list) -> float:
        """
        基于最近几帧距离历史的线性拟合斜率估计相对速度。
        相比相邻帧差分，更抗抖动和噪声。

        逻辑：
        1. 如果历史长度 < 2，返回 0.0
        2. 取最近最多 5 帧距离
        3. 构造时间序列 t = [0, dt, 2*dt, 3*dt, ...]
        4. 用 numpy.polyfit(t, distances, 1) 拟合直线 d = a*t + b
        5. 返回 -a 作为相对速度（负的斜率）

        符号约定：
        - 正速度：距离下降，目标接近
        - 负速度：距离增大，目标远离
        - 0 速度：距离稳定
        """
        if not distance_history or len(distance_history) < 2:
            return 0.0

        # 取最近最多 5 帧距离
        recent_distances = distance_history[-5:]
        n = len(recent_distances)

        # 构造时间序列：t = [0, dt, 2*dt, ...]
        time_seq = np.array([i * self.dt for i in range(n)], dtype=np.float32)
        distance_seq = np.array(recent_distances, dtype=np.float32)

        # 线性拟合：d = a*t + b
        # polyfit 返回 [a, b]
        coeffs = np.polyfit(time_seq, distance_seq, 1)
        a = coeffs[0]  # 斜率（距离相对于时间的导数）

        # 相对速度 = -斜率（距离下降时速度为正）
        velocity_mps = float(-a)

        return velocity_mps

    def estimate_distance_from_bbox(self, class_name: str, bbox_height_px: float) -> float:
        """
        第1周简化版：利用目标真实高度先验 + bbox高度 做近似米制估计
        distance = f * H / h
        """
        if bbox_height_px <= 1e-6:
            return -1.0

        real_height = self.get_object_real_height(class_name)
        distance_m = self.focal_length_px * real_height / bbox_height_px
        return float(distance_m)

    def fuse_depth_and_geometry(self, class_name: str, bbox_height_px: float, depth_score: float) -> float:
        """
        第1周的简化融合：
        主体用几何距离（更容易直接得到“米”）
        深度分数用于修正趋势
        """
        geo_distance = self.estimate_distance_from_bbox(
            class_name, bbox_height_px)

        if geo_distance < 0:
            return -1.0

        # MiDaS depth_map 归一化后，通常“值大/值小”的前后关系需要按实际观察调
        # 这里做一个弱修正：depth_score 越大，认为越近，适当缩小距离
        correction = 1.0 - 0.25 * depth_score
        correction = np.clip(correction, 0.75, 1.10)

        fused_distance = geo_distance * correction
        return float(fused_distance)

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        self.frame_id += 1
        vis_frame = frame_bgr.copy()

        # 【新增】初始化本帧最危险目标追踪
        min_ttc = float("inf")
        min_ttc_target_id = None

        # 【新增】主目标候选列表，用于后续选取最重要的前车目标
        frame_width = frame_bgr.shape[1]
        main_target_candidates = []

        # 第一阶段：收集每个检测目标的中间结果
        detections_info = []

        # 1. 整帧深度估计
        depth_map = self.estimate_depth_map(frame_bgr)

        # 生成深度图可视化
        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        # 保存深度图可视化
        if self.depth_dir:
            cv2.imwrite(os.path.join(
                self.depth_dir, f"depth_{self.frame_id:04d}.png"), depth_vis)

        # 2. 目标检测和跟踪
        results = self.detector.track(
            frame_bgr, persist=True, tracker='bytetrack.yaml', verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = self.detector.names[cls_id]

                if class_name not in self.valid_classes:
                    continue
                if conf < 0.35:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox_height_px = max(1.0, y2 - y1)

                # 3. 从深度图中取目标区域的鲁棒深度分数
                depth_score = self.robust_depth_score(
                    depth_map, (x1, y1, x2, y2))

                # 4. 目标 ID 关联：优先使用 ByteTrack 的 ID，否则自己分配
                if box.id is not None:
                    target_id = int(box.id[0].item())
                    target_id = self.update_track_with_id(
                        target_id, x1, y1, x2, y2, class_name)
                else:
                    target_id = self.assign_track_id(
                        x1, y1, x2, y2, class_name)

                stable_class = self.tracks[target_id]["stable_class"]

                # 获取平滑后的bbox
                smoothed_bbox = self.get_smoothed_bbox(
                    self.tracks[target_id]["bbox_history"])
                smoothed_x1, smoothed_y1, smoothed_x2, smoothed_y2 = smoothed_bbox
                smoothed_center_x = (smoothed_x1 + smoothed_x2) / 2.0
                smoothed_center_y = (smoothed_y1 + smoothed_y2) / 2.0
                smoothed_bbox_height_px = max(1.0, smoothed_y2 - smoothed_y1)

                # 5. 基于稳定类别融合几何弱标定 + 深度趋势
                distance_m = self.fuse_depth_and_geometry(
                    stable_class, smoothed_bbox_height_px, depth_score)

                if distance_m < 0:
                    continue

                if target_id in self.target_distance_history and self.target_distance_history[target_id]:
                    prev_smoothed = self.robust_smooth_distance(
                        self.target_distance_history[target_id])
                    rel_change = abs(distance_m - prev_smoothed) / \
                        max(prev_smoothed, 1e-6)
                    if rel_change > 0.3:
                        distance_m = 0.7 * prev_smoothed + 0.3 * distance_m

                if target_id not in self.target_distance_history:
                    self.target_distance_history[target_id] = []
                self.target_distance_history[target_id].append(distance_m)
                if len(self.target_distance_history[target_id]) > 5:
                    self.target_distance_history[target_id].pop(0)
                smoothed_distance = self.robust_smooth_distance(
                    self.target_distance_history[target_id])

                raw_speed_mps = 0.0
                history = self.target_distance_history[target_id]
                if len(history) >= 2 and self.dt > 1e-6:
                    raw_speed_mps = (history[-2] - history[-1]) / self.dt

                velocity_mps = self.estimate_velocity_from_history(history)
                velocity_mps = np.clip(
                    velocity_mps, -self.max_abs_speed_mps, self.max_abs_speed_mps)

                if target_id not in self.target_speed_history:
                    self.target_speed_history[target_id] = []
                self.target_speed_history[target_id].append(velocity_mps)
                if len(self.target_speed_history[target_id]) > 5:
                    self.target_speed_history[target_id].pop(0)
                smoothed_velocity = np.mean(
                    self.target_speed_history[target_id]) if self.target_speed_history[target_id] else velocity_mps
                smoothed_velocity = np.clip(
                    smoothed_velocity, -self.max_abs_speed_mps, self.max_abs_speed_mps)

                if target_id not in self.target_track_age:
                    self.target_track_age[target_id] = 0
                self.target_track_age[target_id] += 1

                if smoothed_velocity > self.min_valid_approach_speed_mps:
                    self.target_positive_speed_count[target_id] = self.target_positive_speed_count.get(
                        target_id, 0) + 1
                else:
                    self.target_positive_speed_count[target_id] = 0

                center_x = smoothed_center_x
                if smoothed_distance < self.main_target_max_distance_m:
                    center_offset_ratio = abs(
                        center_x - frame_width / 2.0) / frame_width
                    if center_offset_ratio < self.main_target_max_center_offset_ratio:
                        main_target_candidates.append({
                            "target_id": target_id,
                            "smoothed_distance": smoothed_distance,
                            "center_x": center_x
                        })

                if smoothed_velocity > self.min_valid_approach_speed_mps:
                    ttc_s = smoothed_distance / smoothed_velocity
                else:
                    ttc_s = float("inf")

                if ttc_s != float("inf") and ttc_s < min_ttc:
                    min_ttc = ttc_s
                    min_ttc_target_id = target_id
                    if self.debug:
                        print(
                            f"[DEBUG] Frame {self.frame_id} MinTTC Target: ID={target_id}, StableClass={stable_class}, Dist={smoothed_distance:.1f}m, Vel={smoothed_velocity:.1f}m/s, TTC={ttc_s:.1f}s")

                detections_info.append({
                    "target_id": target_id,
                    "class_name": class_name,
                    "stable_class": stable_class,
                    "x1": smoothed_x1,
                    "y1": smoothed_y1,
                    "x2": smoothed_x2,
                    "y2": smoothed_y2,
                    "bbox_height_px": smoothed_bbox_height_px,
                    "depth_score": depth_score,
                    "distance_m": distance_m,
                    "smoothed_distance": smoothed_distance,
                    "raw_speed_mps": raw_speed_mps,
                    "velocity_mps": velocity_mps,
                    "smoothed_velocity": smoothed_velocity,
                    "ttc_s": ttc_s,
                    "center_x": smoothed_center_x,
                })

        if main_target_candidates:
            if self.main_target_id is not None:
                still_main = next(
                    (c for c in main_target_candidates if c["target_id"] == self.main_target_id), None)
                if still_main is not None:
                    self.main_target_id = still_main["target_id"]
                else:
                    self.main_target_id = min(
                        main_target_candidates, key=lambda c: c["smoothed_distance"])["target_id"]
            else:
                self.main_target_id = min(
                    main_target_candidates, key=lambda c: c["smoothed_distance"])["target_id"]
        else:
            self.main_target_id = None

        self.current_min_ttc = min_ttc
        self.current_min_ttc_target_id = min_ttc_target_id

        for det in detections_info:
            target_id = det["target_id"]
            ttc_s = det["ttc_s"]
            smoothed_velocity = det["smoothed_velocity"]
            smoothed_distance = det["smoothed_distance"]
            distance_m = det["distance_m"]
            raw_speed_mps = det["raw_speed_mps"]
            velocity_mps = det["velocity_mps"]

            show_full_ttc = (
                target_id == self.main_target_id
                and self.target_track_age.get(target_id, 0) >= self.main_target_min_track_frames
                and self.target_positive_speed_count.get(target_id, 0) >= 3
            )

            # 先确定框的颜色，后面 cv2.rectangle 才能使用 color
            if show_full_ttc:
                if ttc_s < self.ttc_danger:
                    color = (0, 0, 255)       # 红色：危险
                elif ttc_s < self.ttc_warning:
                    color = (0, 255, 255)     # 黄色：警告
                else:
                    color = (0, 255, 0)       # 绿色：安全
            else:
                color = (0, 255, 0)           # 默认绿色

            cv2.rectangle(
                vis_frame,
                (int(det["x1"]), int(det["y1"])),
                (int(det["x2"]), int(det["y2"])),
                color,
                2
            )

            # 显示层统一车辆类别，避免 car / truck / bus 抖动影响观感
            display_class = det["stable_class"]
            if display_class in {"car", "truck", "bus"}:
                display_class = "vehicle"

            if show_full_ttc:
                if ttc_s == float("inf"):
                    ttc_str = "inf"
                else:
                    ttc_str = f"{min(ttc_s, 99.9):.1f}"

                label = (
                    f"{display_class} | {smoothed_distance:.1f}m | "
                    f"v:{smoothed_velocity:.1f}m/s | TTC:{ttc_str}s"
                )
            else:
                label = f"{display_class} | {smoothed_distance:.1f}m"

            cv2.putText(
                vis_frame,
                label,
                (int(det["x1"]), max(20, int(det["y1"]) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # --- 收集多目标绘图数据 ---
            if target_id not in self.all_target_data:
                self.all_target_data[target_id] = {
                    "frames": [],
                    "distance": [],
                    "speed": [],
                    "ttc": []
                }

            # 存入数据
            self.all_target_data[target_id]["frames"].append(self.frame_id)
            self.all_target_data[target_id]["distance"].append(
                smoothed_distance)
            self.all_target_data[target_id]["speed"].append(smoothed_velocity)
            self.all_target_data[target_id]["ttc"].append(ttc_s)

            ttc_csv = "inf" if ttc_s == float("inf") else round(ttc_s, 3)
            self.csv_writer.writerow([
                self.frame_id, target_id, det["stable_class"],
                round(det["x1"], 2), round(det["y1"], 2), round(
                    det["x2"], 2), round(det["y2"], 2),
                round(det["bbox_height_px"], 2),
                round(det["depth_score"], 4),
                round(smoothed_distance, 2),
                round(smoothed_velocity, 3),
                ttc_csv
            ])

        # 拼接原始帧与深度图
        combined_vis = np.hstack((vis_frame, depth_vis))

        # 保存中间可视化结果
        if self.intermediate_dir:
            cv2.imwrite(os.path.join(
                self.intermediate_dir, f"vis_{self.frame_id:04d}.png"), combined_vis)

        if self.debug and self.frame_id % 10 == 0:
            print(
                f"[DEBUG] Frame {self.frame_id}: MinTTC_ID={self.current_min_ttc_target_id}, MinTTC={self.current_min_ttc:.1f}s")

        return combined_vis

    def save_all_target_plots(self, video_name: str):
        if not self.all_target_data:
            print(f"[{video_name}] 没有收集到目标数据，跳过绘图。")
            return

        # 过滤并排序：选择跟踪帧数最多的前3个目标
        valid_targets = []
        for tid, data in self.all_target_data.items():
            if len(data["frames"]) >= 10:  # 至少跟踪10帧才绘图
                valid_targets.append((tid, data))

        # 按帧数降序排列，取前3个
        top_targets = sorted(valid_targets, key=lambda x: len(
            x[1]["frames"]), reverse=True)[:3]

        if not top_targets:
            print(f"[{video_name}] 没有满足条件（>10帧）的目标，跳过绘图。")
            return

        for tid, data in top_targets:
            frames = data["frames"]
            dist = data["distance"]
            speed = data["speed"]
            ttc = data["ttc"]

            # 1. 距离曲线
            plt.figure()
            plt.plot(frames, dist, marker='o', markersize=2)
            plt.xlabel("Frame ID")
            plt.ylabel("Distance (m)")
            plt.title(f"Target {tid} Distance Curve ({video_name})")
            plt.grid(True)
            dist_name = f"{video_name}_target_{tid}_distance.png"
            dist_path = os.path.join(
                self.plot_distance_dir, dist_name) if self.plot_distance_dir else dist_name
            plt.savefig(dist_path)
            plt.close()

            # 2. 速度曲线
            plt.figure()
            plt.plot(frames, speed, marker='o', markersize=2, color='orange')
            plt.xlabel("Frame ID")
            plt.ylabel("Speed (m/s)")
            plt.title(f"Target {tid} Speed Curve ({video_name})")
            plt.grid(True)
            speed_name = f"{video_name}_target_{tid}_speed.png"
            speed_path = os.path.join(
                self.plot_speed_dir, speed_name) if self.plot_speed_dir else speed_name
            plt.savefig(speed_path)
            plt.close()

            # 3. TTC 曲线
            ttc_plot = [20.0 if x == float("inf") else x for x in ttc]
            plt.figure()
            plt.plot(frames, ttc_plot, marker='o', markersize=2, color='red')
            plt.xlabel("Frame ID")
            plt.ylabel("TTC (s)")
            plt.title(f"Target {tid} TTC Curve ({video_name}, inf->20)")
            plt.grid(True)
            ttc_name = f"{video_name}_target_{tid}_ttc.png"
            ttc_path = os.path.join(
                self.plot_ttc_dir, ttc_name) if self.plot_ttc_dir else ttc_name
            plt.savefig(ttc_path)
            plt.close()

        print(
            f"[{video_name}] 已为前 {len(top_targets)} 个目标生成了 {len(top_targets)*3} 张曲线图。")

    def run(self, video_name: str = "output"):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {self.video_path}")
            self.csv_file.close()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-6:
            fps = 25.0

        # 【新增】更新类变量fps和dt
        self.fps = fps
        self.dt = 1.0 / self.fps

        print(f"开始处理视频，FPS = {fps:.2f}")
        print("按 q 退出。")

        # 初始化视频写入器
        ret, first_frame = cap.read()
        if ret:
            height, width = first_frame.shape[:2]
            # 拼接深度图后，宽度变为 2 倍
            self.video_writer = cv2.VideoWriter(
                self.output_video,
                cv2.VideoWriter_fourcc(*'XVID'),
                fps,
                (width * 2, height)
            )
            # 处理第一帧
            vis = self.process_frame(first_frame)
            cv2.imshow("Mono Distance Baseline", vis)
            if self.video_writer:
                self.video_writer.write(vis)
        else:
            print("无法读取视频帧")
            cap.release()
            self.csv_file.close()
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            vis = self.process_frame(frame)
            cv2.imshow("Mono Distance Baseline", vis)
            if self.video_writer:
                self.video_writer.write(vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        if self.video_writer:
            self.video_writer.release()
        self.csv_file.close()
        self.save_all_target_plots(video_name)
        print(f"处理完成，结果已保存到: {self.output_csv} 和 {self.output_video}")


def process_video_list(video_dir: str, output_dir: str = None):
    """批量处理视频目录下的所有 MP4 视频，并输出对应的 AVI 与 CSV 文件。"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    video_files = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith(".mp4")
    ])

    if not video_files:
        print(f"未找到 {video_dir} 下的 MP4 视频文件。")
        return

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        base_name = os.path.splitext(video_file)[0]
        if output_dir:
            csv_dir = os.path.join(output_dir, "csv")
            avi_dir = os.path.join(output_dir, "avi")
            distance_plot_dir = os.path.join(output_dir, "distance_curve")
            speed_plot_dir = os.path.join(output_dir, "speed_curve")
            ttc_plot_dir = os.path.join(output_dir, "ttc_curve")
            depth_dir = os.path.join(output_dir, "depth")
            intermediate_dir = os.path.join(output_dir, "intermediate")
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(avi_dir, exist_ok=True)
            os.makedirs(distance_plot_dir, exist_ok=True)
            os.makedirs(speed_plot_dir, exist_ok=True)
            os.makedirs(ttc_plot_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(intermediate_dir, exist_ok=True)

            output_csv = os.path.join(
                csv_dir, f"{base_name}.csv")
            output_video = os.path.join(
                avi_dir, f"{base_name}.avi")
        else:
            output_csv = f"{base_name}.csv"
            output_video = f"{base_name}.avi"

        print(f"\n====== 处理视频: {video_file} ======")
        app = MonoDistanceBaseline(video_path, output_csv=output_csv)
        app.output_video = output_video
        if output_dir:
            app.plot_distance_dir = distance_plot_dir
            app.plot_speed_dir = speed_plot_dir
            app.plot_ttc_dir = ttc_plot_dir
            app.depth_dir = depth_dir
            app.intermediate_dir = intermediate_dir
        app.run(video_name=base_name)


if __name__ == "__main__":
    # video_dir = r".\videos"
    # output_dir = r".\results"
    # process_video_list(video_dir, output_dir)

    # 仅跑 video_1 的代码
    video_path = r".\videos\video_5.mp4"
    output_dir = r".\simple"

    # 手动调用 process_video_list 的逻辑，但只针对 video_1
    if not os.path.exists(video_path):
        print(f"错误：找不到视频文件 {video_path}")
    else:
        # 模拟 process_video_list 的目录结构
        base_name = "video_1"
        csv_dir = os.path.join(output_dir, "csv")
        avi_dir = os.path.join(output_dir, "avi")
        distance_plot_dir = os.path.join(output_dir, "distance_curve")
        speed_plot_dir = os.path.join(output_dir, "speed_curve")
        ttc_plot_dir = os.path.join(output_dir, "ttc_curve")
        depth_dir = os.path.join(output_dir, "depth")
        intermediate_dir = os.path.join(output_dir, "intermediate")

        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(avi_dir, exist_ok=True)
        os.makedirs(distance_plot_dir, exist_ok=True)
        os.makedirs(speed_plot_dir, exist_ok=True)
        os.makedirs(ttc_plot_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(intermediate_dir, exist_ok=True)

        output_csv = os.path.join(csv_dir, f"{base_name}.csv")
        output_video = os.path.join(avi_dir, f"{base_name}.avi")

        print(f"\n====== 正在处理 (Simple 模式): {video_path} ======")
        app = MonoDistanceBaseline(video_path, output_csv=output_csv)
        app.output_video = output_video
        app.plot_distance_dir = distance_plot_dir
        app.plot_speed_dir = speed_plot_dir
        app.plot_ttc_dir = ttc_plot_dir
        app.depth_dir = depth_dir
        app.intermediate_dir = intermediate_dir
        app.run(video_name=base_name)
