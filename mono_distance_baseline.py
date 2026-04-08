import cv2
import csv
import torch
import numpy as np
from ultralytics import YOLO


class MonoDistanceBaseline:
    def __init__(self, video_path: str, output_csv: str = "distance_results2.csv"):
        self.video_path = video_path
        self.output_csv = output_csv

        # 只保留和驾驶目标相关的类别
        self.valid_classes = {"car", "bus", "truck", "motorcycle"}

        # ========== 1. 加载 YOLO ==========
        self.detector = YOLO("yolov8n.pt")

        # ========== 2. 加载 MiDaS ==========
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.midas = None
        self.transform = None
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.midas.to(self.device)
            self.midas.eval()

            # MiDaS 预处理
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
        except Exception as e:
            print("警告：无法通过 torch.hub 下载或加载 MiDaS 模型，已禁用深度辅助距离估计。")
            print("请检查网络连接或 GitHub 访问权限。错误详情：", e)
            self.midas = None
            self.transform = None

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

        self.frame_id = 0
        self.target_counter = 0

        # 【新增】帧率和时间步长
        self.fps = 25.0
        self.dt = 1.0 / 25.0

        # 【新增】目标跟踪状态
        # tracks: {track_id: {"center_x": x, "center_y": y, "bbox": (x1,y1,x2,y2), "class_name": str, "last_frame": frame_id}}
        self.tracks = {}

        # 【新增】每个target_id单独维护的距离历史（最近5帧）
        self.target_distance_history = {}

        # 【新增】每个target_id单独维护的速度历史（最近5帧）
        self.target_speed_history = {}

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

    def assign_track_id(self, x1: float, y1: float, x2: float, y2: float, class_name: str) -> int:
        """
        改进的目标跟踪，结合中心点距离、IoU 和类别一致性约束。

        逻辑：
        1. 只在类别相同的 tracks 中找候选
        2. 计算中心点距离和 IoU
        3. 使用评分函数：score = center_distance - 100 * iou
        4. 选择最小 score 的候选，且满足距离 < 50 或 IoU > 0.1
        5. 否则分配新 ID
        6. 更新 track 的中心点、bbox、class_name 和 last_frame
        """
        curr_cx = (x1 + x2) / 2.0
        curr_cy = (y1 + y2) / 2.0
        curr_box = (x1, y1, x2, y2)

        best_track_id = None
        best_score = float('inf')
        distance_threshold = 50.0  # 像素
        iou_threshold = 0.1

        # 只在类别相同的 tracks 中搜索
        for track_id, track in self.tracks.items():
            if track["class_name"] != class_name:
                continue

            track_cx = track["center_x"]
            track_cy = track["center_y"]
            track_box = track["bbox"]

            # 计算中心点距离
            center_distance = np.sqrt((curr_cx - track_cx) ** 2 +
                                      (curr_cy - track_cy) ** 2)

            # 计算 IoU
            iou = self.compute_iou(curr_box, track_box)

            # 评分函数：中心距离减去 IoU 加权
            score = center_distance - 100 * iou

            # 检查是否满足匹配条件
            if (center_distance < distance_threshold or iou > iou_threshold) and score < best_score:
                best_score = score
                best_track_id = track_id

        # 决策：复用已有 ID 或分配新 ID
        if best_track_id is not None:
            assigned_id = best_track_id
            self.tracks[assigned_id]["center_x"] = curr_cx
            self.tracks[assigned_id]["center_y"] = curr_cy
            self.tracks[assigned_id]["bbox"] = curr_box
            self.tracks[assigned_id]["class_name"] = class_name
            self.tracks[assigned_id]["last_frame"] = self.frame_id
        else:
            # 分配新 ID
            self.target_counter += 1
            assigned_id = self.target_counter
            self.tracks[assigned_id] = {
                "center_x": curr_cx,
                "center_y": curr_cy,
                "bbox": curr_box,
                "class_name": class_name,
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

        return assigned_id

    def estimate_depth_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        使用 MiDaS 估计整帧深度图
        输出是相对深度，不是直接米
        如果 MiDaS 模型不可用，则返回全 0 的深度图，后续仅使用几何距离估计。
        """
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

        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 1e-6:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.zeros_like(depth_map, dtype=np.float32)

        depth_map = np.nan_to_num(
            depth_map, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return depth_map

    def get_object_real_height(self, class_name: str) -> float:
        if class_name == "car":
            return self.car_real_height_m
        if class_name == "bus":
            return self.bus_real_height_m
        if class_name == "truck":
            return self.truck_real_height_m
        if class_name == "motorcycle":
            return self.motorcycle_real_height_m
        return 1.5

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

        # 1. 整帧深度估计
        depth_map = self.estimate_depth_map(frame_bgr)

        # 2. 目标检测
        results = self.detector(frame_bgr, verbose=False)

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

                # 4. 融合几何弱标定 + 深度趋势
                distance_m = self.fuse_depth_and_geometry(
                    class_name, bbox_height_px, depth_score)

                # 【新增】检查距离估计有效性，无效则跳过该检测框
                if distance_m < 0:
                    continue

                # 【修改】使用改进跟踪获取target_id，传入class_name以增强匹配约束
                target_id = self.assign_track_id(x1, y1, x2, y2, class_name)

                # ========== 【新增】距离跳变抑制 ==========
                # 核心逻辑：如果该target已有历史且当前distance_m与上一帧平滑距离变化超过30%，
                # 则基于前一帧平滑距离做温和更新而不是直接采用当前值（防止bbox高度抖动污染后续速度/TTC）
                if target_id in self.target_distance_history and self.target_distance_history[target_id]:
                    # 取上一帧的平滑距离
                    prev_smoothed = self.robust_smooth_distance(
                        self.target_distance_history[target_id])

                    # 计算相对变化：当前值 vs 上一帧平滑值
                    rel_change = abs(distance_m - prev_smoothed) / \
                        max(prev_smoothed, 1e-6)

                    # 若相对变化超过30%，做温和指数混合（70% 前帧 + 30% 当前）
                    if rel_change > 0.3:
                        distance_m = 0.7 * prev_smoothed + 0.3 * distance_m

                # 5. 距离平滑（按目标单独维护历史，鲁棒平滑减少抖动对速度/TTC的放大）
                if target_id not in self.target_distance_history:
                    self.target_distance_history[target_id] = []
                self.target_distance_history[target_id].append(distance_m)
                if len(self.target_distance_history[target_id]) > 5:
                    self.target_distance_history[target_id].pop(0)
                smoothed_distance = self.robust_smooth_distance(
                    self.target_distance_history[target_id])

                # 【新增】相对速度计算（线性拟合历史距离斜率，更抗抖动）
                # 正速度表示接近（距离下降），负速度表示远离（距离增大）
                velocity_mps = self.estimate_velocity_from_history(
                    self.target_distance_history[target_id])
                # 【新增】物理限幅：防止极端异常值污染TTC
                velocity_mps = np.clip(
                    velocity_mps, -self.max_abs_speed_mps, self.max_abs_speed_mps)

                # 平滑速度（最近5个速度值的均值）
                if target_id not in self.target_speed_history:
                    self.target_speed_history[target_id] = []
                self.target_speed_history[target_id].append(velocity_mps)
                if len(self.target_speed_history[target_id]) > 5:
                    self.target_speed_history[target_id].pop(0)
                smoothed_velocity = np.mean(
                    self.target_speed_history[target_id]) if self.target_speed_history[target_id] else velocity_mps
                # 【新增】二次限幅：确保最终速度也在合理范围
                smoothed_velocity = np.clip(
                    smoothed_velocity, -self.max_abs_speed_mps, self.max_abs_speed_mps)

                # 【新增】TTC计算
                # 仅当目标接近速度超过最小有效阈值时才计算TTC，以抑制噪声
                if smoothed_velocity > self.min_valid_approach_speed_mps:
                    ttc_s = smoothed_distance / smoothed_velocity
                else:
                    ttc_s = float("inf")
                # 对显示进行简单截断，避免极大值
                ttc_display = min(ttc_s, 99.9) if ttc_s != float(
                    "inf") else 99.9

                # 【新增】更新本帧最危险目标
                if ttc_s != float("inf") and ttc_s < min_ttc:
                    min_ttc = ttc_s
                    min_ttc_target_id = target_id
                    # 【新增】调试：打印最危险目标详细信息
                    if self.debug:
                        print(
                            f"[DEBUG] Frame {self.frame_id} MinTTC Target: ID={target_id}, Class={class_name}, Dist={smoothed_distance:.1f}m, Vel={smoothed_velocity:.1f}m/s, TTC={ttc_s:.1f}s")

                # 6. 可视化
                # 根据TTC值选择框颜色
                if ttc_s < self.ttc_danger:
                    color = (0, 0, 255)  # 红色，危险
                elif ttc_s < self.ttc_warning:
                    color = (0, 255, 255)  # 黄色，警告
                else:
                    color = (0, 255, 0)  # 绿色，安全

                cv2.rectangle(vis_frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 2)

                # 【修改】根据重要性构建标签：重要目标显示完整标签，非重要只显示简短标签
                is_important = (smoothed_distance < 25) or (
                    ttc_s != float("inf")) or (target_id == min_ttc_target_id)
                if is_important:
                    if ttc_s == float("inf"):
                        ttc_str = "inf"
                    else:
                        ttc_str = f"{min(ttc_s, 99.9):.1f}"
                    label = f"{class_name} | {smoothed_distance:.1f}m | v:{smoothed_velocity:.1f}m/s | TTC:{ttc_str}s"
                else:
                    label = f"{class_name} | {smoothed_distance:.1f}m"
                cv2.putText(
                    vis_frame,
                    label,
                    (int(x1), max(20, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                # 7. 写入 CSV
                ttc_csv = "inf" if ttc_s == float("inf") else round(ttc_s, 3)
                self.csv_writer.writerow([
                    self.frame_id, target_id, class_name,
                    round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                    round(bbox_height_px, 2),
                    round(depth_score, 4),
                    round(smoothed_distance, 2),
                    round(smoothed_velocity, 3),
                    ttc_csv
                ])

        # 【新增】保存本帧最危险目标信息
        self.current_min_ttc = min_ttc
        self.current_min_ttc_target_id = min_ttc_target_id

        # 【新增】每隔10帧打印调试信息
        if self.debug and self.frame_id % 10 == 0:
            print(
                f"[DEBUG] Frame {self.frame_id}: MinTTC_ID={self.current_min_ttc_target_id}, MinTTC={self.current_min_ttc:.1f}s")

        return vis_frame

    def run(self):
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
            self.video_writer = cv2.VideoWriter(
                self.output_video,
                cv2.VideoWriter_fourcc(*'XVID'),
                fps,
                (width, height)
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
        print(f"处理完成，结果已保存到: {self.output_csv} 和 {self.output_video}")


if __name__ == "__main__":
    video_path = r".\demo\2\output.mp4"
    app = MonoDistanceBaseline(video_path)
    app.run()
