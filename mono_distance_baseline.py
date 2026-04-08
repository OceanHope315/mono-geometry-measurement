import cv2
import csv
import torch
import numpy as np
from ultralytics import YOLO


class MonoDistanceBaseline:
    def __init__(self, video_path: str, output_csv: str = "distance_results.csv"):
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
            "bbox_height_px", "depth_score", "distance_m"
        ])

        # 距离平滑缓冲区（最近3帧均值）
        self.distance_buffer = []

        # 输出视频
        self.output_video = "distance_baseline_output.avi"
        self.video_writer = None

        self.frame_id = 0
        self.target_counter = 0

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

                # 5. 距离平滑（最近3帧均值）
                self.distance_buffer.append(distance_m)
                if len(self.distance_buffer) > 3:
                    self.distance_buffer.pop(0)
                smoothed_distance = np.mean(
                    self.distance_buffer) if self.distance_buffer else distance_m

                self.target_counter += 1
                target_id = self.target_counter

                # 6. 可视化
                color = (0, 255, 0)
                cv2.rectangle(vis_frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 2)

                label = f"{class_name} | {smoothed_distance:.1f} m | conf {conf:.2f}"
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
                self.csv_writer.writerow([
                    self.frame_id, target_id, class_name,
                    round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                    round(bbox_height_px, 2),
                    round(depth_score, 4),
                    round(smoothed_distance, 2)
                ])

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
    video_path = r".\demo\1\output.mp4"
    app = MonoDistanceBaseline(video_path)
    app.run()
