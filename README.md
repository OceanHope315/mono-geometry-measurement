# 单目驾驶场景几何测量系统

本项目实现了一个基于单目视频的驾驶场景目标级几何测量 baseline。

系统从单目行车视频中检测车辆目标，并估计目标的：

- 距离 Distance
- 相对速度 Relative Speed
- 碰撞时间 Time-To-Collision，简称 TTC

本项目主要用于课程项目、组会汇报和算法原型验证。当前结果更适合作为**趋势分析和风险指标**，不应直接理解为高精度真实测距结果。

---

## 1. 项目功能

本系统支持以下功能：

- 使用 YOLO 检测驾驶场景中的车辆目标
- 使用 ByteTrack 或 IoU + 中心点距离进行跨帧目标跟踪
- 使用 Depth Anything V2 Metric Outdoor 或 MiDaS 估计深度图
- 基于单目几何公式估计目标距离
- 基于距离历史估计目标相对速度
- 根据距离和相对速度计算 TTC
- 输出可视化视频、CSV 数据、深度图和曲线图

---

## 2. 当前系统流程

整体处理流程如下：

```text
输入视频
    ↓
逐帧读取图像
    ↓
YOLO 检测车辆目标
    ↓
ByteTrack / IoU 目标跟踪
    ↓
Depth Anything V2 / MiDaS 深度估计
    ↓
bbox 几何距离估计
    ↓
距离序列平滑
    ↓
相对速度估计
    ↓
TTC 计算
    ↓
输出视频、CSV、深度图和曲线图
````

---

## 3. 项目结构

推荐项目结构如下：

```text
mono-geometry-measurement/
├── mono_distance_baseline.py
├── README.md
├── requirements.txt
├── .gitignore
├── depth_anything_v2/
│   ├── dpt.py
│   └── ...
├── weights/
│   └── .gitkeep
├── videos/
│   └── .gitkeep
└── simple/
    ├── csv/
    ├── avi/
    ├── depth/
    ├── intermediate/
    ├── distance_curve/
    ├── speed_curve/
    └── ttc_curve/
```

说明：

* `mono_distance_baseline.py`：主程序
* `depth_anything_v2/`：Depth Anything V2 相关代码
* `weights/`：存放 YOLO 权重文件
* `videos/`：存放输入视频
* `simple/`：默认输出目录
* `csv/`：输出每一帧的测量结果
* `avi/`：输出可视化视频
* `depth/`：输出深度图
* `intermediate/`：输出原图与深度图拼接后的中间结果
* `distance_curve/`：输出目标距离曲线
* `speed_curve/`：输出目标速度曲线
* `ttc_curve/`：输出目标 TTC 曲线

---

## 4. 环境要求

推荐环境：

* Python 3.10 或 Python 3.11
* PyTorch
* OpenCV
* Ultralytics YOLO
* NumPy
* Matplotlib

安装依赖：

```bash
pip install -r requirements.txt
```

一个基础的 `requirements.txt` 可以写成：

```text
opencv-python
torch
torchvision
numpy
matplotlib
ultralytics
```

如果使用 Depth Anything V2，需要保证项目目录中存在：

```text
depth_anything_v2/
```

否则代码中的导入语句会报错：

```python
from depth_anything_v2.dpt import DepthAnythingV2
```

---

## 5. 模型权重准备

本仓库不建议上传大模型权重文件。

需要用户自行下载并放置模型权重。

---

### 5.1 YOLO 检测模型

代码中默认使用：

```python
self.detector_weights = r"D:\baseline\weights\yolo11s.pt"
```

建议将 YOLO 权重放在：

```text
weights/yolo11s.pt
```

也可以使用其他 YOLO 权重，例如：

```text
yolov8n.pt
yolo11s.pt
yolo11m.pt
```

如果指定的 YOLO11s 权重不存在，代码会尝试回退到：

```text
yolov8n.pt
```

---

### 5.2 Depth Anything V2 Metric Outdoor 权重

代码中默认使用：

```python
self.depth_model_path = r"D:\baseline\Depth-Anything-V2\checkpoints\depth_anything_v2_metric_vkitti_vits.pth"
```

建议将 Depth Anything V2 权重放在：

```text
Depth-Anything-V2/checkpoints/
```

例如：

```text
Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_vits.pth
```

如果 Depth Anything V2 加载失败，程序会尝试回退到 MiDaS。

如果 MiDaS 也无法加载，程序会禁用深度估计，仅使用 bbox 几何距离估计。

---

## 6. 输入视频

将输入视频放到：

```text
videos/
```

例如：

```text
videos/video_5.mp4
```

当前主程序默认处理：

```python
video_path = r".\videos\video_5.mp4"
```

如果要处理其他视频，可以修改这行代码：

```python
video_path = r".\videos\your_video.mp4"
```

---

## 7. 运行方法

在项目根目录运行：

```bash
python mono_distance_baseline.py
```

程序会依次执行：

1. 加载 YOLO 检测模型
2. 加载 Depth Anything V2 或回退到 MiDaS
3. 读取输入视频
4. 逐帧检测和跟踪车辆目标
5. 估计距离、相对速度和 TTC
6. 保存可视化结果和数据文件

运行过程中会弹出可视化窗口。

按下：

```text
q
```

可以提前退出程序。

---

## 8. 输出结果

默认输出目录为：

```text
simple/
```

输出内容包括：

```text
simple/
├── csv/
│   └── video_1.csv
├── avi/
│   └── video_1.avi
├── depth/
│   └── depth_0001.png
├── intermediate/
│   └── vis_0001.png
├── distance_curve/
│   └── video_1_target_x_distance.png
├── speed_curve/
│   └── video_1_target_x_speed.png
└── ttc_curve/
    └── video_1_target_x_ttc.png
```

---

## 9. CSV 字段说明

CSV 文件中每一行表示某一帧中某一个目标的测量结果。

| 字段                   | 含义            |
| -------------------- | ------------- |
| `frame_id`           | 当前帧编号         |
| `target_id`          | 目标跟踪 ID       |
| `class_name`         | 稳定后的目标类别      |
| `x1, y1, x2, y2`     | 目标检测框坐标       |
| `bbox_height_px`     | 检测框高度，单位为像素   |
| `depth_score`        | 目标区域的深度分数     |
| `distance_m`         | 平滑后的估计距离，单位为米 |
| `relative_speed_mps` | 估计相对速度，单位为米/秒 |
| `ttc_s`              | 碰撞时间 TTC，单位为秒 |

---

## 10. 核心算法说明

---

### 10.1 车辆检测

系统使用 YOLO 检测驾驶相关目标。

当前保留的类别包括：

```text
car
bus
truck
motorcycle
```

检测结果包括：

* 类别
* 置信度
* bbox 坐标

低置信度目标会被过滤。

---

### 10.2 目标跟踪

系统优先使用 ByteTrack 提供的目标 ID。

如果 ByteTrack 没有返回 ID，则使用自定义的目标关联逻辑。

自定义关联逻辑主要依据：

* bbox 中心点距离
* IoU
* 类别一致性软约束

这样可以尽量保持同一个真实目标在多帧中的 `target_id` 一致。

---

### 10.3 稳定类别

在实际视频中，同一辆车可能会被 YOLO 在不同帧中识别为不同类别。

例如：

```text
car → truck
truck → car
```

为了减少类别抖动，系统为每个目标维护类别历史，并使用多数投票得到稳定类别。

当前阶段的重点是距离、速度和 TTC，而不是精细车型识别。

因此在显示层中，`car / truck / bus` 可以统一显示为：

```text
vehicle
```

这样可以避免画面中出现“小车被显示成 truck”的问题。

---

### 10.4 距离估计

系统使用单目几何公式估计距离：

```text
distance = focal_length × real_object_height / bbox_height
```

含义是：

* 目标在图像中越大，距离通常越近
* 目标在图像中越小，距离通常越远

当前代码中使用弱先验参数：

```python
self.focal_length_px = 800.0
```

为了降低 `car / truck / bus` 类别抖动对距离的影响，当前可以将这些车辆统一使用相近的真实高度先验。

---

### 10.5 深度估计

系统支持 Depth Anything V2 Metric Outdoor。

如果加载失败，会尝试使用 MiDaS。

深度图主要用于辅助距离趋势修正。

需要注意：

> 当前深度估计结果不是严格的真实距离，只能作为单目测距的辅助信息。

---

### 10.6 距离平滑

bbox 抖动会导致距离估计抖动。

因此系统为每个目标维护最近几帧的距离历史，并进行鲁棒平滑。

主要策略包括：

* 最近 5 帧距离历史
* 中位数异常值过滤
* 距离突变抑制

---

### 10.7 相对速度估计

系统根据目标距离随时间的变化估计相对速度。

如果距离逐渐减小，说明目标正在接近。

如果距离逐渐增大，说明目标正在远离。

系统使用多帧距离历史进行线性拟合，避免只用相邻两帧差分造成速度尖峰。

速度符号约定：

```text
正速度：目标正在接近
负速度：目标正在远离
0 附近：目标距离基本稳定
```

---

### 10.8 TTC 计算

TTC 是 Time-To-Collision，即碰撞时间。

计算公式：

```text
TTC = distance / relative_speed
```

只有当目标明显接近时，才计算有效 TTC。

如果目标没有明显接近，则：

```text
TTC = inf
```

在曲线图中，为了方便显示，`inf` 可能会被截断为 20 秒。

这并不代表真实 TTC 等于 20 秒，而是表示当前没有有效碰撞时间。

---

## 11. 当前局限性

本项目当前仍存在以下限制：

1. 单目距离估计依赖相机焦距和目标真实高度先验。
2. bbox 抖动会影响距离估计。
3. 目标跟踪仍可能出现 ID 切换或轨迹分裂。
4. YOLO 对车辆类别的区分较粗，容易出现 `car / truck` 抖动。
5. Depth Anything V2 / MiDaS 深度结果主要用于辅助趋势，不是严格真值。
6. TTC 更适合作为风险趋势指标，不应被当作真实物理碰撞时间。

---

## 12. 后续改进方向

后续可以从以下方向继续优化：

* 使用更稳定的多目标跟踪算法，例如 ByteTrack、DeepSORT 或 OC-SORT
* 使用驾驶场景数据集微调 YOLO
* 增加车辆细分类模型，识别 sedan、SUV、van、truck 等类别
* 使用更准确的 metric depth 模型
* 加入相机标定参数
* 增加车道线或自车运动信息
* 改进距离、速度和 TTC 的有效性门控
* 将 TTC 明确区分为 valid / invalid 状态
* 添加更完善的实验评估指标

---
