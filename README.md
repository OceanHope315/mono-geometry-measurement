# 单目距离估计基线系统（Mono Distance Baseline）

## 项目概述

本项目实现了一个基于单目视频的目标距离估计系统，结合YOLOv8目标检测和MiDaS深度估计，用于驾驶场景中的车辆距离检测。

### 主要特性

- **YOLOv8 目标检测**：检测车、巴士、卡车、摩托车等驾驶相关目标
- **MiDaS 深度估计**：获取整帧深度信息，辅助距离估计
- **几何距离估计**：基于目标真实高度先验 + bbox高度的弱标定距离估计
- **深度融合**：结合几何约束和深度图进行距离修正
- **距离平滑**：使用3帧均值平滑单帧抖动
- **完整输出**：CSV结果表 + 可视化视频输出

## 文件结构

```
.
├── mono_distance_baseline.py      # 主程序
├── requirements.txt               # 依赖包列表
├── README.md                      # 本文件
├── .gitignore                     # Git忽略文件
├── yolov8n.pt                     # YOLOv8 Nano 权重文件
├── demo/                          # 演示视频目录
│   └── 1/
│       └── output.mp4
├── distance_results.csv           # 输出：距离估计结果表
└── distance_baseline_output.avi   # 输出：可视化视频
```

## 安装

### 环境要求
- Python 3.8+
- CUDA (可选，用于加速，需要NVIDIA GPU)

### 安装步骤

1. **克隆或进入项目目录**
   ```bash
   cd path/to/baseline
   ```

2. **创建虚拟环境** (推荐)
   ```bash
   # 使用venv
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate      # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **准备模型权重**
   - `yolov8n.pt` 会在首次运行时自动下载（需要网络连接）
   - MiDaS 模型也会通过 torch.hub 在首次运行时自动下载

## 使用方法

### 基本运行

```bash
python mono_distance_baseline.py
```

系统将默认处理 `./demo/1/output.mp4`。

### 修改输入视频

编辑 `mono_distance_baseline.py` 最后部分：

```python
if __name__ == "__main__":
    video_path = r".\demo\1\output.mp4"  # 修改这里
    app = MonoDistanceBaseline(video_path)
    app.run()
```

### 处理过程中的交互

- **按 `q` 键**：停止处理并保存结果
- **窗口显示**：实时显示检测框和距离估计值

## 输出结果

### 1. CSV 结果表 (`distance_results.csv`)

包含以下字段：
- `frame_id`：帧编号
- `target_id`：目标ID
- `class_name`：目标类别（car/bus/truck/motorcycle）
- `x1, y1, x2, y2`：检测框坐标
- `bbox_height_px`：检测框高度（像素）
- `depth_score`：归一化深度分数（0-1）
- `distance_m`：**估计距离（米）** （已平滑）

### 2. 可视化视频 (`distance_baseline_output.avi`)

实时标注的视频文件，显示：
- 绿色检测框
- 目标类别 + 距离 + 置信度标签

## 核心算法说明

### 距离估计公式

**基础几何估计：**
$$d = \frac{f \cdot H}{h}$$

其中：
- $f = 800$ px：焦距（像素单位）
- $H$：目标真实高度（米）
  - 车：1.5m
  - 巴士：3.0m
  - 卡车：3.2m
  - 摩托车：1.4m
- $h$：检测框高度（像素）

**深度融合修正：**
$$d_{fused} = d_{geo} \times (1 - 0.25 \times depth\_score)$$

修正系数被clip到 [0.75, 1.10]，避免过度修正。

### 距离平滑

对检测框高度易抖动的问题，使用最近3帧均值平滑：
$$d_{smoothed} = mean(d_i, d_{i-1}, d_{i-2})$$

### 深度采样区域优化

为减少背景干扰，深度评分仅从：
- **竖向**：检测框下1/3区域（y_start到y2）
- **横向**：去掉左右各15%的边距（x1+margin到x2-margin）

## 参数调整指南

### 修改相机内参
如有标定数据，修改 `__init__` 中的：
```python
self.focal_length_px = 800.0  # 改为实际焦距
```

### 修改目标真实高度先验
根据实际应用场景调整：
```python
self.car_real_height_m = 1.5
self.bus_real_height_m = 3.0
# ...
```

### 修改检测置信度阈值
```python
if conf < 0.35:  # 降低:检测更多目标; 提高:减少误检
    continue
```

### 修改目标类别
```python
self.valid_classes = {"car", "bus", "truck", "motorcycle"}
```

### 修改深度融合权重
```python
correction = 1.0 - 0.25 * depth_score  # 0.25改为其他值
```

## 已知限制

1. **单目固有限制**：无法处理自车高度变化（俯仰角）
2. **先验依赖**：距离估计严格依赖目标真实高度的准确性
3. **MiDaS依赖**：深度估计目前主要作为趋势修正，不是主要距离信息源
4. **视频格式**：输出使用XVID编码，某些操作系统可能需要额外解码器

## 未来优化方向

- [ ] 第二周：加入Kalman滤波进行时序优化
- [ ] 第二周：集成多对象跟踪（MOT）
- [ ] 后续：标定相机内参获得准确焦距
- [ ] 后续：使用立体匹配或其他深度方案替代MiDaS
- [ ] 后续：加入车道线检测辅助距离估计

## 故障排除

### 问题：MiDaS 模型下载失败

**原因**：网络连接或GitHub不可访问

**解决**：
```python
# 在 __init__ 中，MiDaS 加载失败时系统会自动降级
# 此时仅使用几何距离估计，距离精度会降低
```

### 问题：YOLO 模型下载失败

**原因**：首次运行需要下载yolov8n.pt

**解决**：
```bash
# 手动下载到当前目录或设置YOLO_HOME环境变量
# https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 问题：视频无法打开

**检查**：
- 视频文件路径是否正确
- 视频格式是否支持（推荐 .mp4）
- OpenCV 是否支持该编码

### 问题：内存溢出

**原因**：大分辨率视频或长视频导致MiDaS占用过多显存

**解决**：
- 降低视频分辨率
- 使用GPU较少的 MiDaS_small（已默认使用）

## 参考文献

- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- MiDaS: [Intel ISL](https://github.com/intel-isl/MiDaS)

## 许可证

[MIT License] (可根据实际项目修改)

## 联系信息

如有问题或建议，欢迎反馈。

---

**最后更新**：2026年4月8日
