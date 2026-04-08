# 单目距离估计与碰撞风险估计系统（Mono Distance & TTC Baseline）

## 项目概述

本项目实现了一个基于单目视频的目标距离、速度和碰撞时间（TTC）估计系统，结合YOLOv8目标检测、MiDaS深度估计和轻量级目标跟踪，用于驾驶场景中的车辆距离和碰撞风险评估。

### 主要特性

- **YOLOv8 目标检测**：检测车、巴士、卡车、摩托车等驾驶相关目标
- **MiDaS 深度估计**：获取整帧深度信息，辅助距离估计
- **轻量级目标跟踪**：基于中心点欧氏距离的跨帧ID关联，无需额外库依赖
- **几何距离估计**：基于目标真实高度先验 + bbox高度的弱标定距离估计
- **深度融合**：结合几何约束和深度图进行距离修正
- **多帧平滑**：距离和速度分别使用5帧均值平滑，按目标单独维护历史
- **相对速度计算**：基于距离时间序列，计算目标接近速度（m/s）
- **TTC 碰撞风险**：Time-To-Collision 实时估计，支持二级告警阈值（危险/警告）
- **动态颜色告警**：检测框颜色实时反映风险等级（绿色/黄色/红色）
- **完整输出**：CSV结果表（含距离、速度、TTC）+ 可视化视频输出

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
- `distance_m`：**估计距离（米）** （已平滑，基于最近5帧）
- `relative_speed_mps`：**相对速度（米/秒）** （已平滑，基于最近5帧，正值表示接近）
- `ttc_s`：**碰撞时间（秒）** （Time-To-Collision，或"inf"表示不接近）

### 2. 可视化视频 (`distance_baseline_output.avi`)

实时标注的视频文件，显示：
- **动态颜色检测框**：
  - 🟢 **绿色**：TTC > 5.0 秒，安全
  - 🟡 **黄色**：2.0 秒 ≤ TTC ≤ 5.0 秒，警告
  - 🔴 **红色**：TTC < 2.0 秒，危险
- **目标标签**：类别 | 距离(m) | 相对速度(m/s) | TTC(秒)
  - 示例：`car | 12.3m | v:2.4m/s | TTC:5.1s`
- **目标ID稳定跨帧**：同一目标在视频中保持一致的追踪ID

### 3. 目标跟踪跨帧ID

系统为每个检测目标分配唯一ID，基于：
- **跟踪策略**：中心点欧氏距离匹配
- **匹配阈值**：50像素
- **历史清理**：10帧未出现的目标自动删除
- **优势**：无需额外库（如DeepSORT），轻量级且高效

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

使用**最近5帧均值**平滑距离，按目标单独维护：
$$d_{smoothed} = mean(d_i, d_{i-1}, d_{i-2}, d_{i-3}, d_{i-4})$$

每个目标有独立的距离历史，避免多目标混淆。

### 深度采样区域优化

为减少背景干扰，深度评分仅从：
- **竖向**：检测框下1/3区域（y_start到y2）
- **横向**：去掉左右各15%的边距（x1+margin到x2-margin）

### 相对速度计算

基于距离时间序列计算目标接近速度：
$$v = \frac{d_{t-1} - d_t}{\Delta t}$$

其中：
- $d_t$：当前帧距离
- $d_{t-1}$：前一帧距离
- $\Delta t$：帧级时间间隔（秒）

**符号约定**：
- **正速度**（v > 0）：目标接近（距离减小）
- **负速度**（v < 0）：目标远离（距离增大）
- **零速度**（|v| ≈ 0）：目标斜行或静止

速度同样使用最近5帧均值平滑，按目标单独维护。

### TTC（Time-To-Collision）

碰撞时间估计公式：
$$TTC = \begin{cases} \frac{d}{v} & \text{if } v > 10^{-6} \\ \infty & \text{otherwise} \end{cases}$$

**物理意义**：
- TTC = 5.0s：按当前速度继续接近，5秒后碰撞
- TTC = inf：目标不在接近（远离或静止）

**应用**：
- **危险阈值** (self.ttc_danger = 2.0s)：即时碰撞风险，红色框
- **警告阈值** (self.ttc_warning = 5.0s)：需要警惕，黄色框
- **安全区间**：TTC > 5.0s，绿色框

### 未检测到的特殊情况

- **距离估计失败** (distance_m < 0)：跳过该检测框，不进入速度/TTC计算
- **速度计算不足** (少于2帧历史)：速度设为0，TTC设为inf
- **目标远离或静止** (v ≤ 1e-6)：TTC设为inf（不显示危险）

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

### 修改目标跟踪参数

tracking中心点匹配阈值：
```python
# 在assign_track_id()中
distance_threshold = 50.0  # 像素，增大:跟踪更宽松; 减小:跟踪更严格

# 清理旧目标
max_age = 10  # 帧数，增大:保留更久; 减小:快速清理
```

### 修改TTC告警阈值

根据应用场景调整危险/警告等级：
```python
self.ttc_danger = 2.0   # 红色危险阈值（秒）
self.ttc_warning = 5.0  # 黄色警告阈值（秒）
```

## 已知限制

1. **单目固有限制**：无法处理自车高度变化（俯仰角）、横向运动
2. **先验依赖**：距离估计严格依赖目标真实高度的准确性
3. **MiDaS依赖**：深度估计目前主要作为趋势修正，相对值有意义但绝对值精度有限
4. **视频格式**：输出使用XVID编码，某些操作系统可能需要额外解码器
5. **跟踪局限**：简单中心点匹配跟踪，不适合高密度场景或大幅度运动
6. **速度计算**：仅基于单目距离变化，未考虑相机运动补偿
7. **TTC假设**：假设目标维持当前速度，不预测目标加减速

## 未来优化方向

- [ ] **第二周**：
  - Kalman滤波进行时序优化
  - 多对象跟踪（MOT）增强跟踪鲁棒性
  - 目标加速度估计改进TTC预测
  
- [ ] **后续**：
  - 集成相机标定获得准确焦距
  - 相机运动补偿（光流或SLAM）
  - 立体匹配或其他深度方案替代MiDaS
  - 车道线检测辅助距离估计
  - TTC曲线实时绘图
  - 事件录制和告警触发

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
