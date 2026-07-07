# 运动目标检测与跟踪课设实验记录

## 当前研究对象

- 任务：基于 YOLOv8-DeepSORT 的行人运动目标检测与跟踪
- 检测对象：person only
- 数据集：自采 3 段视频
  - `bridge.mp4`：天桥上拍摄，直行道路，行人由远及近，远处目标小
  - `cross.mp4`：十字路口，行人较近，存在转弯
  - `car.mp4`：横向道路，行人左右运动，车辆较多，存在遮挡

## 当前代码改动

### 1. 只保留行人类别

文件：`ultralytics/yolo/v8/detect/predict.py`

```python
det = preds[idx]
det = det[det[:, 5] == 0]  # keep person only
```

说明：COCO 中 `person` 类别 ID 为 `0`，该改动用于过滤 bicycle、car、airplane 等无关类别，只分析行人检测与跟踪。

### 2. 输出视频文件名随实验名称变化

文件：`ultralytics/yolo/engine/predictor.py`

当前逻辑：视频输出文件名使用 `name=...` 指定的实验目录名。

示例：

```text
name=bridge_person_smalltarget
输出：runs/detect/bridge_person_smalltarget/bridge_person_smalltarget.mp4
```

### 3. PyTorch 与 NumPy 兼容性修复

- `torch.load(..., weights_only=False)`：兼容 PyTorch 2.6+ 加载旧 YOLOv8 权重
- `np.float` / `np.int` 替换为 `float` / `int`：兼容 NumPy 2.x

### 4. DeepSORT 当前配置

文件：`ultralytics/yolo/v8/detect/deep_sort_pytorch/configs/deep_sort.yaml`

当前已经改为：

```yaml
MAX_AGE: 100
```

原始默认值为：

```yaml
MAX_AGE: 70
```

注意：如果要跑 `MAX_AGE=70` 对照实验，需要手动改回 70 后再运行。

## 基准实验参数

基准设置：

```text
model=yolov8m.pt
imgsz=960
conf=0.3
person only
```

通用 PowerShell 前置命令：

```powershell
cd C:\Users\LuoXue\Desktop\YOLOv8-DeepSORT-Object-Tracking-main\YOLOv8-DeepSORT-Object-Tracking-main
$env:PYTHONPATH = (Get-Location).Path
cd .\ultralytics\yolo\v8\detect
```

## 已设计的对比实验

### 1. bridge 小目标优化实验

目的：改善远距离行人目标过小导致的漏检和 ID 中断。

基准参数：

```powershell
D:\Anaconda3\envs\yolov11\python.exe predict.py model=yolov8m.pt source="runs/detect/bridge.mp4" imgsz=960 conf=0.3 show=False name=bridge_person_base
```

小目标优化参数：

```powershell
D:\Anaconda3\envs\yolov11\python.exe predict.py model=yolov8m.pt source="runs/detect/bridge.mp4" imgsz=1280 conf=0.25 show=False name=bridge_person_smalltarget
```

参数变化：

```text
imgsz: 960 -> 1280
conf: 0.3 -> 0.25
model: yolov8m.pt 不变
类别过滤: person only 不变
```

当前观察结论：

```text
对比截图能看出远距离小目标检测效果提升，行人能更早被检测到。
代价是降低 conf 后可能增加静态物体误检风险。
```

建议截图命名：

```text
figures/before/bridge_smalltarget_before.png
figures/after/bridge_smalltarget_after.png
```

### 2. cross 误检阈值实验

目的：观察提高置信度阈值是否能降低消防水管被误检为 person 的情况。

基准参数：

```powershell
D:\Anaconda3\envs\yolov11\python.exe predict.py model=yolov8m.pt source="runs/detect/cross.mp4" imgsz=960 conf=0.3 show=False name=cross_person_base
```

阈值优化参数：

```powershell
D:\Anaconda3\envs\yolov11\python.exe predict.py model=yolov8m.pt source="runs/detect/cross.mp4" imgsz=960 conf=0.4 show=False name=cross_person_conf04
```

参数变化：

```text
conf: 0.3 -> 0.4
imgsz: 960 不变
model: yolov8m.pt 不变
```

当前观察结论：

```text
conf=0.4 仍可能存在误检，并且漏检率增加。
可作为“提高置信度阈值只能在误检和漏检之间折中”的实验结论。
```

建议截图命名：

```text
figures/before/cross_pipe_false_person.png
figures/after/cross_conf04_compare.png
```

### 3. car 遮挡与 MAX_AGE 实验

目的：观察增大 DeepSORT 的轨迹保留帧数是否能改善短时遮挡后的 ID 连续性。

基准参数，需 `MAX_AGE=70`：

```powershell
D:\Anaconda3\envs\yolov11\python.exe predict.py model=yolov8m.pt source="runs/detect/car.mp4" imgsz=960 conf=0.3 show=False name=car_person_age70
```

遮挡优化参数，当前配置已是 `MAX_AGE=100`：

```powershell
D:\Anaconda3\envs\yolov11\python.exe predict.py model=yolov8m.pt source="runs/detect/car.mp4" imgsz=960 conf=0.3 show=False name=car_person_age100
```

参数变化：

```text
MAX_AGE: 70 -> 100
model: yolov8m.pt 不变
imgsz: 960 不变
conf: 0.3 不变
```

当前观察结论：

```text
car 和 cross 中行人较近、目标较大，遮挡处理本身已经较好。
MAX_AGE=100 主要用于证明短时遮挡可通过延长轨迹保留时间缓解，但无法解决 YOLO 漏检导致的长期丢失。
```

建议截图命名：

```text
figures/before/car_occlusion_age70.png
figures/after/car_occlusion_age100.png
```

## 论文中可写的核心结论

1. `bridge` 场景主要问题是远距离小目标，根源是 YOLO 检测不稳定，不是单纯 DeepSORT 跟踪参数问题。
2. `cross` 场景主要问题是静态物体误检为 person，提高 `conf` 可以降低部分误检，但会增加漏检。
3. `car` 场景遮挡效果相对较好，因为行人距离近、目标大，YOLO 检测较稳定。
4. ID 变化的根本原因往往是检测链路中断：

```text
YOLO 漏检 -> DeepSORT 缺少观测 -> 卡尔曼预测短暂维持 -> 轨迹删除或新建 -> ID 变化
```

5. 本文优化方向不是彻底消除误检/漏检/ID 变化，而是针对不同场景进行参数折中：

```text
小目标：提高 imgsz，降低 conf
误检：提高 conf
遮挡：提高 MAX_AGE
```

## 还需要补充的材料

- [ ] `bridge_smalltarget` 优化前后截图一对，用户已表示已经截出
- [x] `cross_conf04` 误检对比截图
- [x] `car_age100` 遮挡/ID 对比截图
- [ ] 人工统计表：漏检次数、误检次数、ID跳变次数
- [ ] 终端速度信息：记录几组 `Speed: ... ms`
- [ ] 最终论文中确认推荐参数

## 已确认的半定量结果

### bridge 小目标检测对比

对比对象：同一帧、同一区域的远距离行人小目标。

| 实验设置 | 检出行人数 | 误检数 | 结论 |
|---|---:|---:|---|
| `imgsz=960, conf=0.3` | 4 | 0 | 远距离小目标漏检较多 |
| `imgsz=1280, conf=0.25` | 8 | 0 | 小目标检出数量明显增加，且该帧未引入误检 |

论文可写结论：

```text
在 bridge 场景的远距离小目标对比帧中，基准参数仅检出 4 个行人目标；
将输入分辨率由 960 提高到 1280，并将置信度阈值由 0.3 调整为 0.25 后，
同一区域检出目标数提升到 8 个，且该帧误检数仍为 0。
这说明提高输入分辨率能够增强 YOLOv8 对远距离小目标的感知能力。
```

### bridge 误检阈值对比

对比对象：同一帧、同一区域的树木/背景误检场景。

| 实验设置 | 正确检出行人数 | 误检数 | 结论 |
|---|---:|---:|---|
| `conf=0.3` | 6 | 1 | 检出人数较多，但存在树木误检为 person |
| `conf=0.4` | 3 | 0 | 误检被抑制，但正确检出人数下降 |

论文可写结论：

```text
在 bridge 的背景误检对比帧中，将置信度阈值从 0.3 提高到 0.4 后，
误检数由 1 降为 0，说明提高阈值能够抑制部分静态背景误检。
但同一帧正确检出的行人数由 6 降为 3，表明阈值提高会显著降低召回率，
尤其会影响远距离、小尺度或低置信度行人的检测结果。
```

### cross 误检阈值对比

对比对象：同一帧、同一区域的消防水管/静态背景误检场景。

| 实验设置 | 正确检出行人数 | 误检数 | 结论 |
|---|---:|---:|---|
| `conf=0.3` | 8 | 2 | 行人检出较完整，但存在 2 处误检 |
| `conf=0.4` | 7 | 0 | 误检被消除，但漏掉 1 个正确目标 |

论文可写结论：

```text
在 cross 场景中，提高置信度阈值后，误检数由 2 降为 0，
但正确检出人数由 8 降为 7。该结果与 bridge 场景一致，
说明置信度阈值优化具有明确的误检抑制作用，但同时会带来召回率下降。
因此本文将 conf=0.4 作为误检抑制方案，而不作为所有场景的默认最优参数。
```

### car 遮挡与 ID 保持对比

截图位置：`figures/before/car1`、`figures/before/car2`、`figures/before/car3`，分别对应遮挡前、遮挡中、遮挡后。

| 阶段 | 观察内容 | 跟踪结果 |
|---|---|---|
| 遮挡前 | 行人目标正常可见 | ID 已建立 |
| 遮挡中 | 行人被车辆短暂遮挡 | 轨迹由 DeepSORT 预测维持 |
| 遮挡后 | 行人重新出现 | ID 仍然保持 |

论文可写结论：

```text
在 car 场景的短时车辆遮挡片段中，遮挡前、遮挡中和遮挡后三张连续截图显示，
目标在遮挡后仍保持原有 ID。说明 DeepSORT 中的卡尔曼滤波预测和轨迹保留机制
能够在短时间检测缺失时维持目标身份，对近距离大目标的短时遮挡具有较好的鲁棒性。
但该结论主要适用于短时遮挡；若 YOLO 长时间漏检，轨迹仍可能被删除并造成 ID 变化。
```

## 多类交通目标数据集计划

### 标签定义

CVAT Online 项目名称建议为：

```text
traffic_vehicle_detection
```

标签顺序固定为：

```text
0 bus
1 car
2 truck
3 two_wheeler
```

标注规则：

- `car`：小轿车、SUV、出租车、普通私家车、面包车、商务车。
- `bus`：公交车、大巴、校车等明显载客大型车辆。
- `truck`：货车、厢货、工程车等明显载货车辆。
- `two_wheeler`：自行车、摩托车、电动车、电动自行车。
- 行人不标。
- 太远、遮挡严重、类别无法判断的目标不标。
- 停着的车辆也标，因为 YOLO 学习的是目标外观，运动性由后续 DeepSORT 跟踪体现。

### 数据集目录与配置

最终数据集目录建议整理为：

```text
C:/Users/LuoXue/Desktop/traffic_vehicle_dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

工程中已新增数据集配置文件：

```text
traffic_vehicle.yaml
```

内容为：

```yaml
path: C:/Users/LuoXue/Desktop/traffic_vehicle_dataset
train: images/train
val: images/val
test: images/test
nc: 4
names: ['bus', 'car', 'truck', 'two_wheeler']
```

### 训练命令

```powershell
cd C:\Users\LuoXue\Desktop\YOLOv8-DeepSORT-Object-Tracking-main\YOLOv8-DeepSORT-Object-Tracking-main
$env:PYTHONPATH = (Get-Location).Path
cd .\ultralytics\yolo\v8\detect
D:\Anaconda3\envs\yolov11\python.exe train.py model=yolov8m.pt data="C:/Users/LuoXue/Desktop/YOLOv8-DeepSORT-Object-Tracking-main/YOLOv8-DeepSORT-Object-Tracking-main/traffic_vehicle.yaml" epochs=80 imgsz=960 batch=8 name=traffic_4cls_yolov8m
```

### 预测命令

当前 `predict.py` 已将原来的 person-only 硬编码过滤改为可配置参数：

```yaml
class_filter: 0
```

默认值仍为 `0`，用于保持原行人实验的行为。交通四类模型预测时必须显式保留 `0,1,2,3` 四类：

```powershell
D:\Anaconda3\envs\yolov11\python.exe predict.py model="runs/detect/traffic_4cls_yolov8m/weights/best.pt" source="traffic_test.mp4" imgsz=960 conf=0.3 show=False name=traffic_4cls_predict "class_filter=[0,1,2,3]"
```

如需不进行类别过滤，可使用：

```powershell
D:\Anaconda3\envs\yolov11\python.exe predict.py model="runs/detect/traffic_4cls_yolov8m/weights/best.pt" source="traffic_test.mp4" imgsz=960 conf=0.3 show=False name=traffic_4cls_predict_all class_filter=null
```

## 车辆三分类补充实验最终记录

### 数据集划分

本次补充实验将 CVAT 导出的 7 组交通目标数据整理为三分类车辆检测数据集，类别定义如下：

```text
0 bus
1 car
2 two_wheeler
```

由于 `truck` 类样本数为 0，本次训练不保留 truck 类；原始 `two_wheeler` 类别编号由 3 重映射为 2。

数据集划分方式如下，按视频片段划分，避免同一视频相邻帧同时出现在训练集和测试集中：

```text
train: dataset/1, dataset/2, dataset/3, dataset/4, dataset/5
val:   dataset/6
test:  dataset/7
```

合并后的数据规模：

| split | images | labels |
|---|---:|---:|
| train | 2432 | 2432 |
| val | 196 | 196 |
| test | 556 | 556 |

标注框数量：

| class | boxes |
|---|---:|
| bus | 720 |
| car | 9601 |
| two_wheeler | 2814 |

### 训练配置

训练设备为本地 RTX 4060 Laptop GPU。由于 `yolov8m + imgsz=960` 在本地训练时资源占用较高，最终采用轻量化训练配置：

```text
model: yolov8s.pt
epochs: 30
imgsz: 640
batch: 16
workers: 4
cache: False
device: 0
```

训练输出目录：

```text
D:/YOLO/runs/traffic_3cls_yolov8s_fast_b162
```

最优权重：

```text
D:/YOLO/runs/traffic_3cls_yolov8s_fast_b162/weights/best.pt
```

### 验证集训练末轮指标

训练末轮，即 epoch 29，对应指标如下：

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.95897 | 0.70718 | 0.80480 | 0.47277 |

### 测试集评估结果

测试集使用 `dataset/7`，即由 `7.mp4` 抽帧并标注得到的图片序列。测试集评估命令实际使用 `traffic_vehicle_test.yaml`，其中 `val` 指向 `images/test`。

测试集总体结果：

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 556 | 2216 | 0.844 | 0.558 | 0.723 | 0.350 |
| car | 556 | 1137 | 0.841 | 0.966 | 0.967 | 0.517 |
| two_wheeler | 556 | 1079 | 0.848 | 0.149 | 0.478 | 0.183 |

测试结果表明，模型对小汽车目标具有较好的检测能力，`car` 类召回率达到 0.966，mAP50 达到 0.967；但对 `two_wheeler` 类召回率较低，仅为 0.149，说明二轮车目标漏检较多。造成该问题的主要原因包括二轮车目标尺寸较小、外观变化大、遮挡频繁，以及训练样本中二轮车与背景、车辆之间的尺度差异较明显。

### 论文可用结论

本次车辆三分类补充实验表明，自建数据集微调后的 YOLOv8s 能够较好适应实际道路场景中的小汽车检测任务，在测试集上 `car` 类取得 0.967 的 mAP50 和 0.966 的召回率。相比直接使用 COCO 预训练模型，微调模型的类别定义更加贴合本课设场景，可区分 `bus`、`car` 和 `two_wheeler` 三类交通目标。

同时，实验也暴露出二轮车检测性能不足的问题。`two_wheeler` 类 Precision 为 0.848，但 Recall 仅为 0.149，说明当前模型对二轮车预测较谨慎，误检较少但漏检严重。后续若继续提升性能，应优先补充二轮车近景、远景、遮挡和不同方向运动样本，或提高输入分辨率、增加训练轮数，并考虑对小目标进行针对性数据增强。
