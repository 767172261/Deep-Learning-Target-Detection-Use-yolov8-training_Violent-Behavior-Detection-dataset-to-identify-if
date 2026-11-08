深度学习目标检测使用yolov8训练_暴力行为检测数据集来识别是否为暴力打架行为进行检测


@[toc]

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d6258c04456641c596892df78e95dade.png)暴力行为检测数据集，non_violence（非暴力），violence（暴力）两类别。数据集有6126张图片，训练集、验证集与测试集均已按比例划分完成，数据标记为txt格式，标签与图片一一对应，YOLO系列模型训练、![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9715ddfefa7a462f80a2dce819a97ade.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b05446e8a96841dba60ea9e183be671f.png)
以下文章及代码仅供参考。

使用YOLOv8训练一个暴力行为检测系统，基于数据集（包含`non_violence`和`violence`两类标签），我们将经历以下几个步骤：安装必要的依赖、准备数据集、配置模型训练参数、执行训练、评估模型性能以及可视化预测结果，步骤说明和代码示例。

### 1. 安装依赖

首先确保你的环境中已经安装了必要的库：

```bash
pip install ultralytics opencv-python matplotlib
```

### 2. 数据集准备

假设你的数据集已经按照比例划分成了训练集、验证集和测试集，并且每个图像都有对应的`.txt`格式的标签文件。我们创建一个`data.yaml`文件来描述这些信息：

```yaml
train: ./path/to/train/images/
val: ./path/to/val/images/
test: ./path/to/test/images/

nc: 2  # 类别数量
names: ['non_violence', 'violence']  # 类别名称
```

请根据实际路径替换上述`./path/to/...`部分。

### 3. 训练模型

接下来，编写Python脚本来加载YOLOv8模型并开始训练过程。这里是一个完整的例子：

```python
from ultralytics import YOLO
import os

def train_yolov8_model():
    # 加载预训练的YOLOv8模型
    model = YOLO('yolov8n.pt')  # 可以选择其他变体如'yolov8s.pt', 'yolov8m.pt'等
    
    # 开始训练
    results = model.train(
        data='path/to/data.yaml',  # 数据集配置文件路径
        epochs=100,  # 根据需要调整epoch数
        imgsz=640,  # 输入图像尺寸
        batch=16,  # 批次大小
        name='violence_detection',  # 实验名称
        save=True,  # 自动保存最佳模型
        exist_ok=True,  # 如果目录存在则不报错
        patience=50,  # 当验证损失不再改善时提前停止训练的轮数
        lr0=0.01,  # 初始学习率
        lrf=0.1,  # 最终学习率（lr0 * lrf）
        optimizer='SGD',  # 优化器类型
        augment=True,  # 启用数据增强
        device='',  # 使用GPU或CPU ('cuda' or 'cpu')
        workers=8,  # 数据加载的工作线程数
        project='runs/detect',  # 保存结果的项目目录
    )

if __name__ == "__main__":
    train_yolov8_model()
```

### 4. 评估模型

训练完成后，可以使用验证集评估模型性能：

```python
from ultralytics import YOLO

def validate_model():
    model = YOLO('runs/detect/violence_detection/weights/best.pt')  # 加载最佳模型
    metrics = model.val()  # 进行验证
    print(metrics.box.map)  # 输出mAP值作为性能指标之一

if __name__ == "__main__":
    validate_model()
```

### 5. 可视化预测结果

我们可以对单张图片进行预测，并可视化其结果：

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def predict_and_visualize(image_path):
    model = YOLO('runs/detect/violence_detection/weights/best.pt')  # 加载最佳模型
    results = model.predict(image_path, conf=0.5)

    for result in results:
        for box in result.boxes:
            cls, conf = box.cls.item(), box.conf.item()
            x1, y1, x2, y2 = map(int, box.xyxy)
            label = model.model.names[int(cls)]
            image = cv2.imread(image_path)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    predict_and_visualize("path/to/your/image.jpg")
```

### 6. 模型选择

在训练过程中，YOLOv8会自动保存表现最好的模型权重到指定的目录中（例如`runs/detect/violence_detection/weights/best.pt`）。你可以直接使用这个模型进行预测和部署。


为了使用YOLOv8训练一个暴力行为检测系统，基于你提供的数据集（包含`non_violence`和`violence`两类标签），我们将经历以下几个步骤：安装必要的依赖、准备数据集、配置模型训练参数、执行训练、评估模型性能以及可视化预测结果。以下是详细的步骤说明和代码示例。

### 1. 安装依赖

首先确保你的环境中已经安装了必要的库：

```bash
pip install ultralytics opencv-python matplotlib
```

### 2. 数据集准备

假设你的数据集已经按照比例划分成了训练集、验证集和测试集，并且每个图像都有对应的`.txt`格式的标签文件。我们创建一个`data.yaml`文件来描述这些信息：

```yaml
train: ./dataset/train/images/  # 训练集图片路径
val: ./dataset/val/images/      # 验证集图片路径
test: ./dataset/test/images/    # 测试集图片路径

nc: 2  # 类别数量
names: ['non_violence', 'violence']  # 类别名称
```

请根据实际路径替换上述`./dataset/...`部分。

### 3. 训练模型

接下来，编写Python脚本来加载YOLOv8模型并开始训练过程。这里是一个完整的例子，包括详细的注释以帮助理解每一步的作用。

#### 训练脚本 (`train.py`)

```python
from ultralytics import YOLO
import os

def train_yolov8_model():
    """
    使用YOLOv8训练暴力行为检测模型。
    """
    # 加载预训练的YOLOv8模型
    model = YOLO('yolov8n.pt')  # 可以选择其他变体如'yolov8s.pt', 'yolov8m.pt'等
    
    # 开始训练
    results = model.train(
        data='path/to/data.yaml',  # 数据集配置文件路径
        epochs=100,  # 根据需要调整epoch数
        imgsz=640,  # 输入图像尺寸
        batch=16,  # 批次大小
        name='violence_detection',  # 实验名称
        save=True,  # 自动保存最佳模型
        exist_ok=True,  # 如果目录存在则不报错
        patience=50,  # 当验证损失不再改善时提前停止训练的轮数
        lr0=0.01,  # 初始学习率
        lrf=0.1,  # 最终学习率（lr0 * lrf）
        optimizer='SGD',  # 优化器类型
        augment=True,  # 启用数据增强
        device='',  # 使用GPU或CPU ('cuda' or 'cpu')
        workers=8,  # 数据加载的工作线程数
        project='runs/detect',  # 保存结果的项目目录
        verbose=True,  # 显示详细输出
    )

if __name__ == "__main__":
    # 设置当前工作目录为脚本所在目录，以便相对路径正确
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 调用训练函数
    train_yolov8_model()
```

### 4. 评估模型

训练完成后，可以使用验证集评估模型性能：

#### 评估脚本 (`validate.py`)

```python
from ultralytics import YOLO

def validate_model():
    """
    使用验证集评估YOLOv8模型性能。
    """
    model = YOLO('runs/detect/violence_detection/weights/best.pt')  # 加载最佳模型
    metrics = model.val()  # 进行验证
    print(f"Validation mAP: {metrics.box.map}")  # 输出mAP值作为性能指标之一

if __name__ == "__main__":
    validate_model()
```

### 5. 可视化预测结果

我们可以对单张图片进行预测，并可视化其结果：

#### 预测与可视化脚本 (`predict_and_visualize.py`)

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def predict_and_visualize(image_path):
    """
    对指定图片进行预测，并显示预测结果。
    :param image_path: 图片路径
    """
    model = YOLO('runs/detect/violence_detection/weights/best.pt')  # 加载最佳模型
    results = model.predict(image_path, conf=0.5)  # 进行预测，设置置信度阈值

    for result in results:
        for box in result.boxes:
            cls, conf = box.cls.item(), box.conf.item()
            x1, y1, x2, y2 = map(int, box.xyxy)
            label = model.model.names[int(cls)]
            image = cv2.imread(image_path)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # 示例调用
    predict_and_visualize("path/to/your/image.jpg")
```

### 6. 运行步骤

- **训练模型**：
  ```bash
  python train.py
  ```
  
- **评估模型**：
  ```bash
  python validate.py
  ```
  
- **预测与可视化**：
  ```bash
  python predict_and_visualize.py
  ```

