import torch
import torchvision
import torchvision.transforms as transforms
from tools.colorPrinter import ColorPrinter
from PIL import Image
import os

class TongueCoatingClassifier:
    def __init__(self, model_path="model/tongue_coating_classify/tongue_coating_resnet18.pth")->None:
        # 参数设置
        self.IMG_SIZE = 224 # 输入图片大小
        self.NUM_CLASSES = 6 # 类别数量
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_PATH = model_path # 权重文件路径
        self.CLASS_NAMES = [    # 与训练时的类别顺序一致
            '黑苔',
            '地图舌苔',
            '紫苔',
            '红苔黄厚腻苔',
            '红舌厚腻苔',
            '白厚腻苔'
        ]
        
        # 定义图像预处理（与训练时一致）
        self.test_transforms = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 加载模型
        self._initialize_model()
    def _initialize_model(self)->None:
        # 加载并初始化模型
        self.model = torchvision.models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.NUM_CLASSES)
        self.model.load_state_dict(torch.load(self.MODEL_PATH))
        self.model = self.model.to(self.DEVICE)
        self.model.eval()

    def predict_image(self, image_path:str)->str:
        # 检查图片路径是否存在
        if not os.path.exists(image_path):
            ColorPrinter.yellow("SystemMessage:")
            ColorPrinter.red(f"错误: 图片文件 {image_path} 不存在!")
            return f"预测类别: 无", f"置信度: 无"
        
        # 打开图片并转换为 RGB
        image = Image.open(image_path).convert('RGB')
        
        # 应用预处理
        image_tensor = self.test_transforms(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.DEVICE)

        ColorPrinter.green("TongueCoatingAgent:")
        ColorPrinter.white("正在进行舌苔病理诊断...")

        # 进行预测
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)  # 计算概率
            confidence, predicted = torch.max(probabilities, 1)  # 获取最高概率及其索引

        # 获取预测结果
        predicted_class = self.CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item() * 100

        # 输出结果
        ColorPrinter.white(f"预测类别: {predicted_class}")
        ColorPrinter.white(f"置信度: {confidence_score:.2f}%")

        return f"预测类别: {predicted_class}", f"置信度: {confidence_score:.2f}%"