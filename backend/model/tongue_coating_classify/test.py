import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

# 参数设置
IMG_SIZE = 224
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'tongue_coating_resnet18.pth'  # 权重文件路径
CLASS_NAMES = [
    'black_tongue_coating',
    'map_tongue_coating',
    'purple_tongue_coating',
    'red_tongue_yellow_fur_thick_greasy',
    'red_tongue_thick_greasy',
    'white_tongue_thick_greasy'
]  # 与训练时的类别顺序一致

# 定义图像预处理（与训练时一致）
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
model = torchvision.models.resnet18(pretrained=False)  # 不加载预训练权重
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)  # 修改输出层
model.load_state_dict(torch.load(MODEL_PATH))  # 加载保存的权重
model = model.to(DEVICE)
model.eval()  # 设置为评估模式

# 加载并预处理输入图片
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist!")
        return
    
    # 打开图片并转换为 RGB
    image = Image.open(image_path).convert('RGB')
    
    # 应用预处理
    image_tensor = test_transforms(image).unsqueeze(0)  # 添加批次维度 [1, C, H, W]
    image_tensor = image_tensor.to(DEVICE)
    
    # 进行预测
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)  # 计算概率
        confidence, predicted = torch.max(probabilities, 1)  # 获取最高概率及其索引
    
    # 获取预测结果
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item() * 100  # 转换为百分比
    
    # 输出结果
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence_score:.2f}%")
    
    # 可选：返回所有类别的概率
    all_probabilities = probabilities[0].cpu().numpy()
    for i, prob in enumerate(all_probabilities):
        print(f"{CLASS_NAMES[i]}: {prob * 100:.2f}%")

# 测试单张图片
image_path = '/home/ubuntu/MCM/visionDiagnosis/dataset/map_tongue_coating/map tongue coating__1.jpg'  # 替换为你的图片路径
predict_image(image_path)