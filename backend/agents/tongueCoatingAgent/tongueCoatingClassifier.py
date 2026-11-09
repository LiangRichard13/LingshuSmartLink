"""舌苔分类器：真实 ResNet18 实现。

torch/torchvision/PIL 全部延迟导入，仅在实例化并首次预测时加载；
fixture 模式不会导入本模块（见 replay_support.FixtureClassifier）。
权重路径、设备与 map_location 均可配置，不再硬编码。
"""

from __future__ import annotations


class TongueCoatingClassifier:
    CLASS_NAMES = [  # 与训练时的类别顺序一致
        "黑苔",
        "地图舌苔",
        "紫苔",
        "红苔黄厚腻苔",
        "红舌厚腻苔",
        "白厚腻苔",
    ]

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        map_location: str | None = None,
    ) -> None:
        import torch

        self.IMG_SIZE = 224
        self.NUM_CLASSES = 6
        self.DEVICE = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.MODEL_PATH = model_path
        self.MAP_LOCATION = map_location or str(self.DEVICE)
        self._torch = torch
        self._initialize_model()

    def _initialize_model(self) -> None:
        torch = self._torch
        import torchvision

        model = torchvision.models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, self.NUM_CLASSES)
        state = torch.load(self.MODEL_PATH, map_location=self.MAP_LOCATION, weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.DEVICE)
        self.model.eval()

    def _build_transforms(self):
        import torchvision.transforms as transforms

        return transforms.Compose(
            [
                transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict_image(self, image_path: str) -> tuple:
        torch = self._torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        image_tensor = self._build_transforms()(image).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        predicted_class = self.CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item() * 100
        return f"预测类别: {predicted_class}", f"置信度: {confidence_score:.2f}%"
