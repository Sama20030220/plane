import torch
from torchvision import transforms
from PIL import Image
from model import build_model

# 设置参数
model_path = 'airplane_classifier_vgg16.pth'
image_path = 'img3.jpg'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 加载模型
def load_model(model_path, num_classes):
    model = build_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model


# 预测单张图像
def predict_image(model, image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return preds.item()


if __name__ == "__main__":
    num_classes = 3  # 请根据实际情况设置类别数量
    model = load_model(model_path, num_classes)
    prediction = predict_image(model, image_path)
    print(f"Predicted class: {prediction}")
