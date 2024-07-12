from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
from model import build_model

app = Flask(__name__, static_folder='static', template_folder='templates')

# 设置参数
model_path = 'airplane_classifier_vgg16(0.001).pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# 类别映射字典
class_names = {0: '直升飞机', 1: '战斗飞机', 2: '客机'}


# 加载模型
def load_model(model_paths, num_classes):
    model = build_model(num_classes)
    model.load_state_dict(torch.load(model_paths))
    model = model.to(device)
    model.eval()
    return model


# 预测上传的图像
def predict_image(model, image_file):
    image = Image.open(image_file)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        pred_index = preds.item()
        if pred_index not in class_names:
            return "这并不是飞机，请重新选择图片检测"
    return class_names[preds.item()]


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        num_classes = 3  # 请根据实际情况设置类别数量
        model = load_model(model_path, num_classes)
        prediction = predict_image(model, file)
        return jsonify({'predicted_class': prediction})



if __name__ == "__main__":
    app.run(debug=True)
