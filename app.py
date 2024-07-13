import requests
import torch
from flask_sqlalchemy import SQLAlchemy
from torchvision import transforms
from PIL import Image
from werkzeug.utils import secure_filename

from AITrain.model import build_model
from flask import Flask, request, jsonify, render_template, url_for, redirect
from flask_login import LoginManager, UserMixin, login_required, logout_user, login_user, current_user
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = '123456'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/Plane_User'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static')

# 初始化Flask-Login
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# 用户类，继承自UserMixin
# 数据库模型 - 用户表
class User(UserMixin, db.Model):
    def __init__(self, username, password):
        self.username = username
        self.password = password

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)


class RecognitionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    prediction_result = db.Column(db.JSON, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())


# 设置参数
model_path = 'AITrain/airplane_classifier_vgg16(0.001).pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode == "RGBA" else img),  # 如果是RGBA则转换为RGB
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
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 应用softmax获取概率
        values, indices = torch.topk(probabilities, k=len(class_names))  # 获取所有类别的概率和索引
        probabilities = probabilities.cpu().numpy()[0]  # 将概率张量转换为numpy数组
        results = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        return results


def save_file(file, file_path):
    try:
        # 重置文件指针
        file.seek(0)

        # 手动读取文件流并写入磁盘
        with open(file_path, 'wb') as f:
            while True:
                chunk = file.read(4096)
                if not chunk:
                    break
                f.write(chunk)

        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise IOError("File appears to be empty after saving.")

        print(f"File saved successfully. Size: {file_size} bytes")
    except Exception as e:
        print(f"Error saving file: {e}")




@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# 登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))  # 修改这里
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('register.html', error="Username already exists.")
        new_user = User(username=username, password=password)  # 密码应加密存储
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# 在Flask应用中增加一个新路由
@app.route('/history')
@login_required
def history():
    histories = RecognitionHistory.query.filter_by(user_id=current_user.id).all()
    print('hhh')
    if histories:
        return jsonify([{
            'id': history.id,
            'image_path': history.image_path,
            'prediction_result': history.prediction_result,
            'timestamp': history.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        } for history in histories])
    else:
        return jsonify([])  # 返回一个空的JSON数组


# 主页路由，现在需要登录才能访问
@app.route('/')
@login_required
def home():
    histories = RecognitionHistory.query.filter_by(user_id=current_user.id).all()
    histories_data = [{
        'id': history.id,
        'image_path': history.image_path,
        'prediction_result': history.prediction_result,
        'timestamp': history.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    } for history in histories]
    return render_template('Rec_Plane.html', histories=histories_data)



@app.route('/predict', methods=['POST'])
@login_required
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

        # 检查所有类别的概率是否都低于65%
        if all(prob < 0.65 for prob in prediction.values()):
            return jsonify({"message": "这张图片不符合识别规范，请选择其他图片"})
        if not current_user.is_authenticated:
            return jsonify({"error": "User is not authenticated"}), 401
        # 保存图片到静态文件夹
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        save_file(file, file_path)
        # file.save(file_path)
        # 存储识别历史到数据库
        history = RecognitionHistory(user_id=current_user.id, image_path=file_path, prediction_result=prediction)
        db.session.add(history)
        db.session.commit()
        return jsonify(prediction)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # 创建所有定义的数据库表
    app.run(debug=True)
