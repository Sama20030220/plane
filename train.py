import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import time
from model import build_model
from data_loader import load_data


data_dir = 'data'
batch_size = 32
num_epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = 'airplane_classifier_vgg16.pth'  # 已保存模型的路径

# 检查是否有可用的GPU
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a GPU to run.")

# 加载数据
train_loader, num_classes = load_data(data_dir, batch_size)

# 构建模型
model = build_model(num_classes)
model = model.to(device)

# 加载已保存的模型权重
# if model_paths:
#     model.load_state_dict(torch.load(model_paths))
#     print("Loaded model weights from", model_paths)

# 设置优化器和损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# 训练模型
def train_model(model, criterion, optimizer,  num_epochs=25):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # 训练模式

        running_loss = 0.0
        running_corrects = 0

        # 遍历数据
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 反向传播 + 优化
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # 保存模型
    torch.save(model.state_dict(), 'airplane_classifier_vgg16.pth')

    return model


model = train_model(model, criterion, optimizer, num_epochs=num_epochs)
