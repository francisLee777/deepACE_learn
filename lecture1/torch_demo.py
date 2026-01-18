import math
import torch
import torch.nn as nn
import torch.optim as optim

# 查看是否安装完成
print(torch.__version__)  # 打印PyTorch版本号
print(torch.cuda.is_available())  # 查看是否支持GPU加速

SEQ_LEN = 20  # 序列长度，用过去 20 个点预测下一个
TOTAL_POINTS = 1000  # 总共有 1000 个点

# 1.1生成正弦数据
x = torch.linspace(0, 100, TOTAL_POINTS)
sin_data = torch.sin(x)

# 1.2创建数据集
def create_dataset(data, seq_len):
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i + seq_len])
        ys.append(data[i + seq_len])
    return torch.stack(xs), torch.stack(ys)

X, y = create_dataset(sin_data, SEQ_LEN)

# RNN 需要的输入形状为 (batch, seq_len, input_size)
X = X.unsqueeze(-1)  # input_size = 1
y = y.unsqueeze(-1)

# 2. 定义RNN模型
class SinRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, h_n = self.rnn(x)
        # 取最后一个时间步
        last_out = out[:, -1, :]
        return self.fc(last_out)

model = SinRNN()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
EPOCHS = 300  # 训练轮数
for epoch in range(EPOCHS):
    model.train()

    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.6f}")

# 5. 保存模型
torch.save(model.state_dict(), "sin_rnn.pth")

# 6. 加载模型
loaded_model = SinRNN()
loaded_model.load_state_dict(torch.load("sin_rnn.pth"))
loaded_model.eval()

# 7. 使用上面的模型参数，进行推理预测
test_seq = sin_data[-SEQ_LEN:].unsqueeze(0).unsqueeze(-1)  # 取正弦序列最后 SEQ_LEN 个点，并把它变成 RNN 所要求的三维输入形状

with torch.no_grad():
    prediction = loaded_model(test_seq)

print("使用 1000 个数据的最后 20 个数据，来预测下一个数据的预测值，第 1001 个数据的预测值为", prediction.item())
print("第 1000 个数据的真实值为[忽略第1000和1001个数据的差距]", math.sin(x[-1].item()))  # 真实值