"""
Pytorch 字符级 RNN 训练 Shakespeare 文本
模型结构: One-hot 输入 → RNN → Linear → Softmax
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ================================================================
# 👇 参数区域（直接修改这里即可）
# ================================================================
seq_length = 50  # 序列长度（时间步）
hidden_size = 128  # RNN 隐藏层大小
num_epochs = 10
learning_rate = 0.002
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "./shakespeare.txt"  # 需准备这个文本文件
# ================================================================


# ====================== 数据预处理 ======================
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 构建字符字典
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Total chars: {len(text)}, vocab size: {vocab_size}")

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# 将文本编码为数字
data = np.array([char_to_ix[c] for c in text], dtype=np.int64)
split = int(len(data) * 0.9)
train_data = data[:split]
valid_data = data[split:]


# ====================== Dataset 定义 ======================
class CharDataset(torch.utils.data.Dataset):
    """
    Non-overlapping chunk dataset:
    - data: 1D numpy array of token ids
    - chunk length = seq_len + 1 (we need x: seq_len, t: seq_len)
    - __len__ = total_tokens // (seq_len + 1)
    """

    def __init__(self, data, seq_len):
        self.data = np.asarray(data, dtype=np.int64)
        self.seq_len = seq_len
        chunk = seq_len + 1
        total_len = (len(self.data) // chunk) * chunk
        self.data = self.data[:total_len]
        # reshape into (num_chunks, chunk)
        self.chunks = self.data.reshape(-1, chunk)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = chunk[:self.seq_len]
        y = chunk[1:self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


train_dataset = CharDataset(train_data, seq_length)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# ====================== 模型定义 ======================
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(vocab_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h_prev=None):
        # x shape: (batch, seq_len)
        one_hot = torch.nn.functional.one_hot(x, num_classes=vocab_size).float()  # (B, T, C)
        out, h = self.rnn(one_hot, h_prev)  # (B, T, H)
        logits = self.fc(out)  # (B, T, V)
        return logits, h


model = CharRNN(vocab_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ====================== 训练循环 ======================
for epoch in range(num_epochs):
    total_loss = 0
    count = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, _ = model(x_batch)
        loss = criterion(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
        loss.backward()
        optimizer.step()
        count += 1
        total_loss += loss.item()
        print(count)

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}]  Loss: {avg_loss:.4f}")


# ====================== 文本生成函数 ======================
def generate(model, start_text="To be", length=200):
    model.eval()
    chars = [ch for ch in start_text]
    h = None
    x = torch.tensor([[char_to_ix[c] for c in chars]], dtype=torch.long).to(device)

    for _ in range(length):
        logits, h = model(x[:, -seq_length:], h)
        probs = torch.softmax(logits[0, -1], dim=0).cpu().detach().numpy()
        next_ix = np.random.choice(len(probs), p=probs)
        next_char = ix_to_char[next_ix]
        chars.append(next_char)
        x = torch.tensor([[char_to_ix[c] for c in chars[-seq_length:]]], dtype=torch.long).to(device)

    return ''.join(chars)


# ====================== 测试生成 ======================
print("\n=== Sample generation after training ===")
print(generate(model, start_text="Alas, that love, whose view is muffled still", length=500))
# 和真正文本中的内容很像
