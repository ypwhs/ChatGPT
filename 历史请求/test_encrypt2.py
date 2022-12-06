import torch
from io import BytesIO
from Crypto.Cipher import AES
import secrets

# 定义模型
class SmallModel(torch.nn.Module):
  def __init__(self):
    super().__init__()

    # 定义模型的层
    self.layer1 = torch.nn.Linear(10, 20)
    self.layer2 = torch.nn.Linear(20, 30)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    return x

def encrypt_model(model, key):
  # 创建字节流
  buffer = BytesIO()

  # 将模型保存为二进制文件
  torch.save(model, buffer)

  # 读取二进制数据
  model_bytes = buffer.getvalue()

  # 使用密钥初始化 AES 加密器
  aes = AES.new(key, AES.MODE_EAX)

  # 加密模型权重并返回加密后的字节
  encrypted_model_bytes = aes.encrypt(model_bytes)

  return encrypted_model_bytes

# 生成 16 字节的随机字符串作为密钥
key = secrets.token_bytes(16)
print('key', key.hex())

# 初始化模型
model = SmallModel()

# 使用给定的密钥加密模型
encrypted_model_bytes = encrypt_model(model, key)
print('encrypted_model_bytes', encrypted_model_bytes.hex())
