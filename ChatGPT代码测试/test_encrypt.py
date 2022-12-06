import torch
import torch.nn as nn

import hashlib
from io import BytesIO
from Crypto.Cipher import AES

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


model = nn.Linear(4, 4)
key = 'ypw'
key = hashlib.sha256(key.encode()).digest()
print('key', key.hex())

# 使用给定的密钥加密模型
encrypted_model_bytes = encrypt_model(model, key)

print('encrypted_model_bytes', encrypted_model_bytes.hex())
