#!/usr/bin/env python3
"""빠른 테스트 - 이미 다운로드된 모델 사용"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

print("Quick Model Test")
print("="*50)

# 1. 모델 로드 (이미 다운로드됨)
print("\n[1] Loading GPT-2...")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float32)
model.eval()

print(f"✅ Model loaded!")
print(f"   Layers: {model.config.n_layer}")
print(f"   Hidden: {model.config.n_embd}")

# 2. 간단한 테스트
print("\n[2] Testing activation extraction...")
text = "This is a beautiful story about friendship."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

print(f"✅ Extraction works!")
print(f"   Layers: {len(outputs.hidden_states)}")
print(f"   Shape: {outputs.hidden_states[0].shape}")

# 3. 마지막 토큰 활성화
last_token_acts = outputs.hidden_states[-1][0, -1, :].numpy()
print(f"\n[3] Last token activation:")
print(f"   Shape: {last_token_acts.shape}")
print(f"   Mean: {last_token_acts.mean():.4f}")
print(f"   Std: {last_token_acts.std():.4f}")

print("\n" + "="*50)
print("✅ All tests passed! Ready to extract activations.")
