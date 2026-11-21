#!/usr/bin/env python3
"""초간단 테스트 - 출력 버퍼링 해결"""
import sys

def log(msg):
    """즉시 출력"""
    print(msg, flush=True)
    sys.stdout.flush()

log("="*70)
log("Starting Literary Taste Demo...")
log("="*70)

# 1. 임포트 테스트
log("\n[1] Importing libraries...")
try:
    import torch
    log("  ✅ torch")
    import pandas as pd
    log("  ✅ pandas")
    import numpy as np
    log("  ✅ numpy")
    from pathlib import Path
    log("  ✅ pathlib")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log("  ✅ transformers")
except Exception as e:
    log(f"  ❌ Import error: {e}")
    sys.exit(1)

# 2. 데이터 확인
log("\n[2] Checking data...")
DATA_PATH = Path('LiteraryTaste')
if not DATA_PATH.exists():
    log(f"  ❌ Data directory not found: {DATA_PATH}")
    sys.exit(1)

csv_file = DATA_PATH / 'preference_tasks.csv'
if not csv_file.exists():
    log(f"  ❌ CSV file not found: {csv_file}")
    sys.exit(1)

log(f"  ✅ Data found: {csv_file}")

# 3. 데이터 로드
log("\n[3] Loading data...")
try:
    df = pd.read_csv(csv_file)
    log(f"  ✅ Loaded {len(df)} rows")
    sample = df.head(3)
    log(f"  ✅ Using {len(sample)} samples")
except Exception as e:
    log(f"  ❌ Error loading data: {e}")
    sys.exit(1)

# 4. 모델 로드 (진짜 시간 걸리는 부분)
log("\n[4] Loading GPT-2 model...")
log("  (This will take 1-2 minutes on CPU...)")
log("  Please wait...")

try:
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    log("  ✅ Tokenizer loaded")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log("  Loading model... (this is the slow part)")
    model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float32)
    model.eval()
    
    log("  ✅ Model loaded!")
    log(f"     Layers: {model.config.n_layer}")
    log(f"     Hidden: {model.config.n_embd}")
    
except Exception as e:
    log(f"  ❌ Error loading model: {e}")
    import traceback
    log(traceback.format_exc())
    sys.exit(1)

# 5. 간단한 테스트
log("\n[5] Testing activation extraction...")

text = "This is a test."
log(f"  Input: {text}")

try:
    inputs = tokenizer(text, return_tensors='pt')
    log(f"  ✅ Tokenized: {inputs['input_ids'].shape}")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    log(f"  ✅ Forward pass complete")
    log(f"     Hidden states: {len(outputs.hidden_states)} layers")
    
    last_hidden = outputs.hidden_states[-1][0, -1, :].numpy()
    log(f"  ✅ Extracted activation: shape {last_hidden.shape}")
    log(f"     Mean: {last_hidden.mean():.4f}")
    
except Exception as e:
    log(f"  ❌ Error in extraction: {e}")
    import traceback
    log(traceback.format_exc())
    sys.exit(1)

log("\n" + "="*70)
log("✅ ALL TESTS PASSED!")
log("="*70)
log("\nModel is working correctly. You can now run the full analysis.")
