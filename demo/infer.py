import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from os import path

base_path = r'../src'
model_path = r'../src/pytorch_model.bin'
lora_path = r'../src/pytorch_model_lora.bin'
if not path.isfile(lora_path): lora_path = None

ques = '你好'

conf = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
# 分别加载结构和参数
model = AutoModel.from_config(conf, trust_remote_code=True)
stdc = torch.load(model_path)
# 可能会缺少 Lora 权重，或多出来 PrefEncoder 权重，没有任何问题
model.load_state_dict(stdc, False)
if lora_path:
    model.attach_lora()
    lora_stdc = torch.load(lora_path)
    model.load_state_dict(lora_stdc, False)
model = model.cuda()

ans = model.chat(tokenizer, ques)
print(ans)