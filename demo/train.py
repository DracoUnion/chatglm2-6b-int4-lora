from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch
import re
import json
from typing import *
import pickle

def combine_prompt_args(prompt: str, args: Dict[str, Any]):
    return re.sub(r'{(\w+)}', lambda g: args.get(g.group(1), ''), prompt)


base_path = r'../src'
model_path = r'../src/pytorch_model.bin'
save_path = r'../src/pytorch_model_lora.bin'
lora_path = save_path if path.isfile(save_path) else None

ds = [{
    "content": "类型#裙*材质#针织*颜色#纯色*风格#复古*风格#文艺*风格#简约*图案#格子*图案#纯色*图案#复 古*裙型#背带裙*裙长#连衣裙*裙领型#半高领",
    "summary": "这款[[BRAND]]针织两件套连衣裙，简约的纯色半高领针织上衣，修饰着颈部线，尽显优雅气质。同时搭配叠穿起一条背带式的复古格纹裙，整体散发着一股怀旧的时髦魅力，很是文艺范。"
}]
ques_prompt = "请按照以下关键词生成广告：{content}"
ans_prompt = "{summary}"


self_args = {
        'lr': 5e-2,
        'grad_accum': 1,
        'n_epoch': 5,
        'save_step': 30,
        'lora_r': 32,
        'lora_alpha': 32,
        'lora_dropout_rate': 0.0,
}

conf = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
# 分别加载结构和参数
model = AutoModel.from_config(conf, trust_remote_code=True)
model.attach_lora(
    lora_r = self_args['lora_r'],
    lora_alpha = self_args['lora_alpha'],
    lora_dropout_rate = self_args['lora_dropout_rate'],
)
stdc = torch.load(model_path)
# 可能会缺少 Lora 权重，或多出来 PrefEncoder 权重，没有任何问题
model.load_state_dict(stdc, False)
if lora_path:
    lora_stdc = torch.load(lora_path)
    model.load_state_dict(lora_stdc, False)  
model = model.cuda()

# torch.autograd.set_detect_anomaly(True)
optimizer = torch.optim.SGD(model.parameters(), lr=self_args['lr'])
step = 0
for epoch in range(self_args['n_epoch']):
    for i, dsi in enumerate(ds):
        # 组装问答和问答提示词
        ques = tokenizer.build_prompt(combine_prompt_args(ques_prompt, dsi))
        ans = combine_prompt_args(ans_prompt, dsi)
        # 问答转成问答 ID
        ques_ids = tokenizer.encode(text=ques, add_special_tokens=True, truncation=True)
        ans_ids = tokenizer.encode(text=ans, add_special_tokens=False, truncation=True)
        # 问答 ID 拼接输入 ID
        input_ids = ques_ids + ans_ids + [tokenizer.eos_token_id]
        output_ids = [tokenizer.pad_token_id] * len(ques_ids) + ans_ids + [tokenizer.eos_token_id] 
        # 忽略 <PAD>
        output_ids = [(oid if oid != tokenizer.pad_token_id else -100) for oid in output_ids]
        # 因为批量大小为 1，无需填充
        optimizer.zero_grad()
        input_ids = torch.tensor([input_ids]).cuda()
        output_ids = torch.tensor([output_ids]).cuda()
        loss = model.forward(
            input_ids=input_ids, labels=output_ids, 
            return_dict=True, output_hidden_states=True
        ).loss
        loss.backward()
        print(f'epoch: {epoch}, step: {step}, ques: {json.dumps(ques, ensure_ascii=False)}, ans: {json.dumps(ans, ensure_ascii=False)}, loss: {loss}')
        # 一定步骤再更新梯度
        if step % self_args['grad_accum'] == 0:
            optimizer.step()
        # 一定步骤保存权重
        if step % self_args['save_step'] == 0:
            torch.save(model.lora_state_dict(), save_path)
        step += 1
        
optimizer.step()
torch.save(model.lora_state_dict(), save_path)