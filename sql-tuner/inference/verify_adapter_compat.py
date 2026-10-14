#!/usr/bin/env python3
"""Read local safetensors headers only; validate native MXFP4 attention LoRA.
Structural compatibility does not imply BnB/MXFP4 numerical equivalence.
"""
import argparse,json,struct
from pathlib import Path

def tensor_shapes(root):
    files=sorted(root.glob('*.safetensors'))
    if not files:raise ValueError(f'No safetensors in {root}')
    shapes={}
    for path in files:
        with path.open('rb') as stream:
            prefix=stream.read(8)
            if len(prefix)!=8:raise ValueError(f'Invalid safetensors: {path.name}')
            size=struct.unpack('<Q',prefix)[0]
            if size>64*1024*1024:raise ValueError('Unreasonably large safetensors header')
            header=json.loads(stream.read(size))
        for name,info in header.items():
            if name=='__metadata__':continue
            if name in shapes:raise ValueError(f'Duplicate tensor: {name}')
            shapes[name]=info['shape']
    return shapes

def token_identity(root):
    path=root/'tokenizer.json'
    if not path.exists():raise ValueError(f'{path}: required to compare token IDs; copy the actual tokenizer from the trained adapter/base')
    tokenizer=json.loads(path.read_text());vocab=tokenizer.get('model',{}).get('vocab')
    if not isinstance(vocab,dict):raise ValueError('Expected BPE tokenizer vocabulary')
    added={t['content']:t['id'] for t in tokenizer.get('added_tokens',[])}
    return vocab,added

def verify(adapter,native):
    adapter=Path(adapter);native=Path(native)
    cfg=json.loads((adapter/'adapter_config.json').read_text());base=json.loads((native/'config.json').read_text())
    if base.get('model_type')!='gpt_oss':raise ValueError('Base must be a GPT-OSS checkpoint')
    method=base.get('quantization_config',{}).get('quant_method')
    if method not in {'mxfp4','gpt_oss_mxfp4'}:raise ValueError(f'Expected native MXFP4, got quant_method={method!r}; do not point vLLM at the BnB training checkpoint')
    if cfg.get('peft_type')!='LORA' or cfg.get('use_dora') or cfg.get('modules_to_save'):raise ValueError('Only standard attention LoRA without modules_to_save/DoRA is supported by this checker')
    targets=cfg.get('target_modules')
    if not isinstance(targets,list) or not targets or not set(targets)<={'q_proj','k_proj','v_proj','o_proj'}:raise ValueError('Expected q/k/v/o attention-only LoRA')
    if cfg.get('rank_pattern') or cfg.get('alpha_pattern'):raise ValueError('Per-module rank/alpha patterns require separate validation')
    adapter_shapes=tensor_shapes(adapter);native_shapes=tensor_shapes(native);count=0;seen=set()
    for name,shape in adapter_shapes.items():
        if '.lora_A.' not in name:continue
        mate=name.replace('.lora_A.','.lora_B.')
        b=adapter_shapes.get(mate)
        if len(shape)!=2 or not b or len(b)!=2 or shape[0]!=cfg['r'] or b[1]!=cfg['r']:raise ValueError(f'Invalid A/B rank shapes: {name}')
        stem=name.split('.lora_A.')[0]
        if stem.startswith('base_model.model.'):stem=stem[len('base_model.model.'):]
        if stem.split('.')[-1] not in targets:raise ValueError(f'Unexpected adapter target: {stem}')
        native_name=stem+'.weight';expected=native_shapes.get(native_name)
        if expected!=[b[0],shape[1]]:raise ValueError(f'Native attention tensor mismatch {native_name}: native={expected}, adapter={[b[0],shape[1]]}')
        seen.add(stem.split('.')[-1]);count+=1
    if not count or seen!=set(targets):raise ValueError('Missing LoRA targets')
    if len(adapter_shapes)!=count*2:raise ValueError('Unexpected non-A/B tensors in adapter')
    layers=base.get('num_hidden_layers')
    if layers and count!=layers*len(targets):raise ValueError('Adapter does not cover every expected layer/target')
    if token_identity(adapter)!=token_identity(native):raise ValueError('Tokenizer token-ID mapping differs between adapter and native base')
    return {'status':'structurally_compatible','native_model_type':base['model_type'],'quantization':method,'adapter_training_base':cfg.get('base_model_name_or_path'),'rank':cfg['r'],'lora_module_pairs':count,'token_ids_match':True,'numerical_equivalence_verified':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',required=True);p.add_argument('--native-base',required=True);a=p.parse_args()
    print(json.dumps(verify(a.adapter,a.native_base),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
