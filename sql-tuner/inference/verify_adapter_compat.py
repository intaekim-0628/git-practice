#!/usr/bin/env python3
"""Read local safetensors headers only; validate native MXFP4 attention LoRA.
Structural compatibility does not imply BnB/MXFP4 numerical equivalence.
"""
import argparse,json,struct,re
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

def normalize_module(name):
    return name[len('base_model.model.'):] if name.startswith('base_model.model.') else name


def target_matcher(targets):
    allowed={'q_proj','k_proj','v_proj','o_proj'}
    if isinstance(targets,list) and targets and all(isinstance(t,str) for t in targets):
        normalized=[normalize_module(t) for t in targets]
        if any(t.split('.')[-1] not in allowed for t in normalized):
            raise ValueError(f'Non-attention target_modules: {targets!r}')
        return lambda stem:any(stem==t or stem.endswith('.'+t) for t in normalized)
    if isinstance(targets,str) and targets:
        try:pattern=re.compile(targets)
        except re.error as exc:raise ValueError(f'Invalid target_modules regex: {exc}') from exc
        return lambda stem:pattern.fullmatch(stem) is not None
    raise ValueError(f'target_modules must be a nonempty list or regex string, got {targets!r}')


def verify(adapter,native):
    adapter=Path(adapter);native=Path(native)
    cfg=json.loads((adapter/'adapter_config.json').read_text());base=json.loads((native/'config.json').read_text())
    if base.get('model_type')!='gpt_oss':raise ValueError('Base must be a GPT-OSS checkpoint')
    method=base.get('quantization_config',{}).get('quant_method')
    if method not in {'mxfp4','gpt_oss_mxfp4'}:raise ValueError(f'Expected native MXFP4, got quant_method={method!r}; do not point vLLM at the BnB training checkpoint')
    if cfg.get('peft_type')!='LORA' or cfg.get('use_dora') or cfg.get('modules_to_save'):raise ValueError('Only standard attention LoRA without modules_to_save/DoRA is supported by this checker')
    targets=cfg.get('target_modules')
    matches=target_matcher(targets)
    if cfg.get('rank_pattern') or cfg.get('alpha_pattern'):raise ValueError('Per-module rank/alpha patterns require separate validation')
    adapter_shapes=tensor_shapes(adapter);native_shapes=tensor_shapes(native);count=0;seen=set()
    allowed={'q_proj','k_proj','v_proj','o_proj'}
    expected_modules={name[:-7] for name in native_shapes if name.endswith('.weight') and matches(name[:-7])}
    if any(stem.split('.')[-1] not in allowed for stem in expected_modules):raise ValueError('Configured regex selects non-attention native modules')
    for name,shape in adapter_shapes.items():
        if '.lora_A.' not in name:continue
        mate=name.replace('.lora_A.','.lora_B.')
        b=adapter_shapes.get(mate)
        if len(shape)!=2 or not b or len(b)!=2 or shape[0]!=cfg['r'] or b[1]!=cfg['r']:raise ValueError(f'Invalid A/B rank shapes: {name}')
        stem=name.split('.lora_A.')[0]
        stem=normalize_module(stem)
        if stem.split('.')[-1] not in allowed or not matches(stem):raise ValueError(f'Unexpected adapter target: {stem}')
        native_name=stem+'.weight';expected=native_shapes.get(native_name)
        if expected!=[b[0],shape[1]]:raise ValueError(f'Native attention tensor mismatch {native_name}: native={expected}, adapter={[b[0],shape[1]]}')
        seen.add(stem);count+=1
    if not count or seen!=expected_modules:raise ValueError(f'Configured/native LoRA module mismatch: missing={sorted(expected_modules-seen)}, unexpected={sorted(seen-expected_modules)}')
    if len(adapter_shapes)!=count*2:raise ValueError('Unexpected non-A/B tensors in adapter')
    if token_identity(adapter)!=token_identity(native):raise ValueError('Tokenizer token-ID mapping differs between adapter and native base')
    return {'status':'structurally_compatible','native_model_type':base['model_type'],'quantization':method,'adapter_training_base':cfg.get('base_model_name_or_path'),'rank':cfg['r'],'lora_module_pairs':count,'token_ids_match':True,'numerical_equivalence_verified':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',required=True);p.add_argument('--native-base',required=True);a=p.parse_args()
    print(json.dumps(verify(a.adapter,a.native_base),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
