#!/usr/bin/env python3
"""Optional BF16 export. GPU execution and output equivalence not prevalidated."""
import argparse,json,shutil
from pathlib import Path
from serve_sql_tuner import adapter_info

def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',default='outputs/sql-tuner/best_adapter');p.add_argument('--output',required=True);a=p.parse_args();cfg=adapter_info(a.adapter)
    target=Path(a.output)
    if target.exists() and any(target.iterdir()):raise ValueError('Output directory must be absent or empty')
    ancestor=target.resolve().parent
    while not ancestor.exists():ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free<300*2**30:raise RuntimeError('Reserve at least 300 GiB free disk for 120B BF16 export; extra temp/base downloads may need more')
    allowed={'q_proj','k_proj','v_proj','o_proj'}
    if not isinstance(cfg.get('target_modules'),list) or not set(cfg['target_modules'])<=allowed:raise ValueError('This export script is limited to attention-only LoRA')
    from unsloth import FastLanguageModel
    import torch
    model,tokenizer=FastLanguageModel.from_pretrained(model_name=a.adapter,max_seq_length=4096,dtype=torch.bfloat16,load_in_4bit=True,full_finetuning=False)
    if not callable(getattr(model,'save_pretrained_merged',None)):raise RuntimeError('Installed Unsloth has no supported merged export')
    model.save_pretrained_merged(str(target),tokenizer,save_method='merged_16bit',safe_serialization=True,maximum_memory_usage=0.65)
    config=target/'config.json'
    weights=list(target.glob('*.safetensors'))
    if not config.exists() or not weights:raise RuntimeError('Export incomplete: config/weights missing')
    (target/'export_review.json').write_text(json.dumps({'adapter':str(Path(a.adapter).resolve()),'base_model':cfg['base_model_name_or_path'],'method':'merged_16bit','tensor_files':len(weights),'numerical_equivalence_verified':False,'single_h200_full_bf16_inference_supported':False},indent=2))
    print('Export saved; validate adapter-vs-merged outputs before deployment:',target)
if __name__=='__main__':main()
