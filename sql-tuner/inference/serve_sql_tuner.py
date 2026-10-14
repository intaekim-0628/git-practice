#!/usr/bin/env python3
"""Single-worker Unsloth API: identical loaded weights with/without adapter."""
import argparse,contextlib,hmac,json,os,threading,time
from pathlib import Path

def adapter_info(path):
    root=Path(path);cfg=json.loads((root/'adapter_config.json').read_text())
    if not (root/'adapter_model.safetensors').exists():raise ValueError('adapter_model.safetensors missing')
    if not cfg.get('base_model_name_or_path'):raise ValueError('Missing adapter base model identity')
    return cfg

def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',default='outputs/sql-tuner/best_adapter');p.add_argument('--merged-model');p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000);p.add_argument('--max-seq-length',type=int,default=8192);a=p.parse_args()
    key=os.environ.get('SQL_TUNER_API_KEY','')
    if a.host not in ('127.0.0.1','localhost','::1') and not key:raise ValueError('Set SQL_TUNER_API_KEY before binding a non-loopback host')
    if not a.merged_model:print('Adapter configuration:',json.dumps(adapter_info(a.adapter),ensure_ascii=False))
    from unsloth import FastLanguageModel
    import torch
    from fastapi import FastAPI,HTTPException,Request
    from pydantic import BaseModel,Field
    import uvicorn
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from evaluate_outputs import final_content
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    model,tokenizer=FastLanguageModel.from_pretrained(model_name=a.merged_model or a.adapter,max_seq_length=a.max_seq_length,dtype=torch.bfloat16,load_in_4bit=True,full_finetuning=False)
    if not a.merged_model and not callable(getattr(model,'disable_adapter',None)):raise RuntimeError('PEFT disable_adapter unavailable: cannot provide same-base control')
    FastLanguageModel.for_inference(model);lock=threading.Lock();app=FastAPI(title='SQL Tuner development API')
    class Message(BaseModel):
        role:str
        content:str
    class Completion(BaseModel):
        model:str='sql-tuner'
        messages:list[Message]
        max_tokens:int=Field(default=2048,ge=1,le=4096)
        temperature:float=0
        stream:bool=False
    def auth(request):
        if key and not hmac.compare_digest(request.headers.get('authorization',''),'Bearer '+key):raise HTTPException(401,'Invalid API key')
    @app.get('/health')
    def health(request:Request):
        auth(request);return {'status':'ready','mode':'merged-requantized-bnb4' if a.merged_model else 'adapter','gpu':torch.cuda.get_device_name(0)}
    @app.get('/v1/models')
    def models(request:Request):
        auth(request);names=['sql-tuner'] if a.merged_model else ['sql-base','sql-tuner'];return {'object':'list','data':[{'id':n,'object':'model'} for n in names]}
    @app.post('/v1/chat/completions')
    def complete(body:Completion,request:Request):
        auth(request)
        if body.stream or body.temperature!=0:raise HTTPException(400,'Only stream=false, temperature=0 supported')
        if body.model not in (['sql-tuner'] if a.merged_model else ['sql-base','sql-tuner']):raise HTTPException(400,'Unknown model')
        messages=[{'role':m.role,'content':m.content} for m in body.messages]
        if not messages or messages[-1]['role']!='user' or any(m['role'] not in ['system','user','assistant'] for m in messages):raise HTTPException(400,'Invalid conversation')
        with lock:
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,reasoning_effort='low')
            enc=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to(model.device);n=enc['input_ids'].shape[1]
            if n+body.max_tokens>a.max_seq_length:raise HTTPException(400,'Prompt + max_tokens exceeds context')
            context=model.disable_adapter() if body.model=='sql-base' else contextlib.nullcontext()
            torch.cuda.synchronize();start=time.perf_counter()
            with context,torch.inference_mode():
                output=model.generate(**enc,max_new_tokens=body.max_tokens,do_sample=False,pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            ids=output[0,n:];raw=tokenizer.decode(ids,skip_special_tokens=False)
            truncated=len(ids)>=body.max_tokens
            try:content=final_content(raw)
            except ValueError:raise HTTPException(502,'No final channel; increase output limit/context and retry')
            return {'id':f'chatcmpl-{time.time_ns()}','object':'chat.completion','model':body.model,'choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'length' if truncated else 'stop'}],'usage':{'prompt_tokens':n,'completion_tokens':len(ids),'total_tokens':n+len(ids)},'generation_seconds':elapsed}
    uvicorn.run(app,host=a.host,port=a.port,workers=1)
if __name__=='__main__':main()
