#!/usr/bin/env python3
"""Single H200 attention-only QLoRA with assistant-only loss and epoch reports.
Runtime imports are deferred: --validate-only works without CUDA dependencies.
"""
import argparse, importlib.metadata, json, math, os, time
from pathlib import Path


def read_rows(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows: raise ValueError(f'Empty data: {path}')
    for r in rows:
        ms=r.get('messages',[])
        if [m.get('role') for m in ms]!=['system','user','assistant'] or any(not isinstance(m.get('content'),str) or not m['content'].strip() for m in ms):
            raise ValueError(f'Invalid messages: {r.get("id")}')
        if not r.get('group_id'):raise ValueError('Missing query family')
        json.loads(ms[-1]['content'])
    return rows


def validate_splits(train,ev,test):
    parts={'train':train,'eval':ev,'test':test}
    for a in parts:
        for b in parts:
            if a>=b:continue
            overlap={r['group_id'] for r in parts[a]} & {r['group_id'] for r in parts[b]}
            if overlap:raise ValueError(f'Query-family leakage between {a}/{b}: {overlap}')
    return {k:{'rows':len(v),'families':len({r['group_id'] for r in v})} for k,v in parts.items()}


def tokenize_row(row,tokenizer,max_length):
    msgs=row['messages']; answer=msgs[-1]['content']
    text=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False)
    if text.count(answer)!=1:raise ValueError(f'Assistant content not uniquely preserved by template: {row["id"]}')
    begin=text.index(answer);end=begin+len(answer)
    enc=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    offsets=enc.pop('offset_mapping');ids=enc['input_ids']
    if len(ids)>max_length:raise ValueError(f'{row["id"]}: {len(ids)} tokens > {max_length}; increase sequence length or curate this row; no silent truncation')
    # Mask system/user/control-prefix tokens. Predict only final-answer content.
    labels=[tid if start>=begin and stop<=end and stop>start else -100 for tid,(start,stop) in zip(ids,offsets)]
    # Also teach the genuine Harmony end-of-turn token where supplied.
    eos_ids={tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|fim_suffix|>')} - {None, getattr(tokenizer,'unk_token_id',None)}
    last_answer_index=max(i for i,x in enumerate(labels) if x!=-100)
    for i,(start,stop) in enumerate(offsets):
        if i>last_answer_index and ids[i] in eos_ids: labels[i]=ids[i]
    if not any(x!=-100 for x in labels):raise ValueError(f'Zero supervised tokens: {row["id"]}')
    return {'input_ids':ids,'attention_mask':enc.get('attention_mask',[1]*len(ids)),'labels':labels}


def budget_plan(epoch_seconds,budget_hours,max_epochs):
    if epoch_seconds<=0:raise ValueError('Invalid epoch duration')
    # Reserve 15% for remaining checkpoint I/O and final test; epoch 1 is included.
    proposed=max(1,int(budget_hours*3600*.85/epoch_seconds))
    chosen=min(max_epochs,proposed)
    return {'measured_epoch_seconds':epoch_seconds,'budget_hours':budget_hours,'estimated_epochs_for_budget':proposed,'selected_epochs':chosen,'estimated_training_hours':chosen*epoch_seconds/3600,'under_three_hours_at_cap':chosen*epoch_seconds<3*3600}


def analyze_epoch(current,previous=None):
    loss=current['eval_loss']; prev=previous['eval_loss'] if previous else None
    tl=current.get('train_loss');pt=previous.get('train_loss') if previous else None
    return {**current,'eval_loss_delta':None if prev is None else loss-prev,'relative_eval_improvement':None if prev in (None,0) else (prev-loss)/abs(prev),'perplexity':math.exp(min(loss,80)),'possible_overfit':bool(prev is not None and loss>prev and tl is not None and pt is not None and tl<pt),'interpretation':'Teacher-forced answer loss; does not establish SQL equivalence or measured database speedup.'}


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='unsloth/gpt-oss-120b-unsloth-bnb-4bit')
    p.add_argument('--train-file',default='data/train_augmented.jsonl');p.add_argument('--eval-file',default='data/eval.jsonl');p.add_argument('--test-file',default='data/test.jsonl')
    p.add_argument('--output-dir',default='outputs/sql-tuner');p.add_argument('--max-seq-length',type=int,default=8192)
    p.add_argument('--budget-hours',type=float,default=4.0);p.add_argument('--max-epochs',type=int,default=3)
    p.add_argument('--learning-rate',type=float,default=5e-5);p.add_argument('--grad-accum',type=int,default=8)
    p.add_argument('--lora-r',type=int,default=8);p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--generation-eval-samples',type=int,default=2);p.add_argument('--max-new-tokens',type=int,default=2048)
    p.add_argument('--resume-from-checkpoint');p.add_argument('--validate-only',action='store_true')
    args=p.parse_args()
    if not 3<=args.budget_hours<=5:p.error('--budget-hours must be 3..5')
    if min(args.max_epochs,args.grad_accum,args.max_seq_length,args.lora_r,args.max_new_tokens)<1:p.error('Positive numeric settings required')
    if args.generation_eval_samples<0:p.error('Negative generation sample count')
    return args


def main():
    args=parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    train_rows=read_rows(args.train_file);eval_rows=read_rows(args.eval_file);test_rows=read_rows(args.test_file)
    print(json.dumps(validate_splits(train_rows,eval_rows,test_rows),indent=2))
    if args.validate_only:return
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    # Must precede Transformers and TRL import.
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, set_seed
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one CUDA GPU is required; select single H200 141GB')
    props=torch.cuda.get_device_properties(0)
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 required')
    if props.total_memory/2**30<120:raise RuntimeError('This 120B preset requires H200-class VRAM; no CPU offload fallback configured')
    if args.resume_from_checkpoint:
        saved=Path(args.resume_from_checkpoint)/'trainer_state.json'
        if not saved.exists():raise ValueError('Resume requires a full Trainer checkpoint, not an adapter-only directory')
    set_seed(args.seed)
    versions={k:importlib.metadata.version(k) for k in ['torch','transformers','unsloth','peft','bitsandbytes','datasets','accelerate']}
    (out/'run_config.json').write_text(json.dumps({'args':vars(args),'versions':versions,'gpu':props.name,'vram_gib':props.total_memory/2**30},indent=2))
    model,tokenizer=FastLanguageModel.from_pretrained(model_name=args.model,max_seq_length=args.max_seq_length,dtype=torch.bfloat16,load_in_4bit=True,full_finetuning=False)
    if not tokenizer.is_fast:raise RuntimeError('A fast tokenizer with offset mapping is required for assistant-only masking')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    model=FastLanguageModel.get_peft_model(model,r=args.lora_r,target_modules=['q_proj','k_proj','v_proj','o_proj'],lora_alpha=args.lora_r*2,lora_dropout=0.0,bias='none',use_gradient_checkpointing='unsloth',random_state=args.seed)
    model.print_trainable_parameters()
    datasets={k:Dataset.from_list([tokenize_row(r,tokenizer,args.max_seq_length) for r in rows]) for k,rows in [('train',train_rows),('eval',eval_rows),('test',test_rows)]}
    stats={k:{'samples':len(ds),'tokens':sum(len(x['input_ids']) for x in ds),'supervised_tokens':sum(sum(t!=-100 for t in x['labels']) for x in ds),'max_length':max(len(x['input_ids']) for x in ds)} for k,ds in datasets.items()}
    (out/'token_statistics.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))

    def collate(features):
        length=max(len(f['input_ids']) for f in features)
        return {k:torch.tensor([f[k]+[pad]*(length-len(f[k])) for f in features],dtype=torch.long) for k,pad in [('input_ids',tokenizer.pad_token_id),('attention_mask',0),('labels',-100)]}

    def generate_samples(rows,tag):
        from evaluate_outputs import score_prediction
        predictions=[];FastLanguageModel.for_inference(model)
        try:
            for row in rows:
                prompt=tokenizer.apply_chat_template(row['messages'][:-1],tokenize=False,add_generation_prompt=True,reasoning_effort='low')
                enc=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to(model.device)
                if enc['input_ids'].shape[1]+args.max_new_tokens>args.max_seq_length:
                    raise ValueError(f'{row["id"]}: prompt + generation exceeds context; adjust sequence/generation limits')
                with torch.inference_mode():generated=model.generate(**enc,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tokenizer.pad_token_id)
                # Retain Harmony markers to separate analysis and final channels reliably.
                raw=tokenizer.decode(generated[0,enc['input_ids'].shape[1]:],skip_special_tokens=False)
                predictions.append({'id':row['id'],'raw_generation':raw,'metrics':score_prediction(raw,json.loads(row['messages'][-1]['content']))})
        finally:FastLanguageModel.for_training(model)
        (out/f'{tag}_predictions.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in predictions))
        keys=['json_valid','schema_complete','classification_exact','no_unverified_schema_change','no_fabricated_after_metrics']
        return {k:sum(p['metrics'][k] for p in predictions)/len(predictions) for k in keys} if predictions else {}

    class EpochReport(TrainerCallback):
        def __init__(self):
            self.started=None;self.epoch_start=None;self.train_losses=[];self.reports=[];self.target_epochs=args.max_epochs
            if args.resume_from_checkpoint and (out/'epoch_analysis.json').exists():self.reports=json.loads((out/'epoch_analysis.json').read_text())
        def on_train_begin(self,args_,state,control,**kw):self.started=time.monotonic()
        def on_epoch_begin(self,args_,state,control,**kw):self.epoch_start=time.monotonic();self.train_losses=[]
        def on_log(self,args_,state,control,logs=None,**kw):
            if logs and 'loss' in logs:self.train_losses.append(float(logs['loss']))
        def on_step_end(self,args_,state,control,**kw):
            # Wall-time budget starts at Trainer train(), excludes model download/load.
            if time.monotonic()-self.started>=args.budget_hours*3600:
                control.should_training_stop=True;control.should_evaluate=True;control.should_save=True
        def on_evaluate(self,args_,state,control,metrics=None,**kw):
            if self.epoch_start is None or not metrics or 'eval_loss' not in metrics:return
            duration=time.monotonic()-self.epoch_start
            current={'epoch':float(state.epoch or 0),'global_step':state.global_step,'eval_loss':float(metrics['eval_loss']),'train_loss':sum(self.train_losses)/len(self.train_losses) if self.train_losses else None,'epoch_train_eval_seconds':duration,'elapsed_train_hours':(time.monotonic()-self.started)/3600,'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30,'completed_epoch':abs((state.epoch or 0)-round(state.epoch or 0))<1e-6}
            if args.generation_eval_samples:
                current['generation_metrics']=generate_samples(eval_rows[:args.generation_eval_samples],f'epoch_{state.global_step}')
                current['generation_seconds']=time.monotonic()-self.epoch_start-duration
            if not self.reports and current['completed_epoch']:
                plan=budget_plan(duration+current.get('generation_seconds',0),args.budget_hours,args.max_epochs)
                # Keep scheduler's configured horizon stable; calibration can stop earlier.
                self.target_epochs=plan['selected_epochs']
                (out/'time_plan.json').write_text(json.dumps(plan,indent=2))
            report=analyze_epoch(current,self.reports[-1] if self.reports else None);self.reports.append(report)
            (out/'epoch_analysis.json').write_text(json.dumps(self.reports,ensure_ascii=False,indent=2))
            print('[EPOCH ANALYSIS]',json.dumps(report,ensure_ascii=False))
            if (state.epoch or 0)>=self.target_epochs:control.should_training_stop=True

    callback=EpochReport()
    cfg=TrainingArguments(output_dir=str(out/'checkpoints'),num_train_epochs=args.max_epochs,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=args.grad_accum,learning_rate=args.learning_rate,warmup_ratio=.05,lr_scheduler_type='cosine',optim='adamw_8bit',weight_decay=.01,bf16=True,fp16=False,gradient_checkpointing=True,logging_steps=1,eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,prediction_loss_only=True,report_to='none',seed=args.seed,data_seed=args.seed,dataloader_num_workers=0)
    trainer=Trainer(model=model,args=cfg,train_dataset=datasets['train'],eval_dataset=datasets['eval'],data_collator=collate,processing_class=tokenizer,callbacks=[callback,EarlyStoppingCallback(early_stopping_patience=1,early_stopping_threshold=.001)])
    # Baseline BEFORE updates enables measuring actual loss improvement.
    baseline=trainer.evaluate(metric_key_prefix='baseline')
    (out/'baseline_metrics.json').write_text(json.dumps(baseline,indent=2))
    result=trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state();trainer.save_metrics('train',result.metrics)
    adapter=out/'best_adapter';model.save_pretrained(str(adapter));tokenizer.save_pretrained(str(adapter))
    # Test is used once, after selecting best checkpoint using eval only.
    callback.epoch_start=None
    final=trainer.evaluate(datasets['test'],metric_key_prefix='test');trainer.save_metrics('test',final)
    if args.generation_eval_samples:generate_samples(test_rows,'final_test')
    (out/'summary.json').write_text(json.dumps({'best_checkpoint':trainer.state.best_model_checkpoint,'test_metrics':final,'actual_train_hours':result.metrics.get('train_runtime',0)/3600,'budget_is_cap_not_guaranteed_duration':True,'oracle_execution_validated':False},indent=2))
    from analyze_training import render
    render(out)
    print(f'Best adapter: {adapter.resolve()}')

if __name__=='__main__':main()
