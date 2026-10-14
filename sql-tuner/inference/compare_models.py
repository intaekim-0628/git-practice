#!/usr/bin/env python3
"""Compare LLM outputs on identical held-out prompts. Does not execute SQL."""
import argparse,json,os,time,urllib.request,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_outputs import score_prediction

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8000');p.add_argument('--data',default='data/test.jsonl');p.add_argument('--output',default='outputs/model_comparison.jsonl');p.add_argument('--models',nargs='+',default=['sql-base','sql-tuner']);p.add_argument('--limit',type=int);p.add_argument('--max-tokens',type=int,default=2048);a=p.parse_args()
    rows=[json.loads(s) for s in Path(a.data).read_text().splitlines() if s.strip()]
    if a.limit is not None:rows=rows[:a.limit]
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise ValueError('Output exists; choose a new filename')
    with target.open('w') as out:
        for row in rows:
            for name in a.models:
                payload={'model':name,'messages':row['messages'][:-1],'temperature':0,'stream':False,'max_tokens':a.max_tokens}
                headers={'Content-Type':'application/json'}
                if os.environ.get('SQL_TUNER_API_KEY'):headers['Authorization']='Bearer '+os.environ['SQL_TUNER_API_KEY']
                request=urllib.request.Request(a.url.rstrip('/')+'/v1/chat/completions',data=json.dumps(payload).encode(),headers=headers)
                start=time.perf_counter()
                try:
                    with urllib.request.urlopen(request,timeout=1800) as response:result=json.load(response)
                    content=result['choices'][0]['message']['content'];record={'id':row['id'],'model':name,'request_seconds':time.perf_counter()-start,'response':result,'structural_metrics':score_prediction(content,json.loads(row['messages'][-1]['content']))}
                except Exception as e:record={'id':row['id'],'model':name,'error':str(e)}
                out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush()
    print(target)
if __name__=='__main__':main()
