#!/usr/bin/env python3
"""Deterministic train-only paraphrase, formatting and evidence-ablation augmentation."""
import argparse,copy,hashlib,json,collections
from pathlib import Path
from rebuild_data import parse,fingerprint,objects,write,dump
REQUESTS=[
'제공된 근거만으로 Oracle SQL을 분석하고 관측 사실과 미확인 사항을 구분해 주세요.',
'아래 SQL의 튜닝 가능성을 평가하세요. 판단 근거가 부족하면 필요한 추가 자료를 구체적으로 요청하세요.',
'운영 적용을 검토할 수 있도록 객체 분류, DDL 가능 여부, 검증 절차를 포함해 답변하세요.',
'실행 계획과 메타데이터에 없는 정보를 추정하지 말고 SQL 튜닝 의견을 작성하세요.',
'성능 분석을 도와주세요. 실제 측정 결과와 재작성 후보의 예상 효과를 혼동하지 마세요.',
'인덱스 권고 전에 테이블 소유자와 객체 유형을 확인하고, 확인할 수 없는 항목은 명시하세요.',
'이 쿼리에 어떤 개선을 검토할 수 있을까요? 검증되지 않은 속도 향상이나 비용 수치는 제시하지 마세요.',
'아래 자료를 검토해 SQL 유지 또는 재작성 후보와 필요한 실행 검증 항목을 알려 주세요.',
'데이터베이스 튜닝 검토서를 작성하세요. 근거가 없는 인덱스나 숨겨진 파라미터 변경은 권하지 마세요.',
'Oracle SQL을 점검하고 제공된 객체 정보의 한계와 다음 진단 단계를 설명하세요.',
]
MODES=['unchanged_evidence','no_plan','no_indexes','no_ownership','sql_only']

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',default=str(Path(__file__).parent/'data'));a=p.parse_args();root=Path(a.data_dir)
 base=[json.loads(l) for l in (root/'train.jsonl').read_text().splitlines() if l.strip()]
 augmented=[];seen=set();mode_counts=collections.Counter()
 for parent in base:
  original_input=json.loads(parent['messages'][1]['content'])
  for variant in range(21):
   row=copy.deepcopy(parent)
   if variant:
    mode=MODES[(variant-1)%5];request=REQUESTS[(variant-1)//2]
    payload=copy.deepcopy(original_input);inp=payload['input']
    tree=parse(inp['sql']);inp['sql']=tree.sql(dialect='oracle',pretty=bool(variant%2),comments=True)
    if fingerprint(inp['sql'])!=fingerprint(original_input['input']['sql']):raise ValueError('SQL rendering changed AST')
    modifications=[]
    if mode in ('no_plan','sql_only'):
     inp['observed_plan']={'kind':'not_provided','is_actual':False,'operations':[],'raw':None};modifications.append('observed_plan_removed')
    if mode=='no_indexes':
     for key in ['indexes','function_based_indexes']:inp['metadata'][key]=[]
     modifications.append('index_metadata_removed_not_evidence_of_physical_absence')
    if mode=='no_ownership':
     for table in inp['metadata'].get('tables',[]):table['owner']=None
     modifications.append('table_owner_removed')
    if mode=='sql_only':
     inp['metadata']={key:[] for key in inp.get('metadata',{})};modifications.append('all_metadata_removed')
    payload['request']=request
    answer=json.loads(row['messages'][-1]['content'])
    answer['object_classification']=objects(parse(inp['sql']),inp['metadata'])
    answer['observed_plan']=inp['observed_plan']
    if mode!='unchanged_evidence':
     # No inherited rewrite recommendation for evidence intentionally withheld.
     answer['proposed_sql']=None
     answer['recommendation']='현재 제공된 근거만으로 튜닝 변경을 확정할 수 없습니다. 원본 SQL을 유지하고 소유자·객체 유형, 전체 인덱스 메타데이터와 실제 실행 통계를 수집해 재분석하세요.'
    row['messages'][1]['content']=dump(payload);row['messages'][-1]['content']=dump(answer)
    row['id']=parent['id']+f'-SYN{variant:02d}'
    row['augmentation']={'parent_id':parent['id'],'family_id':parent['group_id'],'variant':variant,'method':'request_paraphrase_sql_ast_render_evidence_ablation','evidence_mode':mode,'changes':modifications,'synthetic':True,'expert_verified':False,'sql_semantics_changed':False}
    row['target_origin']='synthetic_evidence_limited_target_not_expert_verified'
   else:mode='original'
   sig=hashlib.sha256(dump(row['messages']).encode()).hexdigest()
   if sig in seen:continue
   seen.add(sig);augmented.append(row);mode_counts[mode]+=1
 # Keep split membership and source query family unchanged.
 heldout={r['group_id'] for name in ['eval','test'] for r in map(json.loads,(root/f'{name}.jsonl').read_text().splitlines())}
 assert not heldout & {r['group_id'] for r in augmented}
 write(root/'train_augmented.jsonl',augmented)
 manifest={'base_train_records':len(base),'augmented_train_records':len(augmented),'synthetic_records':len(augmented)-len(base),'max_candidates_per_parent':21,'parent_record_counts':dict(collections.Counter(r.get('augmentation',{}).get('parent_id',r['id']) for r in augmented)),'query_families':len({r['group_id'] for r in base}),'seed':'deterministic enumeration','modes':dict(mode_counts),'heldout_unchanged':True,'eval_sha256':hashlib.sha256((root/'eval.jsonl').read_bytes()).hexdigest(),'test_sha256':hashlib.sha256((root/'test.jsonl').read_bytes()).hexdigest(),'warning':'More training presentations, not new independent queries or expert-verified tuning results. Evidence-ablation cases teach calibrated uncertainty. Original train kept separately.'}
 (root/'augmentation_audit.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
