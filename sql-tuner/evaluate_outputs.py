"""Offline structural checks; no Oracle runtime correctness claim."""
import json,re
import sqlglot
from sqlglot import exp
REQUIRED={'object_classification','missing_index_detected','recommendation','proposed_sql','observed_plan','after_plan','validation_required'}

def final_content(raw):
    marker='<|channel|>final<|message|>'
    if marker in raw:raw=raw.rsplit(marker,1)[1]
    elif '<|channel|>analysis' in raw or raw.startswith('analysis<|message|>'):
        raise ValueError('No final channel; reasoning-only or truncated generation')
    for end in ['<|fim_suffix|>','<|im_end|>','<|return|>']:
        raw=raw.split(end,1)[0]
    raw=raw.strip()
    if raw.startswith('final<|message|>'):raw=raw[len('final<|message|>'):]
    if raw.startswith('```json') and raw.endswith('```'):raw=raw[7:-3].strip()
    return raw

def score_prediction(raw,expected):
    metrics={k:False for k in ['json_valid','schema_complete','classification_exact','no_unverified_schema_change','no_fabricated_after_metrics']}
    try:
        predicted=json.loads(final_content(raw))
        if not isinstance(predicted,dict):return metrics
    except (ValueError,TypeError):return metrics
    metrics['json_valid']=True;metrics['schema_complete']=REQUIRED<=predicted.keys()
    try:
        def signature(obj):return sorted((str(o['object']).upper(),o['classification'],o['ddl_applicable']) for o in obj)
        metrics['classification_exact']=signature(predicted['object_classification'])==signature(expected['object_classification'])
    except (KeyError,TypeError):pass
    candidate=predicted.get('proposed_sql')
    if candidate is None:metrics['no_unverified_schema_change']=metrics['schema_complete']
    elif isinstance(candidate,str):
        try:
            if re.search(r'\bDBMS_(?:STATS|SQL|SCHEDULER)\b',candidate,re.I):raise ValueError('Unverified package operation')
            trees=[t for t in sqlglot.parse(candidate,read='oracle') if t is not None]
            metrics['no_unverified_schema_change']=len(trees)==1 and isinstance(trees[0],exp.Query) and not list(trees[0].find_all(exp.Hint))
        except Exception:pass
    after=predicted.get('after_plan',{})
    if isinstance(after,dict):
        metrics['no_fabricated_after_metrics']=after.get('verified') is False and after.get('metrics') is None and after.get('operations') is None and not any(after.get(k) is not None for k in ['cost','rows','buffers','elapsed_time'])
    return metrics
