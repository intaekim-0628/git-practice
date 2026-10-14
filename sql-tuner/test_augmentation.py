import json,hashlib,unittest
from pathlib import Path
from train_gpt_oss_120b_qlora import read_rows,validate_splits
from rebuild_data import fingerprint,objects,parse
from evaluate_outputs import score_prediction

class AugmentationChecks(unittest.TestCase):
 def test_augmented_evidence_and_split(self):
  root=Path(__file__).parent/'data';base={r['id']:r for r in read_rows(root/'train.jsonl')}
  rows=read_rows(root/'train_augmented.jsonl');validate_splits(rows,read_rows(root/'eval.jsonl'),read_rows(root/'test.jsonl'))
  signatures=set()
  for r in rows:
   signature=json.dumps(r['messages'],sort_keys=True)
   self.assertNotIn(signature,signatures);signatures.add(signature)
   if not r.get('augmentation'):continue
   aug=r['augmentation'];parent=base[aug['parent_id']]
   inp=json.loads(r['messages'][1]['content'])['input'];old=json.loads(parent['messages'][1]['content'])['input'];answer=json.loads(r['messages'][-1]['content'])
   self.assertEqual(r['group_id'],parent['group_id'])
   self.assertEqual(fingerprint(inp['sql']),fingerprint(old['sql']))
   self.assertEqual(answer['observed_plan'],inp['observed_plan'])
   self.assertEqual(answer['object_classification'],objects(parse(inp['sql']),inp['metadata']))
   self.assertTrue(all(score_prediction(json.dumps(answer),answer).values()))
   if aug['evidence_mode']!='unchanged_evidence':self.assertIsNone(answer['proposed_sql'])
   if aug['evidence_mode'] in ['no_plan','sql_only']:self.assertEqual(inp['observed_plan']['kind'],'not_provided')
   if aug['evidence_mode']=='sql_only':self.assertTrue(all(not v for v in inp['metadata'].values()))
  self.assertTrue(set(base)<=set(r['id'] for r in rows))
 def test_heldout_checksums(self):
  root=Path(__file__).parent/'data';audit=json.loads((root/'augmentation_audit.json').read_text())
  for name in ['eval','test']:self.assertEqual(hashlib.sha256((root/f'{name}.jsonl').read_bytes()).hexdigest(),audit[f'{name}_sha256'])
if __name__=='__main__':unittest.main()
