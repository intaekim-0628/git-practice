import json,unittest
from pathlib import Path
from train_gpt_oss_120b_qlora import read_rows,validate_splits,tokenize_row,budget_plan,analyze_epoch,training_schedule
from evaluate_outputs import score_prediction
from rebuild_data import fingerprint,parse

class CharacterTokenizer:
    eos_token_id=999
    def convert_tokens_to_ids(self,s):return 999
    def apply_chat_template(self,messages,**kw):return ''.join('<'+m['role']+'>'+m['content'] for m in messages)
    def __call__(self,text,**kw):return {'input_ids':list(range(len(text))),'offset_mapping':[(i,i+1) for i in range(len(text))]}

class Checks(unittest.TestCase):
    def test_split_and_grounding(self):
        parts=[read_rows(Path(__file__).parent/'data'/f'{s}.jsonl') for s in ['train','eval','test']]
        validate_splits(*parts)
        for rows in parts:
            for row in rows:
                a=json.loads(row['messages'][-1]['content'])
                self.assertFalse(a['after_plan']['verified']);self.assertIsNone(a['after_plan']['metrics'])
                self.assertTrue(all(not o['ddl_applicable'] for o in a['object_classification']))
                if a['proposed_sql']:parse(a['proposed_sql'])
                self.assertTrue(all(score_prediction(json.dumps(a),a).values()))
        with self.assertRaises(ValueError):validate_splits(parts[0],parts[0],parts[2])
    def test_template_families(self):
        self.assertEqual(fingerprint("select a from users where a=1",True),fingerprint("SELECT a FROM users WHERE a=2",True))
        self.assertNotEqual(fingerprint('select "Foo" from users',True),fingerprint('select "foo" from users',True))
    def test_mask_and_no_truncation(self):
        row={'id':'test','messages':[{'role':'system','content':'SECRET_SYSTEM'},{'role':'user','content':'SECRET_USER'},{'role':'assistant','content':'ANSWER'}]}
        encoded=tokenize_row(row,CharacterTokenizer(),1000)
        self.assertEqual(sum(x!=-100 for x in encoded['labels']),6)
        self.assertTrue(all(x==-100 for x in encoded['labels'][:-6]))
        with self.assertRaises(ValueError):tokenize_row(row,CharacterTokenizer(),10)
    def test_warmup_steps(self):
        self.assertEqual(training_schedule(591,8,3),{'updates_per_epoch':74,'total_optimizer_steps':222,'warmup_steps':12})
        self.assertEqual(training_schedule(1,8,1)['warmup_steps'],1)
        with self.assertRaises(ValueError):training_schedule(0,8,3)

    def test_analysis_and_budget(self):
        a=analyze_epoch({'eval_loss':2.,'train_loss':1.},{'eval_loss':1.,'train_loss':2.})
        self.assertTrue(a['possible_overfit'])
        plan=budget_plan(600,4,3);self.assertEqual(plan['selected_epochs'],3);self.assertTrue(plan['under_three_hours_at_cap'])
    def test_generation_failures(self):
        expected={'object_classification':[]}
        self.assertFalse(score_prediction('analysis<|message|>{"x":1}',expected)['json_valid'])
        a={'object_classification':[],'missing_index_detected':{},'recommendation':'','proposed_sql':'CREATE INDEX x ON SYS.T(C)','observed_plan':{},'after_plan':{'verified':True,'metrics':{'buffers':1}},'validation_required':[]}
        score=score_prediction(json.dumps(a),expected)
        self.assertFalse(score['no_unverified_schema_change']);self.assertFalse(score['no_fabricated_after_metrics'])
        raw='analysis<|message|>hidden<|start|>assistant<|channel|>final<|message|>'+json.dumps(a)+'<|fim_suffix|>'
        self.assertTrue(score_prediction(raw,expected)['json_valid'])
if __name__=='__main__':unittest.main()
