import json,struct,tempfile,unittest,subprocess,os
from pathlib import Path
from verify_adapter_compat import verify

def header(path,tensors):
 content=json.dumps({n:{'dtype':'BF16','shape':s,'data_offsets':[0,0]} for n,s in tensors.items()}).encode()
 path.write_bytes(struct.pack('<Q',len(content))+content)

class VllmConfigTest(unittest.TestCase):
 def fixture(self,root):
  adapter=root/'adapter';base=root/'base';adapter.mkdir();base.mkdir()
  (adapter/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','r':8,'target_modules':['q_proj'],'base_model_name_or_path':'bnb-training'}))
  (base/'config.json').write_text(json.dumps({'model_type':'gpt_oss','quantization_config':{'quant_method':'mxfp4'},'num_hidden_layers':1}))
  tok={'model':{'vocab':{'a':0}},'added_tokens':[{'content':'<eos>','id':1}]}
  for d in [adapter,base]:(d/'tokenizer.json').write_text(json.dumps(tok))
  stem='model.layers.0.self_attn.q_proj'
  header(adapter/'adapter_model.safetensors',{'base_model.model.'+stem+'.lora_A.weight':[8,16],'base_model.model.'+stem+'.lora_B.weight':[32,8]})
  header(base/'model.safetensors',{stem+'.weight':[32,16]});return adapter,base
 def test_dry_run_and_names(self):
  with tempfile.TemporaryDirectory() as temp:
   adapter,base=self.fixture(Path(temp));self.assertEqual(verify(adapter,base)['status'],'structurally_compatible')
   env={**os.environ,'BASE_MODEL':str(base),'LORA_PATH':str(adapter),'DRY_RUN':'1','BASE_SERVED_MODEL':'base-check','SERVED_MODEL':'tuned-check'}
   result=subprocess.run(['bash',str(Path(__file__).parent/'serve_vllm_lora.sh')],env=env,text=True,capture_output=True)
   self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--served-model-name base-check',result.stdout);self.assertIn('tuned-check=',result.stdout);self.assertNotIn('sherpa-yak',result.stdout)
   self.assertIn('--max-model-len 8192',result.stdout)
 def test_expanded_and_regex_targets(self):
  for targets in [['model.layers.0.self_attn.q_proj'],['base_model.model.model.layers.0.self_attn.q_proj'],r'.*\.self_attn\.q_proj']:
   with self.subTest(targets=targets),tempfile.TemporaryDirectory() as temp:
    adapter,base=self.fixture(Path(temp));path=adapter/'adapter_config.json';cfg=json.loads(path.read_text());cfg['target_modules']=targets;path.write_text(json.dumps(cfg))
    self.assertEqual(verify(adapter,base)['lora_module_pairs'],1)
 def test_forbidden_or_missing_targets(self):
  for targets in [['gate_proj'],r'.*\.self_attn\.k_proj']:
   with self.subTest(targets=targets),tempfile.TemporaryDirectory() as temp:
    adapter,base=self.fixture(Path(temp));path=adapter/'adapter_config.json';cfg=json.loads(path.read_text());cfg['target_modules']=targets;path.write_text(json.dumps(cfg))
    with self.assertRaises(ValueError):verify(adapter,base)
 def test_shape_and_token_mismatch(self):
  with tempfile.TemporaryDirectory() as temp:
   adapter,base=self.fixture(Path(temp));header(base/'model.safetensors',{'model.layers.0.self_attn.q_proj.weight':[64,16]})
   with self.assertRaisesRegex(ValueError,'tensor mismatch'):verify(adapter,base)
  with tempfile.TemporaryDirectory() as temp:
   adapter,base=self.fixture(Path(temp));(base/'tokenizer.json').write_text(json.dumps({'model':{'vocab':{'a':5}},'added_tokens':[]}))
   with self.assertRaisesRegex(ValueError,'token-ID'):verify(adapter,base)
if __name__=='__main__':unittest.main()
