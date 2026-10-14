"""API-contract tests with an explicit fake GPU/model; not an inference benchmark."""
import contextlib,importlib.util,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import serve_sql_tuner

class ServingTest(unittest.TestCase):
 def test_same_base_adapter_and_auth(self):
  from fastapi.testclient import TestClient
  class Model:
   device='fake';enabled=True
   @contextlib.contextmanager
   def disable_adapter(self):
    self.enabled=False
    try:yield
    finally:self.enabled=True
   def generate(self,**kw):
    enabled=self.enabled
    class Output:
     def __getitem__(self,index):return [int(enabled)]
    return Output()
  class Encoding(dict):
   def to(self,device):return self
  class Tokenizer:
   pad_token_id=0;eos_token_id=1
   def apply_chat_template(self,*a,**kw):return 'prompt'
   def __call__(self,*a,**kw):return Encoding(input_ids=types.SimpleNamespace(shape=(1,5)))
   def decode(self,ids,**kw):return '<|channel|>final<|message|>'+json.dumps({'adapter_enabled':bool(ids[0])})+'<|fim_suffix|>'
  model=Model();captured={}
  fake_fast=types.SimpleNamespace(from_pretrained=lambda **kw:(model,Tokenizer()),for_inference=lambda m:None)
  fake_torch=types.SimpleNamespace(bfloat16='bf16',inference_mode=contextlib.nullcontext,cuda=types.SimpleNamespace(is_available=lambda:True,get_device_name=lambda n:'FAKE GPU',synchronize=lambda:None))
  with tempfile.TemporaryDirectory() as tmp:
   Path(tmp,'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'test-base'}));Path(tmp,'adapter_model.safetensors').touch()
   with patch.dict(sys.modules,{'unsloth':types.SimpleNamespace(FastLanguageModel=fake_fast),'torch':fake_torch}),patch.dict(os.environ,{'SQL_TUNER_API_KEY':'test-only-key'}),patch.object(sys,'argv',['serve','--adapter',tmp]),patch('uvicorn.run',side_effect=lambda app,**kw:captured.update(app=app)):
    serve_sql_tuner.main()
   client=TestClient(captured['app']);headers={'Authorization':'Bearer test-only-key'}
   self.assertEqual(client.get('/health').status_code,401)
   self.assertEqual(client.get('/health',headers=headers).status_code,200)
   for name,expected in [('sql-base',False),('sql-tuner',True)]:
    response=client.post('/v1/chat/completions',headers=headers,json={'model':name,'messages':[{'role':'user','content':'test'}]})
    self.assertEqual(response.status_code,200,response.text)
    content=json.loads(response.json()['choices'][0]['message']['content']);self.assertEqual(content['adapter_enabled'],expected)
    self.assertTrue(model.enabled)
   self.assertEqual(client.post('/v1/chat/completions',headers=headers,json={'model':'bad','messages':[{'role':'user','content':'test'}]}).status_code,400)
if __name__=='__main__':unittest.main()
