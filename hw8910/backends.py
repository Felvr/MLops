"""A shared generate interface for synthetic and real local inference."""
import time


class StubBackend:
    def __init__(self, manifest, root=None):
        self.loaded_id = manifest['model_id']
        self.runtime_sha256 = manifest['runtime_sha256']
        self.calls = 0

    def warmup(self):
        pass

    def validate_prompt(self, prompt):
        pass

    def generate(self, prompt, limit):
        self.calls += 1
        time.sleep(.01)
        return {'answer': f'STUB ONLY: {self.loaded_id}', 'output_tokens': 1}


class TransformersBackend:
    def __init__(self, manifest, root):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from release import verify_files
        self.loaded_id = manifest['model_id']
        self.runtime_sha256 = manifest['runtime_sha256']
        self.calls = 0
        self.runtime = manifest['runtime']
        self.torch = torch
        torch.set_num_threads(4)
        torch.manual_seed(self.runtime['generation']['seed'])
        verify_files(manifest, root)
        base = (root / manifest["artifact_uri"]).resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(root / manifest.get('tokenizer_uri',manifest['artifact_uri']), local_files_only=True)
        from release import digest
        if digest(self.tokenizer.chat_template.encode()) != self.runtime['template_sha256']:
            raise ValueError('Tokenizer chat template mismatch')
        self.model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
                          torch_dtype=getattr(torch,self.runtime['dtype'])).to(self.runtime['device']).eval()
        adapter_uri = manifest.get('adapter_uri')
        if adapter_uri:
            from peft import PeftModel
            adapter = (root / adapter_uri).resolve()
            self.model = PeftModel.from_pretrained(self.model, adapter).eval()

    def batch(self, prompt):
        g = self.runtime['generation']
        text = self.tokenizer.apply_chat_template(
            [{'role':'system','content':g['system']},{'role':'user','content':prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=g['enable_thinking'])
        return self.tokenizer(text, return_tensors='pt').to(self.runtime['device'])

    def validate_prompt(self, prompt):
        if self.batch(prompt)['input_ids'].shape[1] > self.runtime['max_input_tokens']:
            raise ValueError('Input exceeds token limit')

    def warmup(self):
        self.generate('Read the excerpt: example. Options: A. example B. other',
                      self.runtime['generation']['max_new_tokens'])
        self.calls = 0

    def generate(self, prompt, limit):
        self.calls += 1
        batch = self.batch(prompt)
        with self.torch.inference_mode():
            out = self.model.generate(**batch, max_new_tokens=limit, do_sample=False,
                         pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        tokens = out[0,batch['input_ids'].shape[1]:]
        return {'answer':self.tokenizer.decode(tokens,skip_special_tokens=True),'output_tokens':len(tokens)}
