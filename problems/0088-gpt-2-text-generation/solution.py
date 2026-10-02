import numpy as np

def layer_norm(x,g,b,eps=1e-5):
	m = np.mean(x,axis=-1,keepdims=True)
	v = np.var(x,axis=-1,keepdims=True)
	return g*(x-m)/np.sqrt(v+eps)+b

def linear(x,w,b):
	return x@w+b

def gelu(x):
	return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x,3))))
def mha(x,c_attn,c_proj,n_head):
	x = linear(x,c_attn['w'],c_attn['b'])
	qkv = np.split(x, 3,axis=-1)
	q,k,v = map(lambda t:np.array(np.split(t, n_head,axis=-1)),qkv)
	qk = np.matmul(q, np.swapaxes(k, -1, -2))/np.sqrt(q.shape[-1])
	mask = (1-np.tri(qk.shape[-1]))*-1e9
	qk = qk+mask
	exp_x = np.exp(qk-np.max(qk,axis=-1,keepdims=True))
	w = exp_x/np.sum(exp_x,axis=-1,keepdims=True)
	a = np.matmul(w, v)
	a = np.concatenate(a,axis=-1)
	return linear(a,c_proj['w'],c_proj['b'])

def mlp(x,c_fc,c_proj):
	return linear(gelu(linear(x,c_fc['w'],c_fc['b'])),c_proj['w'],c_proj['b'])

def transformer_block(x,mlp_param,attn_param,ln_1,ln_2,n_head):
	x = x+mha(layer_norm(x,**ln_1),**attn_param,n_head=n_head)
	x = x+mlp(layer_norm(x,**ln_2),**mlp_param)
	return x

def gpt2(inputs,wte,wpe,blocks,ln_f,n_head):
	x= wte[inputs]+wpe[range(len(inputs))]
	for block in blocks:
		x = transformer_block(x,block['mlp'],block['attn'],block['ln_1'],block['ln_2'],n_head=n_head)
	x= layer_norm(x,**ln_f)
	return x @ wte.T



def gen_text(prompt: str, n_tokens_to_generate: int = 40):
	# Your code here
	encoder,hparams,params = load_encoder_hparams_and_params()
	input_ids = encoder.encode(prompt)
	prompt_len = len(input_ids)

	for _ in range(n_tokens_to_generate):
		logits = gpt2(input_ids,**params,n_head=hparams['n_head'])
		next_id = int(np.argmax(logits[-1]))
		input_ids.append(next_id)
	return encoder.decode(input_ids[prompt_len:])

def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
	class DummyBPE:
		def __init__(self):
			self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

		def encode(self, text: str):
			tokens = text.strip().split()
			return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

		def decode(self, token_ids: list):
			reversed_dict = {v: k for k, v in self.encoder_dict.items()}
			return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

	hparams = {
		"n_ctx": 1024,
		"n_head": 2
	}

	params = {
		"wte": np.random.rand(3, 10),
		"wpe": np.random.rand(1024, 10),
		"blocks": [{
			"mlp": {
				"c_fc": {"w": np.random.rand(10, 20), "b": np.random.rand(20)},
				"c_proj": {"w": np.random.rand(20, 10), "b": np.random.rand(10)}
			},
			"attn": {
				"c_attn": {"w": np.random.rand(10, 30), "b": np.random.rand(30)},
				"c_proj": {"w": np.random.rand(10, 10), "b": np.random.rand(10)}
			},
			"ln_1": {"g": np.ones(10), "b": np.zeros(10)},
			"ln_2": {"g": np.ones(10), "b": np.zeros(10)},
		}],
		"ln_f": {
			"g": np.ones(10),
			"b": np.zeros(10),
		}
	}

	encoder = DummyBPE()
	return encoder, hparams, params