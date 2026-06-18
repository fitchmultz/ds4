import numpy as np
# --- GLM-5.2 MLA reference (tiny synthetic dims, same STRUCTURE as the real model) ---
# Real: hidden 6144, q_lora 2048, kv_lora 512, qk_nope 192, qk_rope 64, v 256, 64 heads.
# Tiny:  hidden H=16, q_lora=8, kv_lora=6, nope=3, rope=2, v=4, nhead=4 (rope+nope=5=head_dim_q)
rng = np.random.default_rng(0)
H, nhead = 16, 4
q_lora, kv_lora, nope, rope, vdim = 8, 6, 3, 2, 4
assert nope+rope == 5
def rmsnorm(x, w, eps=1e-5):
    return x * w / np.sqrt((x**2).mean(-1, keepdims=True) + eps)
def rope_interleaved(x, base, t):  # x: [..., d] d even, interleaved convention
    d = x.shape[-1]; half = d//2
    inv = 1.0/(base**(np.arange(0,1,1.0/half)/ (d//2))) if False else (1.0/base)**(np.arange(half)/half*2/d)
    # Use exact GLM interleaved: pairs (x0,x1),(x2,x3)... rotate by theta_i
    theta = 1.0/(base**((np.arange(half)*2)/d))
    ang = t*theta
    out = x.copy()
    out[...,0::2] = x[...,0::2]*np.cos(ang) - x[...,1::2]*np.sin(ang)
    out[...,1::2] = x[...,0::2]*np.sin(ang) + x[...,1::2]*np.cos(ang)
    return out
# projections
x = rng.standard_normal(H).astype(np.float32)
WqA=rng.standard_normal((q_lora,H)).astype(np.float32)*0.1
WqB=rng.standard_normal((nhead*(nope+rope),q_lora)).astype(np.float32)*0.1
WkvA=rng.standard_normal((kv_lora+rope,H)).astype(np.float32)*0.1
WkB=rng.standard_normal((nhead*nope,kv_lora)).astype(np.float32)*0.1   # produces k_nope per head
WvB=rng.standard_normal((nhead*vdim,kv_lora)).astype(np.float32)*0.1   # produces v per head
Wo =rng.standard_normal((H,nhead*vdim)).astype(np.float32)*0.1
wqa=rmsnorm(WqA@x, np.ones(q_lora,np.float32))
q=(WqB@wqa).reshape(nhead,nope+rope)
kva=WkvA@x
latent, k_rope_raw = kva[:kv_lora], kva[kv_lora:]
kvln=rmsnorm(latent, np.ones(kv_lora,np.float32))
k_nope=(WkB@kvln).reshape(nhead,nope)
v=(WvB@kvln).reshape(nhead,vdim)
t=3
q_nope,q_rope=q[:,:nope],q[:,nope:]
q_rope_r=rope_interleaved(q_rope, 8e6, t)
k_rope_r=rope_interleaved(np.broadcast_to(k_rope_raw,(nhead,rope)).copy(), 8e6, t)
# single-token "attention" over just itself (t positions 0..t): build fake prior k/v cache
seq_n=t+1
K_nope_cache=rng.standard_normal((nhead,seq_n,nope)).astype(np.float32)
V_cache=rng.standard_normal((nhead,seq_n,vdim)).astype(np.float32)
K_rope_cache=rng.standard_normal((nhead,seq_n,rope)).astype(np.float32)
K_rope_cache[:,t]=k_rope_r[0] if False else k_rope_r  # place current
K_nope_cache[:,t]=k_nope
V_cache[:,t]=v
scores=np.einsum('hd,hnd->hn', q_nope, K_nope_cache) + np.einsum('hd,hnd->hn', q_rope_r, K_rope_cache)
scores=scores/np.sqrt(nope+rope)
# causal mask (only up to t)
mask=np.triu(np.full((seq_n,seq_n),-1e9),1); mask=np.where(np.arange(seq_n)<=t,0,-1e9)
scores=scores+mask
scores=scores-np.max(scores,axis=1,keepdims=True)
import numpy as np2
attn=np.exp(scores); attn=attn/attn.sum(1,keepdims=True)
out=np.einsum('hn,hnd->hd',attn,V_cache)   # [nhead, vdim]
mla_out=(Wo@out.reshape(-1))
print("MLA_OUT", json:=__import__('json').dumps([round(float(z),6) for z in mla_out]))
# --- MoE sigmoid routing reference ---
E=8; topk=2; scale=2.5
gate=rng.standard_normal((E,H)).astype(np.float32)*0.1
logits=gate@x
prob=1.0/(1.0+np.exp(-logits))
bias=rng.standard_normal(E).astype(np.float32)*0.1
sel=prob+bias
idx=np.argsort(-sel)[:topk]
w=prob[idx]; w=w/w.sum()*scale
print("MOE_IDX", list(map(int,idx)))
print("MOE_W", json:=__import__('json').dumps([round(float(z),6) for z in w]))
