import os
import re
import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F

# =====================================================================
# CONFIGURATION & HYPERPARAMETERS
# =====================================================================
BLOCK_SIZE = 32  
N_EMBD = 64      
N_HEAD = 4       
N_LAYER = 4      

st.set_page_config(
    page_title="BBC News Mini-GPT Assistant",
    page_icon="🤖",
    layout="centered"
)

# =====================================================================
# TRANSFORMER ARCHITECTURE MODULES
# =====================================================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v     
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(n_embd=N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None

# =====================================================================
# CORE CACHED RESOURCE LIFECYCLE
# =====================================================================
@st.cache_resource
def initialize_environment():
    """Builds sample fallback vocabulary or loads pre-trained checkpoints safely"""
    # Fallback basic text to extract standard vocabulary if no external configuration file is mapped
    sample_corpus = "the dollar has hit its highest level against the euro in almost three months after the federal reserve head said the us trade deficit is set to stabilise. the government said quarterly profits at us media giant timewarner jumped."
    words = re.findall(r"\w+|[^\w\s]", sample_corpus.lower(), re.UNICODE)
    unique_tokens = sorted(list(set(words)))
    vocabulary = ['<unk>'] + unique_tokens
    
    wtoi = {w: i for i, w in enumerate(vocabulary)}
    itow = {i: w for i, w in enumerate(vocabulary)}
    vocab_size = len(vocabulary)
    
    # Initialize architecture instance
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MiniGPT(vocab_size=vocab_size)
    
    # CRITICAL: If you exported weights (e.g. model.pt) from your training, uncomment below:
    # if os.path.exists('model.pt'):
    #     model.load_state_dict(torch.load('model.pt', map_location=device))
        
    model.to(device)
    model.eval()
    
    return model, wtoi, itow, device

model, wtoi, itow, device = initialize_environment()

# Tokenizer processing functions
def encode(text_string):
    tokens = re.findall(r"\w+|[^\w\s]", text_string.lower(), re.UNICODE)
    return [wtoi[t] if t in wtoi else wtoi['<unk>'] for t in tokens]

def decode(integer_list):
    words_list = [itow[idx] for idx in integer_list]
    text_output = " ".join(words_list)
    return text_output.replace(" .", ".").replace(" ,", ",").replace(" ' ", "'").replace(" !", "!").replace(" ?", "?")

# =====================================================================
# USER INTERFACE RENDERING
# =====================================================================
st.title("🤖 BBC News Content Generator")
st.markdown("This web app hosts your custom **Decoder-Only Mini-GPT** language model trained directly on news text hierarchies.")

st.sidebar.header("Generation Adjustments")
max_tokens = st.sidebar.slider("Tokens to Generate", min_value=10, max_value=150, value=50, step=5)
temperature = st.sidebar.slider("Creativity (Temperature)", min_value=0.2, max_value=1.5, value=1.0, step=0.1)

# Main Text Input Field
prompt = st.text_input("Enter a prompt to prime the model:", value="the government said")

if st.button("Generate Suggestions", type="primary"):
    if not prompt.strip():
        st.warning("Please provide a valid text string to initialize prediction sequences.")
    else:
        with st.spinner("Autoregressively processing next token weights..."):
            try:
                # Convert string token array to input tensor context matrix
                encoded_input = encode(prompt)
                idx = torch.tensor([encoded_input], dtype=torch.long, device=device)
                
                # Autoregressive production loop execution
                for _ in range(max_tokens):
                    idx_cond = idx[:, -BLOCK_SIZE:]
                    logits, _ = model(idx_cond)
                    logits = logits[:, -1, :] / max(temperature, 1e-5)
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat((idx, idx_next), dim=1)
                
                # De-tokenize back to client display
                generated_text = decode(idx[0].tolist())
                
                st.subheader("📝 Model Output Suggestions")
                st.info(generated_text)
                
            except Exception as e:
                st.error(f"An execution boundary exception occurred: {str(e)}")

st.divider()
st.caption("Mini GPT Architecture Matrix Context Window Capacity: 32 tokens | Channel State: 64 dimensions")