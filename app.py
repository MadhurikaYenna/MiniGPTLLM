import os
import re
import pickle
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

st.set_page_config(page_title="BBC News Mini-GPT Assistant", page_icon="🤖", layout="centered")

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
# RESOURCE LIFECYCLE MECHANICS (LOADING EXPORTED FILES)
# =====================================================================
@st.cache_resource
def initialize_environment():
    """Loads exact saved vocabulary state and model weights dynamically"""
    vocab_path = 'vocab.pkl'
    model_path = 'model.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Read your authentic vocabulary metadata
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        wtoi = vocab_data['wtoi']
        itow = vocab_data['itow']
        vocab_size = vocab_data['vocab_size']
    else:
        st.error(f"❌ '{vocab_path}' not found! Please download it from Colab and drop it in this folder.")
        st.stop()
        
    # 2. Instantiate Architecture Matching Your Dataset Array Boundaries
    model = MiniGPT(vocab_size=vocab_size)
    
    # 3. Inject Trained Neural Node Parameters
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.sidebar.success("✅ Fully Loaded Trained Weights from 'model.pt'!")
    else:
        st.sidebar.warning("⚠️ 'model.pt' missing from directory. Running on randomized structural weights.")
        
    model.to(device)
    model.eval()
    
    return model, wtoi, itow, device, vocab_size

# Unpack cached application engine resources
model, wtoi, itow, device, vocab_size = initialize_environment()

# =====================================================================
# TOKENIZATION INTERFACES
# =====================================================================
def encode(text_string):
    cleaned_string = text_string.lower()
    tokens = re.findall(r"\w+|[^\w\s]", cleaned_string, re.UNICODE)
    return [wtoi[t] if t in wtoi else wtoi['<unk>'] for t in tokens]

def decode(integer_list):
    words_list = [itow[idx] for idx in integer_list]
    text_output = " ".join(words_list)
    text_output = text_output.replace(" .", ".").replace(" ,", ",").replace(" ' ", "'")
    return text_output.replace(" !", "!").replace(" ?", "?")

# =====================================================================
# USER INTERFACE RENDERING
# =====================================================================
st.title("🤖 BBC News Content Generator")
st.markdown("This web application hosts your custom **Decoder-Only Mini-GPT** language model trained directly on news text hierarchies.")

st.sidebar.header("Generation Adjustments")
max_tokens = st.sidebar.slider("Tokens to Generate", min_value=10, max_value=150, value=50, step=5)
temperature = st.sidebar.slider("Creativity (Temperature)", min_value=0.2, max_value=1.5, value=1.0, step=0.1)

# Application state diagnostic information panel
st.sidebar.divider()
st.sidebar.metric(label="Active Vocabulary Size", value=f"{vocab_size:,} words")
st.sidebar.text(f"Hardware Device: {device.upper()}")

# Main Text Input Field
prompt = st.text_input("Enter a prompt to prime the model:", value="the government said")

if st.button("Generate Suggestions", type="primary"):
    if not prompt.strip():
        st.warning("Please provide a valid text string to initialize prediction sequences.")
    else:
        with st.spinner("Autoregressively processing next token weights..."):
            try:
                # Convert input tokens using our uploaded map indexes
                encoded_input = encode(prompt)
                idx = torch.tensor([encoded_input], dtype=torch.long, device=device)
                
                # Autoregressive generation loop execution
                for _ in range(max_tokens):
                    idx_cond = idx[:, -BLOCK_SIZE:]
                    logits, _ = model(idx_cond)
                    logits = logits[:, -1, :] / max(temperature, 1e-5)
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat((idx, idx_next), dim=1)
                
                # Render clean decoded translation back to frontend layout
                generated_text = decode(idx[0].tolist())
                
                st.subheader("📝 Model Output Suggestions")
                st.info(generated_text)
                
            except Exception as e:
                st.error(f"An execution boundary exception occurred: {str(e)}")

st.divider()
st.caption("Mini GPT Architecture Matrix Context Window Capacity: 32 tokens | Channel State: 64 dimensions")
