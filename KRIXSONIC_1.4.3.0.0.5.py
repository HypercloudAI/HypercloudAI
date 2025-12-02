# KRIX — HYPERCLOUD INTELLIGENCE (CLEAN-ENGLISH UPGRADE)
# Save as: krix_hypercloud_upgraded.py
# Based on KRIXSONIC_1.4.3.0.0.4.py (upgraded to fetch only clean English, LEARN_MODE trains on every message)
# MADE BY ARIYAN AHMED; Upgraded by assistant per request.
# Keep this header when you use this file.

from __future__ import annotations
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import sys, time, json, threading, argparse, random, hashlib, traceback, re, tempfile, collections, urllib.parse
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup

# Third-party imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.utils.rnn import pad_sequence
    import numpy as np
except Exception as e:
    print("Missing dependency:", e)
    print("Install: pip install torch torchvision requests beautifulsoup4 numpy")
    raise

# -------------------------
# Config & defaults
# -------------------------
DATASTORE_WINDOWS = Path(r"C:\Users\user\Desktop\HyperCloud_Intelligence\DATASTORE_WINDOWS")
DATASTORE_FILE = DATASTORE_WINDOWS / "datastore.json"
BASE_DIR = Path.home() / "krix_hypercloud"
DATA_DIR = BASE_DIR / "data"
CKPT_PATH = BASE_DIR / "krix_hypercloud.ckpt"
LOG_FILE = DATA_DIR / "krix_hypercloud.log"

FAST_MODE = os.environ.get("KRIX_FAST_MODE", "1") == "1"
LEARN_ENV = os.environ.get("KRIX_LEARN_MODE", "0") == "1"

EMBED_DIM = int(os.environ.get("KRIX_EMBED_DIM", "384"))
N_HEADS = int(os.environ.get("KRIX_N_HEADS", "6"))
N_LAYERS = int(os.environ.get("KRIX_N_LAYERS", "6"))
FF_DIM = int(os.environ.get("KRIX_FF_DIM", "1536"))
BLOCK_SIZE = int(os.environ.get("KRIX_BLOCK_SIZE", "768"))
BATCH_SIZE = int(os.environ.get("KRIX_BATCH_SIZE", "4"))

if FAST_MODE:
    EMBED_DIM = min(EMBED_DIM, 256)
    N_HEADS = min(N_HEADS, 4)
    N_LAYERS = min(N_LAYERS, 4)
    FF_DIM = min(FF_DIM, 1024)
    BLOCK_SIZE = min(BLOCK_SIZE, 512)
    BATCH_SIZE = max(1, min(BATCH_SIZE, 2))

GEN_MAX_TOKENS = int(os.environ.get("KRIX_GEN_MAX_TOKENS", "128"))
TOP_K = int(os.environ.get("KRIX_TOP_K", "40"))
TOP_P = float(os.environ.get("KRIX_TOP_P", "0.92"))
TEMPERATURE = float(os.environ.get("KRIX_TEMPERATURE", "0.7"))

SAVE_INTERVAL = int(os.environ.get("KRIX_SAVE_INTERVAL", "45"))
TRAIN_EPOCHS_BG = int(os.environ.get("KRIX_TRAIN_EPOCHS_BG", "1"))
TRAIN_EPOCHS_QUICK = int(os.environ.get("KRIX_TRAIN_EPOCHS_QUICK", "1"))

RAG_K = int(os.environ.get("KRIX_RAG_K", "3"))
USE_FP16 = os.environ.get("KRIX_FP16", "auto")
COMPILE_MODEL = False

HEAVY_USER_LEARN = os.environ.get("KRIX_HEAVY_USER_LEARN", "1") == "1"
HEAVY_USER_EPOCH_MULT = int(os.environ.get("KRIX_HEAVY_USER_EPOCH_MULT", "2"))

TYPING_CHUNK = int(os.environ.get("KRIX_TYPING_CHUNK", "12"))
TYPING_DELAY = float(os.environ.get("KRIX_TYPING_DELAY", "0.01"))
SPINNER_INTERVAL = float(os.environ.get("KRIX_SPINNER_INTERVAL", "0.08"))

USER_AGENT = "krix-hypercloud/1.0"
MAX_GEN_CHARS = int(os.environ.get("KRIX_MAX_GEN_CHARS", "120"))

# Clean English seed sources (trusted, English-first)
SEED_URLS = [
    "https://docs.python.org/3/",
    "https://www.python.org/doc/",
    "https://realpython.com/",
    "https://developer.mozilla.org/en-US/",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Computer_science",
    "https://www.gutenberg.org/ebooks/1342",  # Pride and Prejudice (English public-domain)
    "https://www.gutenberg.org/ebooks/1",    # Project Gutenberg front (English)
    "https://en.wikipedia.org/wiki/Main_Page"
]

AUTO_FETCH_INTERVAL = int(os.environ.get("KRIX_AUTO_FETCH_INTERVAL", str(60 * 60)))
AUTO_FETCH_MAX_PAGES_PER_CYCLE = int(os.environ.get("KRIX_AUTO_FETCH_MAX_PAGES_PER_CYCLE", str(len(SEED_URLS))))
GEN_TIMEOUT = int(os.environ.get("KRIX_GEN_TIMEOUT", "85"))

MAX_QUEUE_SIZE = int(os.environ.get("KRIX_MAX_QUEUE", "4096"))
CUDA_MAX_FRACTION = float(os.environ.get("KRIX_CUDA_MAX_FRACTION", "0.80"))

SAFE_MODE_DEFAULT = os.environ.get("KRIX_SAFE_MODE", "1") == "1"
SAFE_KEYWORDS = [
    "exploit", "malware", "ransomware", "ddos", "bomb", "detonate", "how to make a bomb",
    "poison", "assassinate", "kill", "illegal", "bypass authentication", "rootkit",
    "privilege escalation", "bios flash exploit", "evil-grade", "dump passwords", "steal",
    "carding", "credit card", "ssn", "personally identifiable", "unauthorized access",
]
SAFE_WORD_PATTERNS = [re.compile(re.escape(w), re.IGNORECASE) for w in SAFE_KEYWORDS]

# -------------------------
# Utilities
# -------------------------
def ensure_dirs():
    try:
        DATASTORE_WINDOWS.mkdir(parents=True, exist_ok=True)
        for p in (DATA_DIR, BASE_DIR):
            p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print("ensure_dirs error:", e)

def now_iso():
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")

def log(msg: str):
    try:
        ensure_dirs()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{now_iso()}] {msg}\n")
    except Exception:
        try:
            print(f"[{now_iso()}] LOG-ERR: {msg}")
        except Exception:
            pass

PRINTABLE_RE = re.compile(r"[^\t\n\r\x20-\x7E\u00A0-\u017F]")
def safe_text(s: str) -> str:
    t = PRINTABLE_RE.sub("", s)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def contains_unsafe(text: str) -> bool:
    low = str(text)
    for patt in SAFE_WORD_PATTERNS:
        if patt.search(low):
            return True
    return False

# -------------------------
# Clean-English detection (heuristics)
# -------------------------
# small set of common English words used to check if content is English-like
_COMMON_ENGLISH = {
    "the","be","to","of","and","a","in","that","have","I","it","for","not","on","with",
    "he","as","you","do","at","this","but","his","by","from","they","we","say","her","she",
    "or","an","will","my","one","all","would","there","their","what","so","up","out","if",
    "about","who","get","which","go","me","when","make","can","like","time","no","just",
    "him","know","take","people","into","year","your","good","some","could","them","see",
    "other","than","then","now","look","only","come","its","over","think","also","back",
    "after","use","two","how","our","work","first","well","way","even","new","want","because",
}

NON_LATIN_RE = re.compile(r"[^\u0000-\u00ff]")  # catch non-latin scripts roughly

def is_mostly_english(text: str, min_ratio: float = 0.20) -> bool:
    """
    Heuristic: return True if text appears to be primarily English.
    - Checks fraction of common-english tokens
    - Rejects if too many non-latin characters present
    """
    if not text or len(text) < 40:
        return False
    # quick non-latin filter
    non_latin = len(NON_LATIN_RE.findall(text))
    frac_non_latin = non_latin / max(1, len(text))
    if frac_non_latin > 0.02:
        # contains significant non-latin characters -> likely multilingual or script noise
        return False
    # token match ratio
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    if not words:
        return False
    matches = sum(1 for w in words if w in _COMMON_ENGLISH)
    ratio = matches / len(words)
    # also check average word length and presence of many punctuation sequences
    avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0
    if avg_word_len < 1.5 or avg_word_len > 20:
        return False
    return ratio >= min_ratio

# -------------------------
# Datastore (unchanged)
# -------------------------
class DataStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: Dict[str, Any] = {
            "meta": {"created": now_iso(), "last_save": None},
            "items": [],
            "topics": {},
            "history": []
        }
        self._dirty = False
        self._load()

    def _load(self):
        ensure_dirs()
        if not self.path.exists():
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                log(f"create datastore file failed: {e}")
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                j = json.load(f)
                if isinstance(j, dict):
                    for k, v in j.items():
                        self.data[k] = v
        except Exception as e:
            log(f"datastore load failed: {e}")

    def _atomic_write(self, target: Path, obj: Any):
        tmpfd, tmppath = tempfile.mkstemp(dir=str(target.parent))
        try:
            with os.fdopen(tmpfd, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmppath, str(target))
        except Exception as e:
            log(f"atomic write failed: {e}")
            try:
                if os.path.exists(tmppath):
                    os.remove(tmppath)
            except Exception:
                pass

    def save(self, force: bool = False):
        with self.lock:
            if not self._dirty and not force:
                return
            try:
                self.data["meta"]["last_save"] = now_iso()
                self._atomic_write(self.path, self.data)
                self._dirty = False
            except Exception as e:
                log(f"datastore save failed: {e}")

    def push_history(self, role: str, text: str):
        with self.lock:
            last = self.data["history"][-1] if self.data["history"] else None
            entry = {"ts": now_iso(), "role": role, "text": text}
            if last and isinstance(last, dict) and last.get("role") == role and last.get("text") == text:
                return
            self.data["history"].append(entry)
            if len(self.data["history"]) > 40000:
                self.data["history"] = self.data["history"][-40000:]
            self._dirty = True

    def _find_item_by_text(self, text: str) -> Optional[str]:
        for it in self.data["items"]:
            if it.get("text") == text:
                return it.get("id")
        return None

    def add_item(self, text: str, src: str="user", embedding: Optional[List[float]]=None, indicators: Optional[List[str]]=None) -> str:
        with self.lock:
            existing = self._find_item_by_text(text)
            if existing:
                return existing
            iid = hashlib.sha256((text + str(time.time()) + str(random.random())).encode("utf-8")).hexdigest()[:16]
            item = {"id": iid, "text": text, "src": src, "ts": now_iso(), "embedding": embedding, "indicators": indicators or [], "topic": None}
            self.data["items"].append(item)
            self._dirty = True
            return iid

    def update_embedding(self, item_id: str, emb: List[float]):
        with self.lock:
            for it in self.data["items"]:
                if it["id"] == item_id:
                    it["embedding"] = emb
                    self._dirty = True
                    return True
        return False

    def update_item(self, item_id: str, **kwargs):
        with self.lock:
            for it in self.data["items"]:
                if it["id"] == item_id:
                    for k,v in kwargs.items():
                        if k == "append_indicator" and isinstance(v, list):
                            it.setdefault("indicators", [])
                            for ins in v:
                                if ins not in it["indicators"]:
                                    it["indicators"].append(ins)
                        else:
                            it[k] = v
                    self._dirty = True
                    return True
        return False

    def list_items(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.data["items"])

    def get_recent_items_last_minutes(self, minutes: int = 10) -> List[Dict[str, Any]]:
        cutoff = time.time() - minutes * 60
        out = []
        with self.lock:
            for it in reversed(self.data["items"]):
                try:
                    ts = it.get("ts")
                    if not ts: continue
                    dt = __import__("datetime").datetime.fromisoformat(ts)
                    if dt.timestamp() >= cutoff:
                        out.append(it)
                    else:
                        break
                except Exception:
                    continue
        return out

    def add_or_create_topic(self, label: str, centroid: List[float]) -> str:
        with self.lock:
            tid = hashlib.sha256((label + str(time.time())).encode("utf-8")).hexdigest()[:12]
            self.data["topics"][tid] = {"id": tid, "label": label, "centroid": centroid, "count": 1}
            self._dirty = True
            return tid

    def update_topic(self, topic_id: str, new_centroid: List[float], add_count: int = 1):
        with self.lock:
            t = self.data["topics"].get(topic_id)
            if t:
                old = np.array(t["centroid"], dtype=np.float32)
                new = np.array(new_centroid, dtype=np.float32)
                total = t.get("count", 1) + add_count
                updated = ((old * t.get("count", 1)) + (new * add_count)) / total
                t["centroid"] = updated.tolist()
                t["count"] = total
                self._dirty = True

    def assign_topic(self, item_id: str, topic_id: str):
        with self.lock:
            for it in self.data["items"]:
                if it["id"] == item_id:
                    it["topic"] = topic_id
                    self._dirty = True
                    return True
        return False

    def get_topics(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.data["topics"])

# -------------------------
# Tokenizer & Model (unchanged)
# -------------------------
class ByteTokenizer:
    def __init__(self, block_size: int = BLOCK_SIZE):
        self.PAD = 0; self.UNK = 1; self.BOS = 2; self.EOS = 3
        self.offset = 4
        self.block_size = block_size

    def encode(self, text: str, add_bos: bool=True, add_eos: bool=True, max_len: Optional[int]=None) -> List[int]:
        if max_len is None:
            max_len = self.block_size - 1
        b = text.encode("utf-8", errors="replace")[:max_len]
        ids = [self.BOS] if add_bos else []
        ids += [min(255, ch) + self.offset for ch in b]
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: List[int]) -> str:
        b = bytearray()
        for i in ids:
            if i in (self.PAD, self.UNK, self.BOS, self.EOS): continue
            v = i - self.offset
            if 0 <= v < 256: b.append(v)
        return b.decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        return 4 + 256

class HyperTransformer(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, n_heads:int, n_layers:int, ff_dim:int, block_size:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim, activation='gelu', batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.LongTensor):
        b,s = x.shape
        device = x.device
        pos = torch.arange(s, device=device).unsqueeze(0).expand(b,s)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(s, s, device=device) * float('-inf'), diagonal=1)
        out = h
        for lay in self.layers:
            out = lay(out, src_mask=mask)
        out = self.ln(out)
        logits = self.head(out)
        return logits

class ModelManager:
    def __init__(self, device: Optional[str]=None, train_enabled: bool=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = ByteTokenizer(block_size=BLOCK_SIZE)
        self.vocab_size = self.tokenizer.vocab_size()
        self.model = HyperTransformer(self.vocab_size, EMBED_DIM, N_HEADS, N_LAYERS, FF_DIM, BLOCK_SIZE).to(self.device)
        self.train_enabled = train_enabled
        self.opt = None
        if self.train_enabled:
            try:
                self.opt = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-2)
            except Exception as e:
                log(f"optimizer init failed: {e}")
                self.opt = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.lock = threading.Lock()
        self.pretrained = False
        self.use_fp16 = (USE_FP16 == "1") or (USE_FP16 == "auto" and torch.cuda.is_available())
        if self.use_fp16 and not self.device.startswith("cuda"):
            self.use_fp16 = False
        self.scaler = torch.cuda.amp.GradScaler() if (self.use_fp16 and self.device.startswith("cuda")) else None
        self.last_loss = None

    def load_checkpoint(self, path: Path) -> bool:
        if not path.exists(): return False
        try:
            with self.lock:
                dd = torch.load(str(path), map_location=self.device)
                if "model_state" in dd:
                    self.model.load_state_dict(dd["model_state"])
                else:
                    self.model.load_state_dict(dd)
                self.pretrained = True
                log(f"loaded ckpt {path}")
                return True
        except Exception as e:
            log(f"load ckpt failed: {e}")
            return False

    def save_checkpoint(self, path: Path):
        try:
            with self.lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = str(path) + ".tmp"
                torch.save({"model_state": self.model.state_dict(), "meta": {"saved": now_iso()}}, tmp)
                os.replace(tmp, str(path))
                log(f"saved ckpt {path}")
        except Exception as e:
            log(f"save ckpt failed: {e}")

    def _prepare_sequences(self, texts: List[str]) -> List[torch.LongTensor]:
        seqs = []
        for t in texts:
            ids = self.tokenizer.encode(t, add_bos=True, add_eos=True, max_len=self.tokenizer.block_size-1)
            if len(ids) < 2: continue
            seqs.append(torch.tensor(ids, dtype=torch.long))
        return seqs

    def train_on_texts(self, texts: List[str], epochs: int = 1, batch_size: int = 2) -> Optional[float]:
        if not self.train_enabled or self.opt is None:
            log("train_on_texts skipped: training disabled")
            return None
        seqs = self._prepare_sequences(texts)
        if not seqs:
            return None
        with self.lock:
            self.model.train()
            random.shuffle(seqs)
            count = 0
            for ep in range(epochs):
                for i in range(0, len(seqs), batch_size):
                    batch = seqs[i:i+batch_size]
                    X = [b[:-1] for b in batch]; Y = [b[1:] for b in batch]
                    Xp = pad_sequence(X, batch_first=True, padding_value=0).to(self.device)
                    Yp = pad_sequence(Y, batch_first=True, padding_value=0).to(self.device)
                    if self.use_fp16 and self.device.startswith("cuda"):
                        self.opt.zero_grad()
                        with torch.cuda.amp.autocast():
                            logits = self.model(Xp)
                            bsz, slen, v = logits.size()
                            loss = self.criterion(logits.view(bsz*slen, v), Yp.view(bsz*slen))
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.opt)
                        self.scaler.update()
                    else:
                        self.opt.zero_grad()
                        logits = self.model(Xp)
                        bsz, slen, v = logits.size()
                        loss = self.criterion(logits.view(bsz*slen, v), Yp.view(bsz*slen))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.opt.step()
                    try:
                        del Xp, Yp, logits, loss
                    except Exception:
                        pass
                    if torch.cuda.is_available():
                        try: torch.cuda.empty_cache()
                        except Exception: pass
                    count += 1
            self.last_loss = None
            log(f"train completed on {len(texts)} texts (fast/low-mem mode)")
            return self.last_loss
        return None

    def _apply_top_k_top_p(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        if top_k and top_k > 0:
            vals, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            min_val = vals[-1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf'), device=logits.device), logits)
        if top_p and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            mask = cumulative > top_p
            mask[..., 0] = False
            indices_to_remove = sorted_idx[mask]
            logits[indices_to_remove] = float('-inf')
        return logits

    def _sample_from_logits(self, last: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
        logits = last / (temperature if temperature > 0 else 1.0)
        logits = self._apply_top_k_top_p(logits, top_k, top_p)
        probs = torch.softmax(logits, dim=-1)
        idx = int(torch.multinomial(probs, num_samples=1).item())
        return idx

    def generate_stream(self, prompt: str, max_new_tokens: int = GEN_MAX_TOKENS,
                        temperature: float = TEMPERATURE, top_k: int = TOP_K, top_p: float = TOP_P,
                        rag_context: Optional[List[str]] = None, stop_event: Optional[threading.Event] = None):
        with self.lock:
            try:
                self.model.eval()
                if rag_context:
                    concat = ""
                    for c in rag_context[::-1]:
                        if len(concat) + len(c) > 1024:
                            break
                        concat = c + "\n\n" + concat
                    cand = concat + "\n\n" + prompt
                    ids = self.tokenizer.encode(cand, add_bos=True, add_eos=False, max_len=self.tokenizer.block_size-1)
                else:
                    ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False, max_len=self.tokenizer.block_size-1)
                cur = torch.tensor([ids], dtype=torch.long).to(self.device)
                generated: List[int] = []
                inf = torch.inference_mode()
                try:
                    inf.__enter__()
                    for step in range(1, max_new_tokens + 1):
                        if stop_event is not None and stop_event.is_set():
                            out = self.tokenizer.decode(generated)
                            out = safe_text(out)
                            yield (out, step)
                            break
                        if cur.shape[1] > self.model.block_size:
                            cur = cur[:, -self.model.block_size:]
                        if self.use_fp16 and self.device.startswith("cuda"):
                            with torch.cuda.amp.autocast():
                                logits = self.model(cur)
                        else:
                            logits = self.model(cur)
                        last = logits[0, -1, :]
                        idx = self._sample_from_logits(last, temperature, top_k, top_p)
                        generated.append(idx)
                        cur = torch.cat([cur, torch.tensor([[idx]], dtype=torch.long).to(self.device)], dim=1)
                        out = self.tokenizer.decode(generated)
                        out = safe_text(out)
                        if len(out) >= MAX_GEN_CHARS:
                            out = out[:MAX_GEN_CHARS]
                            yield (out, step)
                            break
                        if step % 2 == 0 or idx == self.tokenizer.EOS:
                            yield (out, step)
                        if idx == self.tokenizer.EOS:
                            break
                        try:
                            del logits, last
                        except Exception:
                            pass
                        if torch.cuda.is_available():
                            try: torch.cuda.empty_cache()
                            except Exception: pass
                    return
                finally:
                    inf.__exit__(None, None, None)
            except Exception as e:
                log(f"generate_stream err: {e}")
                yield (f"[generation error: {e}]", 0)
                return

    def get_embedding(self, text: str) -> List[float]:
        self.model.eval()
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True, max_len=self.tokenizer.block_size-1)
        x = torch.tensor([ids], dtype=torch.long).to(self.device)
        with torch.inference_mode():
            tok = self.model.tok_emb(x)
            vec = tok.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(vec) + 1e-12
            vec = vec / norm
            return vec.tolist()

# -------------------------
# Helpers: improved fetch & safe fetcher
# -------------------------
HEADERS = {"User-Agent": USER_AGENT}

def _strip_noncontent(soup: BeautifulSoup):
    """
    Remove obvious nav, lang links, sidebars, footers and elements with non-en lang attributes.
    This reduces multilingual sidebar noise.
    """
    # remove scripts/styles
    for el in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside"]):
        try:
            el.extract()
        except Exception:
            pass
    # remove common wiki sidebars and language lists
    for sel in ["#mw-navigation", "#mw-panel", ".navbox", ".mw-lang-link", ".langlinks", "#footer", ".toc", ".mw-headline"]:
        for el in soup.select(sel):
            try:
                el.extract()
            except Exception:
                pass
    # remove any element with a lang attribute not 'en'
    for el in soup.find_all(attrs={"lang": True}):
        try:
            if el.attrs.get("lang", "").lower() != "en":
                el.extract()
        except Exception:
            pass
    return soup

def fetch_url_text(url: str, max_len:int = 30000) -> Optional[str]:
    """
    Fetch page, attempt to return cleaned English-only text or None.
    """
    try:
        if not url.startswith("http"):
            url = "https://" + url
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            log(f"fetch {url} status {r.status_code}")
            return None
        ct = r.headers.get("Content-Type", "")
        if "text" not in ct:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        soup = _strip_noncontent(soup)
        lines = [ln.strip() for ln in soup.get_text().splitlines() if ln.strip()]
        txt = "\n".join(lines)
        txt = safe_text(txt)
        # quick English check
        if not is_mostly_english(txt, min_ratio=0.18):
            # reject if not English-like
            log(f"fetch skipped non-English or low-quality: {url}")
            return None
        return txt[:max_len]
    except Exception as e:
        log(f"fetch err {e} for {url}")
        return None

def fetch_and_chunk_to_bg(url: str, bg: 'BackgroundEngine', max_chars:int=20000, chunk_chars:int=800) -> bool:
    """
    Only queue chunks if they pass the clean-English filter.
    """
    txt = fetch_url_text(url, max_len=max_chars)
    if not txt:
        return False
    # further split by paragraphs and filter short/bad paragraphs
    paras = [p.strip() for p in re.split(r'\n\s*\n', txt) if len(p.strip()) > 60]
    chunks = []
    cur = ""
    for p in paras:
        cand = (cur + "\n\n" + p).strip() if cur else p
        if len(cand) <= chunk_chars:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
            # break p into smaller slices
            for i in range(0, len(p), chunk_chars):
                chunks.append(p[i:i+chunk_chars])
            cur = ""
    if cur:
        chunks.append(cur)
    # filter chunks by english heuristics
    good_chunks = [c for c in chunks if is_mostly_english(c, min_ratio=0.16)]
    if not good_chunks:
        log(f"fetch_no_good_chunks {url}")
        return False
    bg.queue_many(good_chunks, show_progress=False, priority=False)
    log(f"fetched {url} queued {len(good_chunks)} chunks")
    return True

def try_fetch_reference(prompt: str, max_chars: int = 1200) -> Optional[Dict[str,str]]:
    """
    Try Wikipedia API first (clean extracts). If found, verify extract is English and return.
    Fallback to DuckDuckGo + seed scan but only if results pass English filter.
    """
    try:
        q = prompt.strip()
        if not q: return None
        # Wikipedia search via API
        try:
            params = {"action":"query","list":"search","srsearch": q,"format":"json","srlimit":3}
            url = "https://en.wikipedia.org/w/api.php"
            r = requests.get(url, params=params, headers=HEADERS, timeout=6)
            if r.ok:
                j = r.json()
                hits = j.get("query", {}).get("search", [])
                for h in hits:
                    title = h.get("title")
                    if not title: continue
                    page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
                    txt = fetch_url_text(page_url, max_len=max_chars)
                    if not txt: continue
                    tokens = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 2][:8]
                    score = sum(1 for t in tokens if t in txt.lower())
                    if score >= max(1, len(tokens)//3):
                        snippet = txt[:max_chars]
                        return {"source": page_url, "title": title, "snippet": safe_text(snippet)}
        except Exception:
            pass

        # DuckDuckGo fallback (lightweight) -- but vet results for English
        try:
            search_url = "https://duckduckgo.com/html/"
            params = {"q": q}
            r = requests.post(search_url, data=params, headers=HEADERS, timeout=6)
            if r.ok:
                soup = BeautifulSoup(r.text, "html.parser")
                results = []
                for a in soup.select("a.result__a")[:6]:
                    href = a.get("href")
                    if not href: continue
                    results.append(href)
                if not results:
                    for a in soup.select("a")[:10]:
                        href = a.get("href")
                        if href and href.startswith("http"):
                            results.append(href)
                for rurl in results[:5]:
                    try:
                        txt = fetch_url_text(rurl, max_len=max_chars)
                        if not txt: continue
                        tokens = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 3][:6]
                        if not tokens: continue
                        score = sum(1 for t in tokens if t in txt.lower())
                        if score >= 1:
                            snippet = txt[:max_chars]
                            title = rurl.split("/")[-1][:80]
                            return {"source": rurl, "title": title, "snippet": safe_text(snippet)}
                    except Exception:
                        continue
        except Exception:
            pass

        # Fallback: seed scan (English-first) - only return if match
        for u in SEED_URLS:
            try:
                txt = fetch_url_text(u, max_len=max_chars)
                if not txt: continue
                tokens = [t for t in re.findall(r"\w+", prompt.lower()) if len(t) > 3][:6]
                if not tokens: continue
                score = sum(1 for t in tokens if t in txt.lower())
                if score >= 1:
                    snippet = txt[:max_chars]
                    return {"source": u, "title": u, "snippet": safe_text(snippet)}
            except Exception:
                continue
    except Exception as e:
        log(f"try_fetch_reference err: {e}")
    return None

def analyze_grammar_and_patterns(text: str) -> Dict[str, Any]:
    try:
        sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = re.findall(r"\b\w+\b", text.lower())
        avg_w = (sum(len(w) for w in words) / len(words)) if words else 0.0
        top = collections.Counter(words).most_common(6)
        contractions = len(re.findall(r"\b\w+'\w+\b", text))
        capitalized = len([w for w in re.findall(r"\b[A-Z][a-z]+\b", text)])
        punctuation_density = len(re.findall(r"[.,;:!?]", text)) / (len(text) + 1)
        return {
            "sentences": len(sents),
            "words": len(words),
            "avg_word_len": round(avg_w, 2),
            "top_words": top,
            "contractions": contractions,
            "capitalized_words": capitalized,
            "punct_density": round(punctuation_density, 4)
        }
    except Exception as e:
        log(f"analyze_grammar_and_patterns err: {e}")
        return {}

# -------------------------
# Background engine (unchanged logic, but uses clean fetch functions)
# -------------------------
class BackgroundEngine:
    def __init__(self, store: DataStore, model: ModelManager, enabled: bool = True):
        self.store = store
        self.model = model
        self.enabled = enabled
        self.train_q: Queue = Queue()
        self.high_train_q: Queue = Queue()
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self.trainer_thread = threading.Thread(target=self._trainer_loop, daemon=True)
        self.saver_thread = threading.Thread(target=self._saver_loop, daemon=True)
        self._last_save = 0
        self._active_training = threading.Event()
        self._max_batch = max(1, BATCH_SIZE * 8)

    def start(self):
        if not self.enabled:
            log("background disabled")
            return
        if not self.trainer_thread.is_alive():
            self.stop_event.clear()
            self.trainer_thread = threading.Thread(target=self._trainer_loop, daemon=True)
            self.trainer_thread.start()
        if not self.saver_thread.is_alive():
            self.saver_thread = threading.Thread(target=self._saver_loop, daemon=True)
            self.saver_thread.start()
        log("background started")

    def stop(self):
        self.stop_event.set()
        self.resume()
        try:
            if self.trainer_thread.is_alive():
                self.trainer_thread.join(timeout=1.0)
        except Exception:
            pass

    def pause(self, block: bool = False, timeout: float = 5.0):
        self.paused.set()
        log("background pause requested")
        if block:
            start = time.time()
            while self._active_training.is_set() and (time.time() - start) < timeout:
                time.sleep(0.05)
            if self._active_training.is_set():
                log("background pause: active training still running after timeout")
            else:
                log("background pause: confirmed idle")

    def resume(self):
        if self.paused.is_set():
            self.paused.clear()
            log("background resumed")

    def queue_training(self, text: str, item_id: Optional[str]=None, priority: bool=False):
        try:
            total_size = self.train_q.qsize() + self.high_train_q.qsize()
            if total_size >= MAX_QUEUE_SIZE:
                log("train_q full: dropping incoming item")
                return
            if priority:
                self.high_train_q.put_nowait((item_id, text))
            else:
                self.train_q.put_nowait((item_id, text))
        except Exception:
            pass

    def queue_many(self, texts: List[Union[str, Tuple[Optional[str], str]]], show_progress: bool = False, priority: bool=False):
        n = len(texts)
        for i, t in enumerate(texts, 1):
            if isinstance(t, (list, tuple)) and len(t) == 2:
                iid = t[0]; txt = t[1]
                self.queue_training(txt, item_id=iid, priority=priority)
            else:
                self.queue_training(t, item_id=None, priority=priority)
            if show_progress:
                pct = int(i / n * 100)
                sys.stdout.write(f"\r[enqueueing chunks: {i}/{n} ({pct}%) {'■' * (pct//10)}]")
                sys.stdout.flush()
        if show_progress:
            sys.stdout.write("\n")

    def _cuda_memory_high(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            total = props.total_memory
            used = torch.cuda.memory_allocated(dev)
            if total <= 0:
                return False
            fraction = float(used) / float(total)
            return fraction > CUDA_MAX_FRACTION
        except Exception:
            return False

    def _assign_topic(self, item_id: str, emb: List[float], text: str):
        topics = self.store.get_topics()
        if not topics:
            label = text[:64].strip()
            tid = self.store.add_or_create_topic(label, emb)
            self.store.assign_topic(item_id, tid)
            return
        qv = np.array(emb, dtype=np.float32)
        best_tid = None; best_sim = -1.0
        for tid, info in topics.items():
            cent = np.array(info.get("centroid", []), dtype=np.float32)
            if cent.size == 0: continue
            sim = float(np.dot(qv, cent))
            if sim > best_sim:
                best_sim = sim; best_tid = tid
        if best_tid and best_sim > 0.78:
            self.store.assign_topic(item_id, best_tid)
            self.store.update_topic(best_tid, emb, add_count=1)
        else:
            label = text[:64].strip()
            tid = self.store.add_or_create_topic(label, emb)
            self.store.assign_topic(item_id, tid)

    def assign_topic_for_item(self, item_id: str, emb: List[float], text: str):
        try:
            self._assign_topic(item_id, emb, text)
        except Exception as e:
            log(f"assign_topic_for_item err: {e}")

    def _trainer_loop(self):
        while not self.stop_event.is_set():
            try:
                if self.paused.is_set():
                    time.sleep(0.05)
                    continue

                batch: List[Tuple[Optional[str], str]] = []

                try:
                    first = self.high_train_q.get(timeout=0.5)
                    batch.append(first)
                except Empty:
                    try:
                        first = self.train_q.get(timeout=1.0)
                        batch.append(first)
                    except Empty:
                        recent = [h.get("text") for h in self.store.data.get("history", [])[-512:] if isinstance(h, dict)]
                        batch = [(None, t) for t in recent[:128]] if recent else []
                        if not batch:
                            time.sleep(0.3)
                            continue

                while len(batch) < self._max_batch:
                    try:
                        nxt = self.high_train_q.get_nowait()
                        batch.append(nxt)
                        continue
                    except Empty:
                        pass
                    try:
                        nxt = self.train_q.get_nowait()
                        batch.append(nxt)
                    except Empty:
                        break

                if self._cuda_memory_high():
                    log("trainer: cuda memory high; requeueing batch and sleeping")
                    for it in batch:
                        try:
                            self.train_q.put_nowait(it)
                        except Exception:
                            pass
                    time.sleep(2.0)
                    continue

                if self.paused.is_set():
                    for it in batch:
                        try:
                            self.train_q.put_nowait(it)
                        except Exception:
                            pass
                    time.sleep(0.05)
                    continue

                texts = [t for (_id, t) in batch]
                contains_high = any((_id is not None and _id.startswith("")) for (_id, _t) in batch)
                self._active_training.set()
                try:
                    per_chunk = max(1, min(len(texts), BATCH_SIZE))
                    epochs = TRAIN_EPOCHS_BG if not FAST_MODE else 1
                    if HEAVY_USER_LEARN and len(batch) and (len(batch) <= self._max_batch):
                        epochs = max(epochs, TRAIN_EPOCHS_BG * HEAVY_USER_EPOCH_MULT)
                    self.model.train_on_texts(texts, epochs=epochs, batch_size=per_chunk)
                finally:
                    self._active_training.clear()

                for item_id, txt in batch:
                    try:
                        emb = self.model.get_embedding(txt)
                        inds = []
                        if "def " in txt or "class " in txt or "import " in txt:
                            inds = extract_indicators(txt)
                        if item_id:
                            self.store.update_embedding(item_id, emb)
                            self.store.update_item(item_id, indicators=inds)
                            try:
                                self.assign_topic_for_item(item_id, emb, txt)
                            except Exception:
                                pass
                        else:
                            iid = self.store.add_item(txt, src="bg", embedding=emb, indicators=inds)
                            try:
                                self.assign_topic_for_item(iid, emb, txt)
                            except Exception:
                                pass
                    except Exception as e:
                        log(f"bg embed err: {e}")
            except Exception as e:
                log(f"trainer loop err: {e}")
                time.sleep(0.5)

    def _saver_loop(self):
        while not self.stop_event.is_set():
            try:
                nowt = time.time()
                try:
                    self.store.save(force=False)
                except Exception as e:
                    log(f"saver store save err: {e}")
                if nowt - self._last_save > SAVE_INTERVAL * 2:
                    try:
                        if not FAST_MODE or (FAST_MODE and nowt - self._last_save > SAVE_INTERVAL * 6):
                            self.model.save_checkpoint(CKPT_PATH)
                    except Exception as e:
                        log(f"saver save ckpt err: {e}")
                    self._last_save = nowt
                time.sleep(max(0.5, SAVE_INTERVAL / 2.0))
            except Exception as e:
                log(f"saver err: {e}")
                time.sleep(0.5)

    def is_training_active(self) -> bool:
        return self._active_training.is_set()

# -------------------------
# Controller / REPL (LEARN_MODE behavior preserved and enforced)
# -------------------------
def extract_indicators(text: str) -> List[str]:
    low = text.lower()
    indicators = []
    if "def " in text or re.search(r"\bdef\b", text): indicators.append("function")
    if "class " in text or re.search(r"\bclass\b", text): indicators.append("class")
    if "import " in text: indicators.append("import")
    if "http://" in low or "https://" in low or "www." in low: indicators.append("web")
    if "async " in text or "await " in text: indicators.append("async")
    if "for " in text and " in " in text: indicators.append("loop")
    if "=" in text and "==" not in text: indicators.append("assignment")
    if "sql" in low or "select " in low: indicators.append("sql")
    if "tensor" in low or "numpy" in low or "pytorch" in low or "tensorflow" in low: indicators.append("ml")
    return list(dict.fromkeys(indicators))

def chunk_large_text(tokenizer: ByteTokenizer, text: str, max_tokens: int) -> List[str]:
    pieces = re.split(r'(?<=[\.\?\!]\s)|\n', text)
    chunks = []
    cur = ""
    for p in pieces:
        candidate = (cur + " " + p).strip() if cur else p.strip()
        ids = tokenizer.encode(candidate, add_bos=True, add_eos=True, max_len=max_tokens)
        if len(ids) <= max_tokens:
            cur = candidate
        else:
            if cur:
                chunks.append(cur.strip())
            words = re.split(r'(\s+)', p)
            cur2 = ""
            for w in words:
                cand2 = (cur2 + w).strip()
                ids2 = tokenizer.encode(cand2, add_bos=True, add_eos=True, max_len=max_tokens)
                if len(ids2) <= max_tokens:
                    cur2 = cand2
                else:
                    if cur2:
                        chunks.append(cur2.strip())
                    tr = tokenizer.decode(tokenizer.encode(w, add_bos=True, add_eos=True, max_len=max_tokens)[:max_tokens-1])
                    chunks.append(tr)
                    cur2 = ""
            cur = cur2
    if cur:
        chunks.append(cur.strip())
    return [c for c in chunks if c]

class KRIXHyper:
    def __init__(self, model_path: Optional[Path] = None, auto_fetch: bool=True, no_bg: bool=False):
        def print_loading(percent: int, msg: str):
            sys.stdout.write(f"\rKRIX loading {percent}% — {msg}" + " " * 10 + "\n")
            sys.stdout.flush()

        print_loading(5, "creating dirs")
        ensure_dirs()

        print_loading(20, "initializing datastore")
        DATASTORE_WINDOWS.mkdir(parents=True, exist_ok=True)
        self.store = DataStore(DATASTORE_FILE)

        print_loading(40, "initializing model manager")
        model_train_enabled = not no_bg and (not FAST_MODE)
        self.mgr = ModelManager(train_enabled=model_train_enabled)

        print_loading(55, "configuring background engine")
        self.bg = BackgroundEngine(self.store, self.mgr, enabled=(not no_bg))

        self.exec = ThreadPoolExecutor(max_workers=4)
        self.auto_fetch_enabled = auto_fetch

        # generation coordination
        self._gen_stop_event = threading.Event()
        self._gen_lock = threading.Lock()
        self._generation_active = threading.Event()

        # safety and modes
        self.safe_mode = SAFE_MODE_DEFAULT
        self.LEARN_MODE = LEARN_ENV
        self.TRAINING_MODE = False
        self.BLAST_MODE = False

        print_loading(70, "loading checkpoint if provided")
        if model_path:
            try:
                loaded = self.mgr.load_checkpoint(model_path)
                print_loading(80, f"checkpoint {'loaded' if loaded else 'not found'}")
            except Exception as e:
                log(f"checkpoint load err: {e}")

        print_loading(85, "starting background engine")
        self.bg.start()

        threading.Thread(target=self._compute_missing_embeddings_loop, daemon=True).start()

        if self.auto_fetch_enabled:
            threading.Thread(target=self._auto_fetch_seeds, daemon=True).start()

        threading.Thread(target=self._blast_train_loop, daemon=True).start()

        log("KRIXHyper initialized")
        print_loading(100, "initialized")

    def _compute_missing_embeddings_loop(self):
        while True:
            try:
                items = self.store.list_items()
                missing = [it for it in items if not it.get("embedding")]
                if missing:
                    if not self.bg.is_training_active() and not self.bg._cuda_memory_high():
                        for it in missing:
                            try:
                                emb = self.mgr.get_embedding(it["text"])
                                self.store.update_embedding(it["id"], emb)
                                try:
                                    self.bg.assign_topic_for_item(it["id"], emb, it["text"])
                                except Exception:
                                    pass
                            except Exception as e:
                                log(f"compute_missing_embeddings err: {e}")
                                time.sleep(0.05)
                    else:
                        time.sleep(1.0)
                else:
                    time.sleep(5.0)
            except Exception as e:
                log(f"_compute_missing_embeddings_loop err: {e}")
                time.sleep(2.0)

    def _auto_fetch_seeds(self):
        # when auto_fetch is enabled, only fetch english-clean seeds
        try:
            self.bg.pause(block=True, timeout=5.0)
        except Exception:
            pass

        while getattr(self, "auto_fetch_enabled", False):
            try:
                urls = SEED_URLS[:AUTO_FETCH_MAX_PAGES_PER_CYCLE]
                for u in urls:
                    try:
                        ok = fetch_and_chunk_to_bg(u, self.bg)
                        log(f"seed fetch {u} ok={ok}")
                        time.sleep(0.6)
                    except Exception as e:
                        log(f"seed fetch err: {e}")
                try:
                    self.bg.resume()
                except Exception:
                    pass
                for _ in range(max(1, int(AUTO_FETCH_INTERVAL // 2))):
                    if not getattr(self, "auto_fetch_enabled", False):
                        break
                    time.sleep(2)
                self.bg.pause(block=False)
            except Exception as e:
                log(f"_auto_fetch_seeds loop err: {e}")
                time.sleep(1.0)
        try:
            self.bg.resume()
        except Exception:
            pass

    def _blast_train_loop(self):
        while True:
            try:
                if not self.BLAST_MODE:
                    time.sleep(2.0)
                    continue
                for cycle in range(0, 6):
                    for u in SEED_URLS[:min(len(SEED_URLS), AUTO_FETCH_MAX_PAGES_PER_CYCLE)]:
                        try:
                            ok = fetch_and_chunk_to_bg(u, self.bg, max_chars=40000, chunk_chars=600)
                            log(f"blast fetch {u} ok={ok}")
                            time.sleep(0.4)
                        except Exception as e:
                            log(f"blast fetch err: {e}")
                    time.sleep(0.8)
                time.sleep(60)
            except Exception as e:
                log(f"_blast_train_loop err: {e}")
                time.sleep(1.0)

    def shutdown(self):
        try:
            self.auto_fetch_enabled = False
            self.BLAST_MODE = False
            self.bg.stop()
            time.sleep(0.3)
            try:
                self.mgr.save_checkpoint(CKPT_PATH)
            except Exception:
                pass
            self.store.save(force=True)
            log("shutdown")
        except Exception as e:
            log(f"shutdown err: {e}")

    def status(self) -> str:
        parts = [
            f"KRIX : HyperCloud status @ {now_iso()}",
            f"Device: {self.mgr.device}  FP16:{self.mgr.use_fp16}",
            f"Pretrained loaded: {self.mgr.pretrained}",
            f"Background trainer alive: {self.bg.trainer_thread.is_alive() and self.bg.enabled}",
            f"Training active right now: {self.bg.is_training_active()}",
            f"Generation active: {self._generation_active.is_set()}",
            f"Queue size: {self.bg.train_q.qsize()} high:{self.bg.high_train_q.qsize()}",
            f"Stored items: {len(self.store.list_items())}",
            f"Topics: {len(self.store.get_topics().keys())}",
            f"Last train loss: {self.mgr.last_loss}",
            f"Safe mode: {self.safe_mode}",
            f"Auto-fetch: {self.auto_fetch_enabled}",
            f"FAST_MODE: {FAST_MODE}",
            f"LEARN_MODE: {self.LEARN_MODE}",
            f"TRAINING_MODE: {self.TRAINING_MODE}",
            f"BLAST_MODE: {self.BLAST_MODE}"
        ]
        return "\n".join(parts)

    def _prep_rag(self, prompt: str, k: int = RAG_K) -> List[str]:
        items = self.store.list_items()
        if not items: return []
        try:
            qv = np.array(self.mgr.get_embedding(prompt), dtype=np.float32)
            hits = knn_search(qv, items, k=k)
            contexts = [it["text"] for score,it in hits if it.get("text")]
            trimmed = []
            chars = 0
            for c in contexts:
                c2 = c if len(c) <= 800 else c[:800]
                if chars + len(c2) > 1200:
                    break
                trimmed.append(c2)
                chars += len(c2)
            return trimmed
        except Exception as e:
            log(f"prep_rag err: {e}")
            return []

    def force_stop_generation(self):
        self._gen_stop_event.set()
        start = time.time()
        while self._generation_active.is_set() and (time.time() - start) < 1.0:
            time.sleep(0.02)
        self._gen_stop_event.clear()
        return True

    def _print_thinking_progress(self, step:int, max_steps:int, spinner_index:int, tag:Optional[str]=None):
        pct = min(100, int(step / max_steps * 100)) if max_steps>0 else 0
        sp = ["-","/","\\","|"]
        ch = sp[spinner_index % len(sp)]
        tagtxt = f"[{tag}]" if tag else ""
        sys.stdout.write(f"\r{tagtxt} [thinking, {pct}% done. Please wait.{ch}]")
        sys.stdout.flush()

    def _simple_indicator(self, text: str):
        try:
            sys.stdout.write(f"\r[KRIX STATUS] {text}" + " " * 20 + "\n")
            sys.stdout.flush()
        except Exception:
            pass

    def handle(self, text: str) -> Optional[str]:
        t = text.strip()
        if not t: return None

        # Commands
        if t.startswith("/"):
            parts = t[1:].split(" ",1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts)>1 else ""
            if cmd in ("status","stat"): return self.status()
            if cmd == "fetch" and arg:
                ok = fetch_and_chunk_to_bg(arg, self.bg)
                return f"KRIX : queued fetch {arg} (ok={ok})"
            if cmd == "autofetch_on":
                self.auto_fetch_enabled = True
                threading.Thread(target=self._auto_fetch_seeds, daemon=True).start()
                return "KRIX : auto-fetch enabled"
            if cmd == "autofetch_off":
                self.auto_fetch_enabled = False
                return "KRIX : auto-fetch disabled"
            if cmd == "last10":
                items = self.store.get_recent_items_last_minutes(10)
                if not items:
                    return "KRIX : nothing learned in the last 10 minutes"
                lines = []
                for it in items:
                    ind = ",".join(it.get("indicators",[])) if it.get("indicators") else "-"
                    preview = it.get("text","")[:140].replace("\n"," ")
                    lines.append(f"{it.get('ts')} [{ind}] {preview}")
                return "KRIX : learned (last 10m):\n" + "\n".join(lines)
            if cmd == "topics":
                topics = self.store.get_topics()
                if not topics: return "KRIX : (no topics yet)"
                lines = [f"{tid}: {info.get('label')[:80]} (count={info.get('count',0)})" for tid,info in topics.items()]
                return "KRIX : topics:\n" + "\n".join(lines)
            if cmd == "save":
                self.mgr.save_checkpoint(CKPT_PATH); self.store.save(force=True)
                return "KRIX : saved checkpoint & datastore"
            if cmd in ("exit","quit"):
                self.shutdown(); sys.exit(0)
            if cmd == "forcestop":
                ok = self.force_stop_generation()
                return "KRIX : force-stop sent" if ok else "KRIX : force-stop failed"
            if cmd == "learn":
                content = arg.strip()
                if not content:
                    return "KRIX : /learn requires text: /learn <your giant text here>"
                chunks = chunk_large_text(self.mgr.tokenizer, content, max_tokens=self.mgr.tokenizer.block_size-2)
                enqueued = 0
                for i, c in enumerate(chunks, 1):
                    inds = extract_indicators(c)
                    iid = self.store.add_item(c, src="user_learn", embedding=None, indicators=inds)
                    self.bg.queue_training(c, item_id=iid, priority=True)
                    enqueued += 1
                self.store.push_history("user", "/learn " + (content[:120] + ("..." if len(content)>120 else "")))
                return f"KRIX : accepted and queued {enqueued} chunk(s) for training (via /learn)"
            if cmd == "safe_on":
                self.safe_mode = True
                return "KRIX : safe-mode enabled"
            if cmd == "safe_off":
                self.safe_mode = False
                return "KRIX : safe-mode disabled"
            if cmd == "learn_mode_on":
                self.LEARN_MODE = True
                return "KRIX : LEARN_MODE enabled (every message will be queued for training)"
            if cmd == "learn_mode_off":
                self.LEARN_MODE = False
                return "KRIX : LEARN_MODE disabled"
            if cmd == "training_on":
                self.TRAINING_MODE = True
                return "KRIX : TRAINING_MODE enabled (KRIX will respond to commands only)"
            if cmd == "training_off":
                self.TRAINING_MODE = False
                return "KRIX : TRAINING_MODE disabled"
            if cmd == "blast_on":
                self.BLAST_MODE = True
                return "KRIX : BLAST_MODE enabled (aggressive auto-seeding)"
            if cmd == "blast_off":
                self.BLAST_MODE = False
                return "KRIX : BLAST_MODE disabled"
            return f"KRIX : unknown command: {cmd}"

        # If TRAINING_MODE: queue for learning automatically (no conversational reply)
        if self.TRAINING_MODE:
            chunks = chunk_large_text(self.mgr.tokenizer, t, max_tokens=self.mgr.tokenizer.block_size-2)
            for c in chunks:
                iid = self.store.add_item(c, src="user_learn", embedding=None, indicators=extract_indicators(c))
                self.bg.queue_training(c, item_id=iid, priority=True)
            self.store.push_history("user", t)
            return "KRIX : in TRAINING_MODE — input queued for learning (no conversational reply)."

        # If LEARN_MODE: everything typed is queued for high-priority training (no /learn: manual needed)
        if self.LEARN_MODE:
            chunks = chunk_large_text(self.mgr.tokenizer, t, max_tokens=self.mgr.tokenizer.block_size-2)
            for c in chunks:
                iid = self.store.add_item(c, src="user_learn", embedding=None, indicators=extract_indicators(c))
                # high priority ensures fast learning of user style
                self.bg.queue_training(c, item_id=iid, priority=True)
                try:
                    analysis = analyze_grammar_and_patterns(c)
                    meta_text = json.dumps({"src_chunk_preview": c[:160], "analysis": analysis}, ensure_ascii=False)
                    self.store.add_item(meta_text, src="grammar_meta", embedding=None, indicators=["grammar","pattern"])
                except Exception as e:
                    log(f"learn_mode grammar analysis err: {e}")
            self.store.push_history("user", t)
            return "KRIX : LEARN_MODE active — text queued for learning. (krix learning…)"

        # Safe-mode filter
        if self.safe_mode and contains_unsafe(t):
            self.store.push_history("user", t)
            resp = "KRIX : request refused — content triggers safe-mode filters."
            self.store.push_history("ai", resp)
            return resp

        # Try to fetch authoritative info first (only clean English references)
        try:
            self._simple_indicator("krix searching — scanning web sources for similar content...")
            ref = try_fetch_reference(t)
            if ref:
                title = ref.get("title") or ref.get("source")
                snippet = ref.get("snippet", "")[:2000]
                source = ref.get("source", "unknown")
                prompt = f"Reference material:\n{snippet}\n\nSource: {source}\n\nUser question: {t}\n\nUsing the reference above, answer concisely and in a friendly tone, customizing wording to the user's style. Keep the answer short and add a note saying you used web reference. Do not invent facts beyond the reference."
                with self._gen_lock:
                    if self._generation_active.is_set():
                        out = f"{snippet}\n\n[source: {source} | title: {title}]"
                        self.store.push_history("user", t)
                        self.store.push_history("ai", out)
                        iid = self.store.add_item(out, src="reference", embedding=None, indicators=["reference","wiki"])
                        return f"KRIX : {out}"
                    self._generation_active.set()
                self.bg.pause(block=True, timeout=5.0)
                gen_stop = self._gen_stop_event
                self._simple_indicator("krix analyzing — customizing web content with model.")
                gen = self.mgr.generate_stream(prompt, max_new_tokens=min(220, GEN_MAX_TOKENS), temperature=0.6, top_k=TOP_K, top_p=TOP_P, rag_context=None, stop_event=gen_stop)
                last_partial = ""
                reply_text = ""
                start_time = time.time()
                last_yield_time = start_time
                spinner_idx = 0
                try:
                    for pair in gen:
                        if not isinstance(pair, tuple):
                            partial = str(pair)
                            step = 0
                        else:
                            partial, step = pair
                        nowt = time.time()
                        if partial is not None:
                            last_yield_time = nowt
                            if len(partial) > len(last_partial):
                                last_partial = partial
                            spinner_idx += 1
                            self._print_thinking_progress(step if step else 1, GEN_MAX_TOKENS, spinner_idx, tag="krix analyzing")
                            if len(last_partial) >= MAX_GEN_CHARS:
                                last_partial = last_partial[:MAX_GEN_CHARS]
                                gen_stop.set()
                                break
                        if nowt - start_time > GEN_TIMEOUT:
                            gen_stop.set()
                            reply_text = f"[generation timeout after {GEN_TIMEOUT} seconds]"
                            log(f"generation timeout for prompt: {t[:120]}")
                            break
                        if nowt - last_yield_time > GEN_TIMEOUT:
                            gen_stop.set()
                            reply_text = f"[generation inactivity timeout after {GEN_TIMEOUT} seconds]"
                            log(f"generation inactivity timeout for prompt: {t[:120]}")
                            break
                    if not reply_text:
                        reply_text = last_partial.strip()
                    if reply_text and len(reply_text) > MAX_GEN_CHARS:
                        reply_text = reply_text[:MAX_GEN_CHARS]
                    if not reply_text:
                        reply_text = f"[KRIX produced no output — generation ended empty or was interrupted after {GEN_TIMEOUT}s]"
                        log(f"empty generation for prompt: {t[:120]}")
                    self.store.push_history("user", t)
                    reply_text = reply_text + f"\n\n[source: {source} | title: {title}]"
                    self.store.push_history("ai", reply_text)
                    iid = self.store.add_item(reply_text, src="generated_web", embedding=None, indicators=extract_indicators(reply_text))
                    def postproc():
                        try:
                            emb = self.mgr.get_embedding(reply_text)
                            self.store.update_embedding(iid, emb)
                            try:
                                self.bg.assign_topic_for_item(iid, emb, reply_text)
                            except Exception:
                                pass
                            combo = t + "\n" + reply_text
                            self.mgr.train_on_texts([combo], epochs=TRAIN_EPOCHS_QUICK, batch_size=1)
                            if not FAST_MODE:
                                self.mgr.save_checkpoint(CKPT_PATH)
                            self.store.save(force=False)
                        except Exception as e:
                            log(f"postproc err: {e}")
                    try:
                        self.exec.submit(postproc)
                    except Exception:
                        pass
                except Exception as e:
                    log(f"generation err: {e}")
                    reply_text = f"[generation error: {e}]"
                    self.store.push_history("ai", reply_text)
                finally:
                    with self._gen_lock:
                        self._generation_active.clear()
                    self._gen_stop_event.clear()
                    self.bg.resume()
                return f"KRIX : {reply_text}"
        except Exception:
            pass

        # Normal generation path
        with self._gen_lock:
            if self._generation_active.is_set():
                try:
                    stopped = self.force_stop_generation()
                    time.sleep(0.05)
                    if self._generation_active.is_set():
                        return "KRIX : previous generation still active — use /forcestop and try again"
                except Exception:
                    return "KRIX : unable to start generation because another generation is active"
            self._generation_active.set()

        self.store.push_history("user", t)
        rag_ctx = self._prep_rag(t, k=RAG_K)

        self.bg.pause(block=True, timeout=5.0)
        time.sleep(0.01)
        gen_stop = self._gen_stop_event

        self._simple_indicator("krix thinking — generating an answer using model and memory...")
        gen = self.mgr.generate_stream(t, max_new_tokens=GEN_MAX_TOKENS, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, rag_context=rag_ctx, stop_event=gen_stop)
        last_partial = ""
        reply_text = ""
        start_time = time.time()
        last_yield_time = start_time
        spinner_idx = 0
        try:
            for pair in gen:
                if not isinstance(pair, tuple):
                    partial = str(pair)
                    step = 0
                else:
                    partial, step = pair
                nowt = time.time()
                if partial is not None:
                    last_yield_time = nowt
                    if len(partial) > len(last_partial):
                        last_partial = partial
                    spinner_idx += 1
                    self._print_thinking_progress(step if step else 1, GEN_MAX_TOKENS, spinner_idx, tag="krix thinking")
                    if len(last_partial) >= MAX_GEN_CHARS:
                        last_partial = last_partial[:MAX_GEN_CHARS]
                        gen_stop.set()
                        break
                if nowt - start_time > GEN_TIMEOUT:
                    gen_stop.set()
                    reply_text = f"[generation timeout after {GEN_TIMEOUT} seconds]"
                    log(f"generation timeout for prompt: {t[:120]}")
                    break
                if nowt - last_yield_time > GEN_TIMEOUT:
                    gen_stop.set()
                    reply_text = f"[generation inactivity timeout after {GEN_TIMEOUT} seconds]"
                    log(f"generation inactivity timeout for prompt: {t[:120]}")
                    break
            sys.stdout.write("\r" + " " * 80 + "\r"); sys.stdout.flush()

            if not reply_text:
                reply_text = last_partial.strip()
            if reply_text and len(reply_text) > MAX_GEN_CHARS:
                reply_text = reply_text[:MAX_GEN_CHARS]
            if not reply_text:
                reply_text = f"[KRIX produced no output — generation ended empty or was interrupted after {GEN_TIMEOUT}s]"
                log(f"empty generation for prompt: {t[:120]}")

            try:
                if rag_ctx:
                    ctx_preview = " | ".join([c[:80].replace("\n"," ") for c in rag_ctx[:3]])
                    reply_text = reply_text + ("\n\n[context used: " + ctx_preview + "]" if ctx_preview else "")
            except Exception:
                pass

            self.store.push_history("ai", reply_text)
            iid = self.store.add_item(reply_text, src="generated", embedding=None, indicators=extract_indicators(reply_text))
            def postproc():
                try:
                    emb = self.mgr.get_embedding(reply_text)
                    self.store.update_embedding(iid, emb)
                    try:
                        self.bg.assign_topic_for_item(iid, emb, reply_text)
                    except Exception:
                        pass
                    combo = t + "\n" + reply_text
                    self.mgr.train_on_texts([combo], epochs=TRAIN_EPOCHS_QUICK, batch_size=1)
                    if not FAST_MODE:
                        self.mgr.save_checkpoint(CKPT_PATH)
                    self.store.save(force=False)
                except Exception as e:
                    log(f"postproc err: {e}")
            try:
                self.exec.submit(postproc)
            except Exception:
                pass

        except Exception as e:
            log(f"generation err: {e}")
            reply_text = f"[generation error: {e}]"
            self.store.push_history("ai", reply_text)
        finally:
            with self._gen_lock:
                self._generation_active.clear()
            self._gen_stop_event.clear()
            self.bg.resume()

        return f"KRIX : {reply_text}"

# -------------------------
# CLI / REPL
# -------------------------
BANNER = r"""
===============================================
  KRIX — HYPERCLOUD INTELLIGENCE (CLEAN-ENGLISH)
  Commands:
    /status /fetch /autofetch_on /autofetch_off /last10 /topics /save /forcestop
    /learn <text> /learn_mode_on /learn_mode_off /training_on /training_off
    /blast_on /blast_off /safe_on /safe_off /exit
  Notes:
    - When LEARN_MODE is ON, every text you type is queued and trained automatically.
    - Background fetches only accept English-clean pages.
===============================================
"""

def repl(model_path: Optional[str]=None, auto_fetch: bool=True, no_bg: bool=False):
    try:
        kh = KRIXHyper(model_path=Path(model_path) if model_path else None, auto_fetch=auto_fetch, no_bg=no_bg)
    except Exception as e:
        print("KRIX initialization error:")
        traceback.print_exc()
        log(f"KRIX init failed: {e}")
        return
    print(BANNER)
    try:
        while True:
            try:
                txt = input("ask: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nShutting down KRIX.")
                try: kh.shutdown()
                except Exception: pass
                break
            if not txt:
                continue
            out = kh.handle(txt)
            if out:
                print(out)
    except Exception as e:
        print("Runtime error in REPL:")
        traceback.print_exc()
        try:
            kh.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to model checkpoint", default=None)
    parser.add_argument("--no-auto-fetch", action="store_true", help="disable auto fetch")
    args = parser.parse_args()
    repl(model_path=args.model, auto_fetch=not args.no_auto_fetch, no_bg=False)
