"""
forge.py — forge.py v6.5
Run:  python forge.py

v6.5: RECORD VOICE TO .WAV — dedicated recorder in the Audio menu.
Free-form: press Enter, speak, stop, keep. Saves .wav + optional .txt caption.
"""

import glob, json, math, os, random, re, secrets, shutil, sys, threading, time
import torch, torch.nn as nn, torch.nn.functional as F

VERSION = "6.5"


# ═══════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
SETTINGS_FILE = "settings.json"
DEFAULTS = {
    "temperature": 0.8, "top_k": 40, "max_new_tokens": 150,
    "batch_size": 16, "learning_rate": 0.0006, "device": "auto",
    "block_size": 128, "memory_turns": 6, "persona": "",
    "banned_words": [], "seed": -1,
    "thinking_style": "brief", "learning_think": True,
    "mic_threshold": 0.015,
    "mic_silence_ms": 700,
    "mic_max_sec": 8,
    "subtitle_window": 1.0,
    "subtitle_hop": 0.2,
    "subtitle_stable": 2,
    "tts_rate": 170,
    "record_folder": "samples/audio",
}


def load_json(p, d):
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return d


def save_json(p, d):
    json.dump(d, open(p, "w"), indent=2)


CFG = {**DEFAULTS, **load_json(SETTINGS_FILE, {})}


def save_cfg():
    save_json(SETTINGS_FILE, CFG)


def get_device():
    if CFG["device"] == "cpu":
        return torch.device("cpu")
    if CFG["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_seed():
    if CFG["seed"] >= 0:
        torch.manual_seed(CFG["seed"])
        random.seed(CFG["seed"])


def filter_output(text):
    for w in CFG["banned_words"]:
        if w:
            text = re.sub(re.escape(w), "*" * len(w), text, flags=re.IGNORECASE)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════
class BPE:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def _stats(self, ids):
        c = {}
        for p in zip(ids, ids[1:]):
            c[p] = c.get(p, 0) + 1
        return c

    def _merge(self, ids, pair, idx):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(idx); i += 2
            else:
                out.append(ids[i]); i += 1
        return out

    def train(self, text, target, min_tokens):
        ids = list(text.encode("utf-8"))
        merges, vocab = {}, {i: bytes([i]) for i in range(256)}
        for i in range(max(0, target - 256)):
            if len(ids) <= min_tokens:
                break
            stats = self._stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self._merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.merges, self.vocab = merges, vocab
        return self

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = self._stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        try:
            return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
        except KeyError:
            return ""

    def to_list(self):
        return [list(p) for p in self.merges.keys()]

    def from_list(self, ml):
        self.merges, self.vocab = {}, {i: bytes([i]) for i in range(256)}
        for i, item in enumerate(ml):
            idx = 256 + i
            if len(item) == 2 and isinstance(item[0], int):
                a, b = item
            elif len(item) == 2 and isinstance(item[0], (list, tuple)):
                a, b = item[0]
            else:
                continue
            self.merges[(a, b)] = idx
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        return self


# ═══════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════
class _Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TextLM(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4,
                 block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([_Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class ImageGen(nn.Module):
    def __init__(self, vocab_size, n_embd=64, hidden=128, latent=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.gru = nn.GRU(n_embd, hidden, batch_first=True)
        self.to_z = nn.Sequential(nn.Linear(hidden, latent), nn.LayerNorm(latent), nn.ReLU())
        self.from_z = nn.Linear(latent, 256 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid())

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids):
        _, h = self.gru(self.tok_emb(ids))
        return self.dec(self.from_z(self.to_z(h[-1])).view(-1, 256, 4, 4))


class AudioGen(nn.Module):
    SR = 16000
    def __init__(self, vocab_size, n_embd=64, hidden=128, latent=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.gru = nn.GRU(n_embd, hidden, batch_first=True)
        self.to_z = nn.Sequential(nn.Linear(hidden, latent), nn.LayerNorm(latent), nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(latent, 512), nn.ReLU(), nn.Linear(512, 2048), nn.ReLU(),
            nn.Linear(2048, self.SR), nn.Tanh())

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids):
        _, h = self.gru(self.tok_emb(ids))
        return self.dec(self.to_z(h[-1]))


class AudioToText(nn.Module):
    def __init__(self, vocab_size, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 16, 8, 4), nn.ReLU(),
            nn.Conv1d(32, 64, 8, 4, 2), nn.ReLU(),
            nn.Conv1d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.to_h = nn.Linear(128, hidden)
        self.emb = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, 2, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def enc(self, w):
        return self.to_h(self.conv(w.unsqueeze(1)).squeeze(-1))

    def forward(self, w, c):
        h0 = self.enc(w).unsqueeze(0).repeat(2, 1, 1)
        out, _ = self.gru(self.emb(c[:, :-1]), h0)
        logits = self.head(out)
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       c[:, 1:].reshape(-1), ignore_index=0)


class ImageCaption(nn.Module):
    def __init__(self, vocab_size, hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.to_h = nn.Linear(256, hidden)
        self.emb = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, 2, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def enc(self, im):
        return self.to_h(self.cnn(im).flatten(1))

    def forward(self, im, c):
        h0 = self.enc(im).unsqueeze(0).repeat(2, 1, 1)
        out, _ = self.gru(self.emb(c[:, :-1]), h0)
        logits = self.head(out)
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       c[:, 1:].reshape(-1), ignore_index=0)


MODELS = {"text": TextLM, "image": ImageGen, "tts": AudioGen,
          "stt": AudioToText, "caption": ImageCaption}


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT I/O
# ═══════════════════════════════════════════════════════════════════════════
def save_ckpt(path, mode, model, tok, config, extra=None, optimizer=None, step=0):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = {"mode": mode, "version": VERSION, "model": model.state_dict(),
         "tokenizer_merges": tok.to_list(), "config": config, "training_step": step}
    if optimizer is not None:
        d["optimizer"] = optimizer.state_dict()
    if extra:
        d.update(extra)
    torch.save(d, path)


def load_ckpt(path, device, need_optimizer=False):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    tok = BPE().from_list(ckpt["tokenizer_merges"])
    model = MODELS[ckpt["mode"]](vocab_size=len(tok.vocab), **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, tok, ckpt


# ═══════════════════════════════════════════════════════════════════════════
# CONVERSATION
# ═══════════════════════════════════════════════════════════════════════════
CHATS_DIR = "chats"


class Conversation:
    def __init__(self, persona=""):
        self.persona = persona
        self.messages = []
        self.started = time.strftime("%Y-%m-%d %H:%M:%S")

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})

    def build_prompt(self, new_input, dialogue_mode, max_turns, thought=None):
        parts = []
        if self.persona:
            parts.append(f"System: {self.persona}")
        recent = self.messages[-max_turns * 2:] if max_turns > 0 else []
        for m in recent:
            tag = "User" if m["role"] == "user" else "Bot"
            parts.append(f"{tag}: {m['content']}")
        if dialogue_mode:
            if thought:
                parts.append(f"User: {new_input}\nThink: {thought}\nBot:")
            else:
                parts.append(f"User: {new_input}\nBot:")
        else:
            parts.append(new_input)
        return "\n".join(parts)

    def save(self):
        os.makedirs(CHATS_DIR, exist_ok=True)
        fn = os.path.join(CHATS_DIR, f"chat_{int(time.time() * 1000)}.json")
        save_json(fn, {"started": self.started, "persona": self.persona,
                       "messages": self.messages})
        return fn


# ═══════════════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════════════
def format_prompt(user, style, dialogue_mode):
    if style == "chat" and dialogue_mode:
        return f"User: {user}\nBot:"
    if style == "chat":
        return user
    if style == "qa":
        return f"Q: {user}\nA:"
    if style == "story":
        return f"Story about {user}:\n\nOnce upon a time"
    if style == "poem":
        return f"Poem about {user}:\n\n"
    return user


@torch.no_grad()
def generate_text(model, tok, prompt, block_size, temp=None, top_k=None,
                  max_new=None, stop_chars="\n"):
    temp = temp if temp is not None else CFG["temperature"]
    k = top_k if top_k is not None else CFG["top_k"]
    mx = max_new if max_new is not None else CFG["max_new_tokens"]
    ids = tok.encode(prompt)
    if not ids:
        return ""
    idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    out = []
    for _ in range(mx):
        logits, _ = model(idx[:, -block_size:])
        logits = logits[:, -1, :] / max(temp, 0.01)
        if k:
            v, _ = torch.topk(logits, min(k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        idx = torch.cat([idx, nxt], 1)
        out.append(nxt.item())
        s = tok.decode(out)
        if stop_chars and any(c in s for c in stop_chars):
            for c in stop_chars:
                if c in s:
                    return s.split(c)[0].strip()
            return s.strip()
    return tok.decode(out).strip()


def real_thought(model, tok, user_input, block_size):
    prompt = f"User: {user_input}\nThink:"
    ids = tok.encode(prompt)
    if not ids:
        return None
    try:
        out = generate_text(model, tok, prompt, block_size,
                            temp=0.7, top_k=20, max_new=80,
                            stop_chars=["\n", "Bot:"])
        out = out.replace("Think:", "").strip()
        return out if len(out) >= 3 else None
    except Exception:
        return None


def run_with_spinner(label, fn):
    result = {"v": None, "err": None}

    def _work():
        try:
            result["v"] = fn()
        except Exception as e:
            result["err"] = e

    t = threading.Thread(target=_work)
    t.start()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    start = time.time()
    while t.is_alive():
        print(f"\r    {frames[i % len(frames)]} {label}", end="", flush=True)
        time.sleep(0.05)
        i += 1
    t.join()
    elapsed = time.time() - start
    print(f"\r    ✓ {label}  ({elapsed:.2f}s)          ")
    if result["err"]:
        raise result["err"]
    return result["v"]


# ═══════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def clear():
    os.system("cls" if os.name == "nt" else "clear")


def header(title):
    print("\n" + "═" * 56)
    print(f"  {title}")
    print("═" * 56)


def ask(question, options, subtitle=None):
    print()
    if subtitle:
        print(f"  {subtitle}")
    print(f"  {question}")
    print()
    for i, o in enumerate(options, 1):
        print(f"    {i}) {o}")
    print()
    while True:
        try:
            s = input(f"  Choose 1-{len(options)} ('b' = back) > ").strip().lower()
            if s in ("b", "back", "exit", "quit"):
                return -1
            n = int(s)
            if 1 <= n <= len(options):
                return n
        except (ValueError, EOFError):
            pass
        print(f"  (please type 1 to {len(options)})")


def ask_path(prompt, default):
    print()
    print(f"  {prompt}")
    print(f"  (blank = '{default}')")
    s = input("  > ").strip().strip('"').strip("'")
    return s if s else default


def ask_number(prompt, default, lo, hi, kind=float):
    print()
    print(f"  {prompt}")
    print(f"  (current: {default}, range: {lo} to {hi})")
    while True:
        s = input("  > ").strip()
        if not s:
            return default
        try:
            v = kind(s)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  (please type {lo} to {hi})")


def ask_text(prompt, default=""):
    print()
    print(f"  {prompt}")
    if default:
        print(f"  (blank = '{default}')")
    s = input("  > ").strip()
    return s if s else default


def pause():
    try:
        input("\n  (Enter to continue...)")
    except (EOFError, KeyboardInterrupt):
        pass


def progress(step, total, loss, extra=""):
    pct = step / total
    bars = int(pct * 30)
    bar = "█" * bars + "░" * (30 - bars)
    print(f"\r  [{bar}] {step}/{total}  loss={loss:.3f} {extra}", end="", flush=True)


def human_size(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def discover(paths, exts):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for e in exts:
                out.extend(glob.glob(os.path.join(p, "**", f"*{e}"), recursive=True))
        elif any(c in p for c in "*?["):
            out.extend(glob.glob(p, recursive=True))
        elif os.path.exists(p):
            out.append(p)
    return sorted(set(out))


def find_files_with_ext(exts):
    results = []
    for ext in exts:
        results.extend(glob.glob(f"**/*{ext}", recursive=True))
    return sorted(set(results))


PRESETS = {
    1: ("Tiny     (seconds)", 300),
    2: ("Quick    (a minute or two)", 1000),
    3: ("Normal   (a few minutes)", 3000),
    4: ("Long     (10-30 min)", 10000),
    5: ("Extreme  (1+ hour)", 50000),
    6: ("Custom   (choose yourself)", None),
}


def ask_preset():
    c = ask("How long should training take?", [n for n, _ in PRESETS.values()])
    if c == -1:
        return None
    _, steps = PRESETS[c]
    if steps is None:
        steps = ask_number("How many steps?", 3000, 10, 500000, int)
    return steps


# ═══════════════════════════════════════════════════════════════════════════
# AUDIO HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _normalize_audio(audio):
    import numpy as np
    if len(audio) == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak > 0.01:
        return (audio / peak) * 0.8
    return audio


def trim_silence(audio, threshold=0.01, pad_ms=40):
    import numpy as np
    if len(audio) == 0:
        return audio
    env = np.abs(audio)
    above = np.where(env > threshold)[0]
    if len(above) == 0:
        return audio
    pad = int(len(audio) * pad_ms / 1000)
    start = max(0, above[0] - pad)
    end = min(len(audio), above[-1] + pad)
    return audio[start:end]


def _resample_to_16k(audio, sr):
    import numpy as np
    if sr == 16000:
        return audio.astype("float32")
    r = sr / 16000
    L = max(1, int(len(audio) / r))
    ix = (np.arange(L) * r).astype(int)
    ix = ix[ix < len(audio)]
    return audio[ix].astype("float32")


def _fit_to_1s(audio, sr=16000):
    import numpy as np
    if len(audio) == 0:
        return np.zeros(sr, dtype="float32")
    if len(audio) > sr:
        return audio[:sr]
    return np.pad(audio, (0, sr - len(audio)))


def _generate_via_pyttsx3(phrase, out_path, rate=170):
    try:
        import pyttsx3
    except ImportError:
        return False
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.setProperty('volume', 0.9)
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        engine.save_to_file(phrase, out_path)
        engine.runAndWait()
        return os.path.exists(out_path) and os.path.getsize(out_path) > 200
    except Exception:
        return False


def generate_speech_samples(phrases, rate=None):
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("  pip install soundfile")
        return None

    rate = rate if rate is not None else CFG.get("tts_rate", 170)
    folder = "samples/audio"
    os.makedirs(folder, exist_ok=True)
    SR = 16000

    def safe_name(s):
        return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_') or "sample"

    print(f"  generating {len(phrases)} speech samples at rate {rate}...")
    count = 0
    for i, phrase in enumerate(phrases):
        name = safe_name(phrase)
        wav_path = os.path.join(folder, f"{name}.wav")
        print(f"\r  [{i+1}/{len(phrases)}] {phrase:<40}", end="", flush=True)
        tmp = os.path.join(folder, f"_tmp_{name}.wav")
        ok = _generate_via_pyttsx3(phrase, tmp, rate=rate)
        if not ok or not os.path.exists(tmp):
            print(f"\r  [{i+1}/{len(phrases)}] {phrase:<40} SKIP", flush=True)
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass
            continue
        try:
            d, sr = sf.read(tmp)
            if d.ndim > 1:
                d = d.mean(axis=1)
            d = _resample_to_16k(d, sr)
            d = trim_silence(d, threshold=0.01)
            d = _normalize_audio(d)
            d = _fit_to_1s(d, SR)
            sf.write(wav_path, d, SR)
            with open(os.path.join(folder, f"{name}.txt"), "w") as f:
                f.write(phrase)
            count += 1
        except Exception:
            pass
        finally:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass

    print()
    if count > 0:
        print(f"  ✓ created {count} real speech samples in {folder}/")
        return folder
    return None


def _generate_synthetic_varied(folder):
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return None
    os.makedirs(folder, exist_ok=True)
    SR = 16000
    t = np.arange(SR) / SR
    samples = [
        ("low_pure_tone", 220, "sine"), ("mid_pure_tone", 440, "sine"),
        ("high_pure_tone", 880, "sine"), ("low_buzz", 220, "square"),
        ("mid_buzz", 440, "square"), ("sawtooth", 330, "saw"),
        ("triangle_wave", 660, "triangle"), ("rising_sweep", (200, 1200), "sweep"),
        ("falling_sweep", (1200, 200), "sweep"), ("am_pulse", 440, "pulse"),
        ("low_chord", [220, 330, 440], "chord"), ("high_chord", [440, 550, 660], "chord"),
        ("fast_warble", 440, "warble"), ("slow_warble", 220, "warble"),
        ("noise_burst", 0, "noise"), ("noise_low", 0, "noise_low"),
    ]
    for name, freq, kind in samples:
        if kind == "sine": wav = 0.5 * np.sin(2 * np.pi * freq * t)
        elif kind == "square": wav = 0.3 * np.sign(np.sin(2 * np.pi * freq * t))
        elif kind == "saw": wav = 0.3 * (2 * ((t * freq) % 1) - 1)
        elif kind == "triangle": wav = 0.4 * (2 * np.abs(2 * ((t * freq) % 1) - 1) - 1)
        elif kind == "sweep":
            f0, f1 = freq
            wav = 0.4 * np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t ** 2))
        elif kind == "pulse":
            wav = 0.4 * np.sin(2 * np.pi * freq * t) * (np.sin(2 * np.pi * 3 * t) > 0)
        elif kind == "chord":
            wav = 0.3 * sum(np.sin(2 * np.pi * f * t) for f in freq)
        elif kind == "warble":
            mod = 1 + 0.3 * np.sin(2 * np.pi * 5 * t)
            wav = 0.4 * np.sin(2 * np.pi * freq * mod * t)
        elif kind == "noise":
            wav = 0.3 * np.random.randn(SR)
        elif kind == "noise_low":
            wav = np.cumsum(np.random.randn(SR))
            wav = wav / max(abs(wav).max(), 1) * 0.3
        else: wav = 0.4 * np.sin(2 * np.pi * 440 * t)
        env = np.ones_like(wav)
        attack = int(SR * 0.02); release = int(SR * 0.15)
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] = np.linspace(1, 0, release)
        wav = wav * env
        sf.write(os.path.join(folder, f"{name}.wav"), wav.astype("float32"), SR)
        with open(os.path.join(folder, f"{name}.txt"), "w") as f:
            f.write(name.replace("_", " "))
    print(f"  created {len(samples)} synthetic samples in {folder}/")
    return folder


def make_sample_audio():
    phrases = [
        "hello", "goodbye", "yes", "no", "please", "thanks",
        "good morning", "good night", "how are you", "help",
        "sorry", "welcome", "excuse me", "see you",
        "one two three", "four five six",
        "the cat sat", "the dog ran",
        "it is raining", "i am happy",
        "open the door", "close the window",
        "what time is it", "where are you",
    ]
    folder = generate_speech_samples(phrases)
    if not folder:
        print()
        print("  ⚠ No TTS available — falling back to synthetic sounds (NOT speech).")
        print("  ⚠ For real speech: pip install pyttsx3, or use 'Record voice to .wav'")
        print()
        folder = _generate_synthetic_varied("samples/audio")
    return folder


def record_own_samples(n_target=20):
    sd = _check_sounddevice()
    if sd is None:
        return None
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("  pip install soundfile")
        return None
    folder = "samples/audio"
    os.makedirs(folder, exist_ok=True)
    SR = 16000
    prompts = [
        "hello", "goodbye", "yes", "no", "please", "thanks",
        "good morning", "good night", "how are you", "what is your name",
        "help me", "i am fine", "see you later",
        "one two three", "four five six",
        "the cat sat", "the dog ran",
        "it is raining", "the sun is bright", "i am happy",
        "open the door", "close the window", "where are you", "thank you very much",
    ][:n_target]
    print()
    print("  ══════════════════════════════════════════════════════════")
    print("  🎙️  GUIDED RECORDING  (training phrases)")
    print("  ══════════════════════════════════════════════════════════")
    print(f"  Recording {len(prompts)} phrases. Press Enter, say the phrase.")
    print()
    count = 0
    existing = find_files_with_ext([".wav"])
    start_index = len([f for f in existing if os.path.basename(f).startswith("own_")])
    for i, phrase in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] Say: \"{phrase}\"")
        try:
            cmd = input("  (Enter to record, 'skip', 'quit') > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd == "quit": break
        if cmd == "skip": continue
        audio = record_until_silence(max_sec=4, silence_ms=800)
        if audio is None:
            print("  (nothing recorded)"); continue
        print("  playing back...")
        play_audio(audio)
        keep = input("  keep? (Y/n/r) > ").strip().lower()
        if keep == "r":
            audio = record_until_silence(max_sec=4, silence_ms=800)
            if audio is None: continue
            play_audio(audio)
            keep = input("  keep? (Y/n) > ").strip().lower()
        if keep.startswith("n"): continue
        name = re.sub(r'[^a-z0-9]+', '_', phrase.lower()).strip('_')
        fname = f"own_{start_index + count:03d}_{name}"
        wav_path = os.path.join(folder, f"{fname}.wav")
        audio = trim_silence(audio, threshold=0.01)
        audio = _normalize_audio(audio)
        audio = _fit_to_1s(audio, SR)
        sf.write(wav_path, audio.astype("float32"), SR)
        with open(os.path.join(folder, f"{fname}.txt"), "w") as f:
            f.write(phrase)
        count += 1
        print(f"  ✓ saved: {wav_path}")
    print(f"\n  ✓ recorded {count} sample(s)")
    return folder if count > 0 else None


# ═══════════════════════════════════════════════════════════════════════════
# ✨ NEW: RECORD VOICE TO .WAV (free-form recorder) ✨
# ═══════════════════════════════════════════════════════════════════════════
def record_to_wav_session():
    """Free-form voice recorder. Speak, save to .wav + optional .txt caption."""
    sd = _check_sounddevice()
    if sd is None:
        return
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("  pip install soundfile")
        return

    folder = CFG.get("record_folder", "samples/audio")
    os.makedirs(folder, exist_ok=True)

    def next_rec_number(f):
        existing = glob.glob(os.path.join(f, "rec_*.wav"))
        nums = []
        for fp in existing:
            m = re.search(r"rec_(\d+)", os.path.basename(fp))
            if m:
                nums.append(int(m.group(1)))
        return max(nums, default=0) + 1

    clear()
    print()
    print("  ══════════════════════════════════════════════════════════════")
    print("  🎙️  RECORD VOICE TO .WAV")
    print("  ══════════════════════════════════════════════════════════════")
    print(f"  Save folder : {folder}/")
    print(f"  Auto-stop   : after {CFG['mic_silence_ms']}ms silence")
    print(f"  Max length  : {CFG['mic_max_sec']} seconds")
    print()
    print("  Each recording becomes a .wav file. Optional caption is")
    print("  saved as .txt next to it — perfect for STT/TTS training.")
    print()

    custom = ask_path("Change folder?", folder)
    if custom and custom != folder:
        folder = custom
        os.makedirs(folder, exist_ok=True)

    print()
    print("  Press Enter to record. Say what you want. Pause to stop.")
    print("  You'll hear playback, then choose to keep or retry.")
    print("  'q' at any prompt to quit.")
    print()

    saved = 0
    while True:
        print()
        try:
            cmd = input("  (Enter to record, 'q' to quit) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd in ("q", "quit", "exit"):
            break

        try:
            caption = input("  Caption (optional, becomes .txt) > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        audio = record_until_silence()
        if audio is None:
            print("  (nothing recorded — try again)")
            continue

        # trim + normalize for playback too
        audio_clean = trim_silence(audio, threshold=0.01)
        audio_clean = _normalize_audio(audio_clean)

        print("  playback:")
        play_audio(audio_clean)

        try:
            choice = input("  keep? (Y/n/r=retry) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "r":
            print("  retry — speak again")
            audio = record_until_silence()
            if audio is None:
                continue
            audio_clean = trim_silence(audio, threshold=0.01)
            audio_clean = _normalize_audio(audio_clean)
            play_audio(audio_clean)
            try:
                choice = input("  keep? (Y/n) > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

        if choice.startswith("n"):
            print("  discarded")
            continue

        # Build filename
        n = next_rec_number(folder)
        fname = f"rec_{n:03d}"
        if caption:
            safe = re.sub(r'[^a-z0-9]+', '_', caption.lower()).strip('_')[:40]
            if safe:
                fname = f"{fname}_{safe}"

        wav_path = os.path.join(folder, f"{fname}.wav")
        sf.write(wav_path, audio_clean.astype("float32"), AudioGen.SR)

        if caption:
            with open(os.path.join(folder, f"{fname}.txt"), "w") as f:
                f.write(caption)

        duration = len(audio_clean) / AudioGen.SR
        print(f"  ✓ saved: {wav_path}  ({duration:.2f}s)")
        if caption:
            print(f"           caption: {caption}")
        saved += 1

    print()
    if saved > 0:
        print(f"  ✓ {saved} recording(s) saved to {folder}/")
        print(f"    Use them for training:")
        print(f"      Audio → Train STT  (speech → text)")
        print(f"      Audio → Train TTS  (text → speech)")
    else:
        print(f"  (no recordings saved)")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SAMPLES (text/image)
# ═══════════════════════════════════════════════════════════════════════════
DIALOGUE_SAMPLE = """User: hello
Think: greeting, greet back
Bot: Hello! How can I help you today?

User: what is 2 plus 2
Think: 2 + 2 = 4
Bot: 2 plus 2 is 4.

User: what is your name
Think: identity question, say I am a small AI
Bot: I am a small AI trained on text data.

User: how are you
Think: well-being question, say doing well
Bot: I am doing well, thank you.

User: tell me a joke
Think: joke request, tell a short joke
Bot: Why did the computer go to the doctor? It had a virus.

User: goodbye
Think: farewell, say goodbye
Bot: Goodbye! Have a great day.

User: who made you
Think: origin question, made with PyTorch
Bot: I was made with PyTorch as a small transformer.

User: thanks
Think: thanks given, acknowledge
Bot: You are welcome.

User: help
Think: help request, ask what they need
Bot: Sure, tell me what you need help with.

User: good night
Think: farewell, say good night
Bot: Good night! Sleep well.
"""


def make_sample_text():
    if not os.path.exists("data.txt"):
        open("data.txt", "w", encoding="utf-8").write(DIALOGUE_SAMPLE)
        print("  created data.txt")
    return "data.txt"


def make_sample_images():
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  pip install pillow")
        return None
    folder = "samples/images"
    os.makedirs(folder, exist_ok=True)
    samples = [("a_red_square", (255, 0, 0), "sq"), ("a_blue_square", (0, 0, 255), "sq"),
               ("a_green_square", (0, 200, 0), "sq"), ("a_yellow_square", (255, 220, 0), "sq"),
               ("a_red_circle", (255, 0, 0), "ci"), ("a_blue_circle", (0, 0, 255), "ci"),
               ("a_green_circle", (0, 200, 0), "ci"), ("a_purple_circle", (160, 0, 200), "ci")]
    for name, color, shape in samples:
        img = Image.new("RGB", (128, 128), (30, 30, 30))
        d = ImageDraw.Draw(img)
        if shape == "sq": d.rectangle([20, 20, 108, 108], fill=color)
        else: d.ellipse([20, 20, 108, 108], fill=color)
        img.save(os.path.join(folder, f"{name}.png"))
        open(os.path.join(folder, f"{name}.txt"), "w").write(name.replace("_", " "))
    print(f"  created {len(samples)} images in {folder}/")
    return folder


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════
def train_text(data_arg, steps, model_dir, resume_from=None, fine_tune_from=None):
    device = get_device()
    parts = []
    for p in discover([data_arg], [".txt", ".md"]):
        try:
            parts.append(open(p, encoding="utf-8").read())
        except Exception as e:
            print(f"  skip {p}: {e}")
    text = "\n\n".join(parts)
    if not text.strip():
        print("  no text"); return None, None

    dialogue_mode = "User:" in text and "Bot:" in text
    has_think = "Think:" in text
    print(f"  text = {len(text):,}  style = {'dialogue' if dialogue_mode else 'free'}"
          f"{'  ✓ Think: format' if has_think else ''}")

    tok = None; start_step = 0
    if resume_from and os.path.exists(resume_from):
        model, tok, ckpt = load_ckpt(resume_from, device)
        config = ckpt["config"]; start_step = ckpt.get("training_step", 0)
        print(f"  resuming from step {start_step}")
    elif fine_tune_from and os.path.exists(fine_tune_from):
        model, tok, ckpt = load_ckpt(fine_tune_from, device)
        config = ckpt["config"]
        print(f"  fine-tuning existing")
    else:
        tok = BPE().train(text, 1024, max(64, len(text.encode()) // 2))
        bs = min(CFG["block_size"], max(32, len(text.encode()) // 10))
        n = len(text)
        if n < 3000: config = dict(n_embd=32, n_head=2, n_layer=1, block_size=bs)
        elif n < 15000: config = dict(n_embd=64, n_head=2, n_layer=2, block_size=bs)
        elif n < 100000: config = dict(n_embd=128, n_head=4, n_layer=4, block_size=bs)
        else: config = dict(n_embd=192, n_head=6, n_layer=4, block_size=bs)
        model = TextLM(vocab_size=len(tok.vocab), **config).to(device)

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    bs = config["block_size"]
    print(f"  tokens = {len(data):,}  params = {model.n_params():,}")
    if len(data) < bs + 4:
        print("  not enough tokens"); return None, None

    n_val = int(len(data) * 0.1)
    train_data, val_data = (data[:-n_val], data[-n_val:]) if n_val >= bs + 2 else (data, None)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"],
                            betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    print(f"\n  training {steps} steps on {device}...\n")
    best_val = float("inf")
    for step in range(1, steps + 1):
        lr = CFG["learning_rate"] * (0.5 * (1 + math.cos(math.pi * step / steps))) + 3e-5
        for g in opt.param_groups: g["lr"] = lr
        ix = torch.randint(0, len(train_data) - bs - 1, (CFG["batch_size"],))
        x = torch.stack([train_data[i:i + bs] for i in ix]).to(device)
        y = torch.stack([train_data[i + 1:i + bs + 1] for i in ix]).to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        global_step = start_step + step
        if step % max(1, steps // 100) == 0 or step == steps:
            vmsg = ""
            if val_data is not None and step % max(1, steps // 10) == 0:
                model.eval()
                with torch.no_grad():
                    n = min(10, len(val_data) - bs - 1)
                    vx = torch.stack([val_data[i:i + bs] for i in range(n)]).to(device)
                    vy = torch.stack([val_data[i + 1:i + bs + 1] for i in range(n)]).to(device)
                    _, vl = model(vx, vy)
                    vmsg = f" val={vl.item():.3f}"
                    if vl.item() < best_val:
                        best_val = vl.item()
                        save_ckpt(os.path.join(model_dir, "best.pt"), "text", model, tok,
                                  config, {"dialogue_mode": dialogue_mode,
                                           "has_think": has_think}, opt, global_step)
                model.train()
            progress(step, steps, loss.item(), vmsg)
    print()
    save_ckpt(os.path.join(model_dir, "last.pt"), "text", model, tok, config,
              {"dialogue_mode": dialogue_mode, "has_think": has_think}, opt, start_step + steps)
    if not os.path.exists(os.path.join(model_dir, "best.pt")):
        save_ckpt(os.path.join(model_dir, "best.pt"), "text", model, tok, config,
                  {"dialogue_mode": dialogue_mode, "has_think": has_think}, opt, start_step + steps)
    print(f"  saved -> {model_dir}/")
    return model, tok


def load_image_items(arg):
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("  pip install pillow"); return []
    items = []
    for p in discover([arg], [".png", ".jpg", ".jpeg", ".bmp"]):
        try:
            sc = os.path.splitext(p)[0] + ".txt"
            cap = open(sc).read().strip() if os.path.exists(sc) else \
                  os.path.splitext(os.path.basename(p))[0].replace("_", " ")
            im = Image.open(p).convert("RGB").resize((32, 32))
            arr = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.
            items.append((cap, arr))
        except Exception as e:
            print(f"  skip {p}: {e}")
    return items


def load_audio_items(arg):
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("  pip install soundfile"); return []
    items, n = [], AudioGen.SR
    for p in discover([arg], [".wav", ".mp3", ".flac", ".ogg"]):
        try:
            sc = os.path.splitext(p)[0] + ".txt"
            cap = open(sc).read().strip() if os.path.exists(sc) else \
                  os.path.splitext(os.path.basename(p))[0].replace("_", " ")
            d, sr = sf.read(p)
            if d.ndim > 1: d = d.mean(axis=1)
            d = _resample_to_16k(d, sr)
            d = _fit_to_1s(d, n)
            items.append((cap, torch.from_numpy(d.astype("float32"))))
        except Exception as e:
            print(f"  skip {p}: {e}")
    return items


def _enc_caps(tok, caps, device, prefix="\x00"):
    seqs = [tok.encode(prefix + c + "\n")[:32] for c in caps]
    L = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs): out[i, :len(s)] = torch.tensor(s)
    return out.to(device)


def _warn_if_synthetic(items, mode="audio"):
    if len(items) < 10:
        print(f"  ⚠ only {len(items)} {mode} files — expect memorization")
    beeps = sum(1 for c, _ in items if any(w in c.lower() for w in
        ["beep", "tone", "buzz", "sweep", "pulse", "chord", "warble", "noise", "saw", "triangle"]))
    if beeps >= len(items) * 0.8:
        print(f"  ⚠ {beeps}/{len(items)} of your samples are synthetic (not speech)")
        print(f"    For real speech: Audio → Record voice to .wav")


def train_image(folder, steps, model_dir):
    device = get_device(); items = load_image_items(folder)
    if len(items) < 4: print("  need 4+ images"); return
    caps = " ".join(c for c, _ in items)
    tok = BPE().train(caps, 256, max(8, len(caps) // 4))
    config = dict(n_embd=64, hidden=128, latent=256)
    model = ImageGen(vocab_size=len(tok.vocab), **config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=0.01)
    model.train(); print(f"\n  training {steps} steps...\n")
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(items), (min(CFG["batch_size"], len(items)),))
        b = [items[i] for i in ix.tolist()]
        imgs = torch.stack([x[1] for x in b]).to(device)
        pred = model(_enc_caps(tok, [x[0] for x in b], device))
        loss = F.mse_loss(pred, imgs)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % max(1, steps // 100) == 0: progress(step, steps, loss.item())
    print()
    save_ckpt(os.path.join(model_dir, "best.pt"), "image", model, tok, config)
    save_ckpt(os.path.join(model_dir, "last.pt"), "image", model, tok, config)
    print(f"  saved -> {model_dir}/")


def train_tts(folder, steps, model_dir):
    device = get_device(); items = load_audio_items(folder)
    if len(items) < 4: print("  need 4+ files"); return
    _warn_if_synthetic(items, "audio")
    caps = " ".join(c for c, _ in items)
    tok = BPE().train(caps, 256, max(8, len(caps) // 4))
    config = dict(n_embd=64, hidden=128, latent=128)
    model = AudioGen(vocab_size=len(tok.vocab), **config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=0.01)
    model.train(); print(f"\n  training {steps} steps...\n")
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(items), (min(CFG["batch_size"], len(items)),))
        b = [items[i] for i in ix.tolist()]
        wavs = torch.stack([x[1] for x in b]).to(device)
        pred = model(_enc_caps(tok, [x[0] for x in b], device, prefix=""))
        loss = F.mse_loss(pred, wavs)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % max(1, steps // 100) == 0: progress(step, steps, loss.item())
    print()
    save_ckpt(os.path.join(model_dir, "best.pt"), "tts", model, tok, config)
    save_ckpt(os.path.join(model_dir, "last.pt"), "tts", model, tok, config)
    print(f"  saved -> {model_dir}/")


def train_stt(folder, steps, model_dir):
    device = get_device(); items = load_audio_items(folder)
    if len(items) < 4: print("  need 4+ files"); return
    _warn_if_synthetic(items, "audio")
    caps = " ".join(c for c, _ in items)
    tok = BPE().train(caps, 256, max(8, len(caps) // 4))
    config = dict(hidden=256)
    model = AudioToText(vocab_size=len(tok.vocab), **config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=0.01)
    model.train(); print(f"\n  training {steps} steps...\n")
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(items), (min(CFG["batch_size"], len(items)),))
        b = [items[i] for i in ix.tolist()]
        wavs = torch.stack([x[1] for x in b]).to(device)
        _, loss = model(wavs, _enc_caps(tok, [x[0] for x in b], device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % max(1, steps // 100) == 0: progress(step, steps, loss.item())
    print()
    save_ckpt(os.path.join(model_dir, "best.pt"), "stt", model, tok, config)
    save_ckpt(os.path.join(model_dir, "last.pt"), "stt", model, tok, config)
    print(f"  saved -> {model_dir}/")


def train_caption(folder, steps, model_dir):
    device = get_device(); items = load_image_items(folder)
    if len(items) < 4: print("  need 4+ images"); return
    caps = " ".join(c for c, _ in items)
    tok = BPE().train(caps, 256, max(8, len(caps) // 4))
    config = dict(hidden=256)
    model = ImageCaption(vocab_size=len(tok.vocab), **config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=0.01)
    model.train(); print(f"\n  training {steps} steps...\n")
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(items), (min(CFG["batch_size"], len(items)),))
        b = [items[i] for i in ix.tolist()]
        imgs = torch.stack([x[1] for x in b]).to(device)
        _, loss = model(imgs, _enc_caps(tok, [x[0] for x in b], device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % max(1, steps // 100) == 0: progress(step, steps, loss.item())
    print()
    save_ckpt(os.path.join(model_dir, "best.pt"), "caption", model, tok, config)
    save_ckpt(os.path.join(model_dir, "last.pt"), "caption", model, tok, config)
    print(f"  saved -> {model_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
# LIVE AUDIO HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _check_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        print("\n  ⚠ need 'sounddevice' for live audio:")
        print("    pip install sounddevice")
        return None


def record_until_silence(max_sec=None, silence_ms=None, threshold=None):
    sd = _check_sounddevice()
    if sd is None: return None
    import numpy as np
    SR = AudioGen.SR
    max_sec = max_sec if max_sec is not None else CFG["mic_max_sec"]
    silence_ms = silence_ms if silence_ms is not None else CFG["mic_silence_ms"]
    threshold = threshold if threshold is not None else CFG["mic_threshold"]
    chunk_ms = 50
    chunk_size = int(SR * chunk_ms / 1000)
    silent_limit = silence_ms // chunk_ms
    max_chunks = max_sec * 1000 // chunk_ms
    buf = []; silent = 0; started = False
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32") as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_size)
                chunk = chunk.flatten()
                energy = float(np.sqrt(np.mean(chunk ** 2)))
                buf.append(chunk.copy())
                if energy > threshold:
                    started = True; silent = 0
                elif started:
                    silent += 1
                    if silent >= silent_limit: break
                meter = "█" * min(20, int(energy * 200))
                status = "listening" if started else "waiting"
                print(f"\r  🎤 {frames[i % len(frames)]} {status} [{meter:<20}]",
                      end="", flush=True)
                i += 1
    except Exception as e:
        print(f"\n  ⚠ mic error: {e}"); return None
    print("\r" + " " * 60 + "\r", end="")
    if not buf: return None
    audio = np.concatenate(buf).flatten()
    if len(audio) < SR // 4: return None
    return audio


def transcribe_audio(model, tok, device, audio, sr=AudioGen.SR):
    import numpy as np
    audio = _resample_to_16k(audio, sr)
    SR = AudioGen.SR
    results = []
    for start in range(0, len(audio), SR):
        chunk = _fit_to_1s(audio[start:start + SR], SR)
        wav = torch.tensor(chunk.astype("float32")).unsqueeze(0).to(device)
        h = model.enc(wav).unsqueeze(0).repeat(2, 1, 1)
        cur = torch.zeros(1, 1, dtype=torch.long, device=device)
        out = []
        with torch.no_grad():
            for _ in range(32):
                o, h = model.gru(model.emb(cur), h)
                nxt = model.head(o[:, -1]).argmax(-1).item()
                if nxt in (0, 10): break
                out.append(nxt)
                cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
        text = tok.decode(out).strip()
        if text: results.append(text)
    return " ".join(results).strip()


def transcribe_one_window(model, tok, device, window):
    import numpy as np
    window = _fit_to_1s(window, AudioGen.SR)
    wav = torch.tensor(window.astype("float32")).unsqueeze(0).to(device)
    h = model.enc(wav).unsqueeze(0).repeat(2, 1, 1)
    cur = torch.zeros(1, 1, dtype=torch.long, device=device)
    out = []
    with torch.no_grad():
        for _ in range(16):
            o, h = model.gru(model.emb(cur), h)
            nxt = model.head(o[:, -1]).argmax(-1).item()
            if nxt in (0, 10): break
            out.append(nxt)
            cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
    return tok.decode(out).strip()


def play_audio(wav, sr=AudioGen.SR):
    sd = _check_sounddevice()
    if sd is None: return False
    try:
        sd.play(wav, sr); sd.wait(); return True
    except Exception as e:
        print(f"  ⚠ speaker error: {e}"); return False


def list_audio_devices():
    sd = _check_sounddevice()
    if sd is None: return
    print("\n  Audio devices:")
    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        default_out = sd.default.device[1]
        for i, d in enumerate(devices):
            in_ch = d["max_input_channels"]; out_ch = d["max_output_channels"]
            mi = " ◀ default mic" if i == default_in and in_ch > 0 else ""
            mo = " ◀ default spk" if i == default_out and out_ch > 0 else ""
            kinds = []
            if in_ch > 0: kinds.append("IN")
            if out_ch > 0: kinds.append("OUT")
            ks = "/".join(kinds) if kinds else "—"
            print(f"    [{i}] {d['name']} ({ks}){mi}{mo}")
    except Exception as e:
        print(f"    error: {e}")


def live_subtitles_session(model_dir):
    sd = _check_sounddevice()
    if sd is None: return
    import numpy as np
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path):
        print("  no STT model trained yet"); return
    model, tok, ckpt = load_ckpt(path, device)
    SR = AudioGen.SR
    window_sec = CFG["subtitle_window"]
    hop_sec = CFG["subtitle_hop"]
    stable_need = CFG["subtitle_stable"]
    min_energy = CFG["mic_threshold"]
    window_size = int(SR * window_sec)
    hop_size = int(SR * hop_sec)
    clear()
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║  🎤 LIVE SUBTITLES                                   ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  window: {window_sec:.1f}s   hop: {hop_sec:.2f}s   stable: {stable_need}x{'':<14}║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print("\n  Speak normally. Lines appear when stable. Ctrl+C to stop.\n")
    print("  ──────────────────────────────────────────────────────\n")
    ring = np.zeros(window_size, dtype="float32")
    current_text = ""; stable_count = 0
    last_committed = ""; committed = []
    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=hop_size) as stream:
            while True:
                chunk, _ = stream.read(hop_size)
                chunk = chunk.flatten()
                if len(chunk) >= window_size: ring = chunk[-window_size:]
                else: ring = np.concatenate([ring[len(chunk):], chunk])
                energy = float(np.sqrt(np.mean(ring ** 2)))
                if energy < min_energy:
                    if current_text and stable_count >= stable_need:
                        if current_text != last_committed:
                            committed.append(current_text)
                            last_committed = current_text
                            ts = time.strftime("%H:%M:%S")
                            print(f"  [{ts}] {current_text}")
                    current_text = ""; stable_count = 0
                    continue
                text = transcribe_one_window(model, tok, device, ring)
                if not text: continue
                if text == current_text: stable_count += 1
                else: current_text = text; stable_count = 1
                if stable_count >= stable_need and text != last_committed:
                    committed.append(text)
                    last_committed = text
                    ts = time.strftime("%H:%M:%S")
                    print(f"  [{ts}] {text}")
                    current_text = ""; stable_count = 0
    except KeyboardInterrupt:
        print("\n")
        print("  ──────────────────────────────────────────────────────")
        print(f"  stopped. {len(committed)} line(s) transcribed.\n")


def stt_live_session(model_dir):
    if _check_sounddevice() is None: return
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path):
        print("  no STT model trained yet"); return
    model, tok, ckpt = load_ckpt(path, device)
    print("\n  === speech → text  (LIVE MICROPHONE) ===")
    print(f"  ✓ model loaded: {path}")
    print("\n  Press Enter to record, speak, stop automatically. 'exit' to leave.\n")
    while True:
        try:
            cmd = input("  (Enter to record, or 'exit') > ").strip().lower()
        except (EOFError, KeyboardInterrupt): break
        if cmd in ("exit", "quit", "q"): break
        audio = record_until_silence()
        if audio is None:
            print("  (nothing recorded)"); continue
        print("  🧠 transcribing...", end="", flush=True)
        text = transcribe_audio(model, tok, device, audio)
        print(f"\r  📝 {text if text else '(could not understand)'}          \n")


def tts_live_session(model_dir):
    if _check_sounddevice() is None: return
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path):
        print("  no TTS model trained yet"); return
    model, tok, ckpt = load_ckpt(path, device)
    print("\n  === text → speech  (LIVE PLAYBACK) ===")
    print(f"  ✓ model loaded: {path}")
    print("  Type text, hear it play. 'exit' to leave.\n")
    while True:
        try: p = input("  say> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if not p: continue
        if p.lower() in ("exit", "quit"): break
        ids = tok.encode(p)[:32]
        if not ids: print("  (no known words)"); continue
        with torch.no_grad():
            wav = model(torch.tensor([ids], dtype=torch.long, device=device))[0].cpu().numpy()
        print("  🔊 ", end="", flush=True)
        if play_audio(wav): print("\r  ✓ played                    ")
        print()


def voice_conversation_session():
    if _check_sounddevice() is None: return
    device = get_device()
    need = {"text": "checkpoints/text/best.pt", "stt": "checkpoints/stt/best.pt",
            "tts": "checkpoints/tts/best.pt"}
    missing = [k for k, p in need.items() if not os.path.exists(p)]
    if missing:
        print(f"\n  Missing models: {', '.join(missing)}"); return
    print("  loading models...")
    tm, ttk, tck = load_ckpt(need["text"], device)
    sm, stk, sck = load_ckpt(need["stt"], device)
    am, atk, ack = load_ckpt(need["tts"], device)
    dialogue_mode = tck.get("dialogue_mode", False)
    has_think = tck.get("has_think", False)
    bs = tck["config"]["block_size"]
    conv = Conversation(persona=CFG.get("persona", ""))
    print("\n  === VOICE CONVERSATION ===")
    print("  Speak → AI thinks → AI speaks back. 'exit' to leave.\n")
    while True:
        try: cmd = input("  (Enter to talk, or 'exit') > ").strip().lower()
        except (EOFError, KeyboardInterrupt): break
        if cmd in ("exit", "quit", "q"): break
        audio = record_until_silence()
        if audio is None:
            print("  (nothing heard)"); continue
        print("  🧠 hearing...", end="", flush=True)
        user_text = transcribe_audio(sm, stk, device, audio)
        print(f"\r  👤 you said: {user_text or '(unclear)'}          ")
        if not user_text: continue
        thought = None
        if has_think and CFG["learning_think"] and CFG["thinking_style"] != "off":
            thought = real_thought(tm, ttk, user_text, bs)
            if thought: print(f"  💭 {thought}")
        print("  🤖 thinking...", end="", flush=True)
        prompt = conv.build_prompt(user_text, dialogue_mode,
                                   CFG["memory_turns"], thought=thought)
        apply_seed()
        reply = filter_output(generate_text(tm, ttk, prompt, bs))
        print(f"\r  🤖 bot: {reply or '...'}          ")
        conv.add("user", user_text); conv.add("bot", reply or "")
        if reply:
            ids = atk.encode(reply)[:32]
            if ids:
                with torch.no_grad():
                    wav = am(torch.tensor([ids], dtype=torch.long, device=device))[0].cpu().numpy()
                print("  🔊 speaking...", end="", flush=True)
                play_audio(wav)
                print("\r  ✓ done                    ")
        print()


def audio_settings_flow():
    while True:
        header("🎙️  MICROPHONE SETTINGS")
        print(f"  silence threshold  : {CFG['mic_threshold']}")
        print(f"  stop after silence : {CFG['mic_silence_ms']} ms")
        print(f"  max recording      : {CFG['mic_max_sec']} s")
        print(f"  subtitle window    : {CFG['subtitle_window']} s")
        print(f"  subtitle hop       : {CFG['subtitle_hop']} s")
        print(f"  subtitle stable    : {CFG['subtitle_stable']} x")
        print(f"  TTS rate (samples) : {CFG['tts_rate']} wpm")
        print(f"  record folder      : {CFG['record_folder']}")
        c = ask("Change:", [
            "Silence threshold", "Stop after silence (ms)", "Max recording time (s)",
            "Subtitle window (s)", "Subtitle hop (s)", "Subtitle stability",
            "TTS rate", "Record folder",
            "List audio devices", "Test microphone", "Back",
        ])
        if c in (-1, 11): return
        if c == 1: CFG["mic_threshold"] = ask_number("Threshold?", CFG["mic_threshold"], 0.001, 0.5)
        elif c == 2: CFG["mic_silence_ms"] = ask_number("Silence ms?", CFG["mic_silence_ms"], 200, 3000, int)
        elif c == 3: CFG["mic_max_sec"] = ask_number("Max seconds?", CFG["mic_max_sec"], 1, 30, int)
        elif c == 4: CFG["subtitle_window"] = ask_number("Window seconds?", CFG["subtitle_window"], 0.5, 3.0)
        elif c == 5: CFG["subtitle_hop"] = ask_number("Hop seconds?", CFG["subtitle_hop"], 0.1, 1.0)
        elif c == 6: CFG["subtitle_stable"] = ask_number("Stable repeats?", CFG["subtitle_stable"], 1, 5, int)
        elif c == 7: CFG["tts_rate"] = ask_number("TTS rate (wpm)?", CFG["tts_rate"], 100, 250, int)
        elif c == 8: CFG["record_folder"] = ask_text("Record folder?", CFG["record_folder"])
        elif c == 9:
            list_audio_devices(); pause(); continue
        elif c == 10:
            print("\n  recording 3 seconds...")
            audio = record_until_silence(max_sec=3)
            if audio is None: print("  (nothing)")
            else:
                print(f"  recorded {len(audio) / AudioGen.SR:.1f}s — playing back")
                play_audio(audio)
            pause(); continue
        save_cfg(); print("  saved!")


# ═══════════════════════════════════════════════════════════════════════════
# CLI CHAT
# ═══════════════════════════════════════════════════════════════════════════
def thinking_chat_session(model_dir, style="chat", title="Chat"):
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path):
        print("  no trained model"); return
    model, tok, ckpt = load_ckpt(path, device)
    dialogue_mode = ckpt.get("dialogue_mode", False)
    has_think = ckpt.get("has_think", False)
    bs = ckpt["config"]["block_size"]
    conv = Conversation(persona=CFG.get("persona", ""))
    print(f"\n  ┌─ {title} ────────────────────────────────")
    print(f"  │ model thinks : {'yes' if has_think else 'no'}")
    print(f"  │ show thinking: {CFG['thinking_style']}")
    print(f"  │ memory       : {CFG['memory_turns']} turns")
    print(f"  └──────────────────────────────────────────")
    print("  commands: 'exit' · 'save' · 'clear' · 'style'\n")
    while True:
        try: user = input("  you> ").strip()
        except (EOFError, KeyboardInterrupt): print(); break
        if not user: continue
        low = user.lower()
        if low in ("exit", "quit"): break
        if low == "save":
            fn = conv.save(); print(f"  ✓ saved -> {fn}"); continue
        if low == "clear":
            conv = Conversation(persona=conv.persona); print("  ✓ memory cleared"); continue
        if low == "style":
            cycle = {"off": "brief", "brief": "verbose", "verbose": "off"}
            CFG["thinking_style"] = cycle[CFG["thinking_style"]]
            save_cfg(); print(f"  style = {CFG['thinking_style']}"); continue
        thought = None
        if has_think and CFG["learning_think"] and CFG["thinking_style"] != "off":
            thought = run_with_spinner("thinking...", lambda u=user: real_thought(model, tok, u, bs))
            if thought and CFG["thinking_style"] != "off":
                print(f"  ┌─ model thought ─────────────────────")
                print(f"  │ {thought}")
                print(f"  └──────────────────────────────────────\n")
        prompt = conv.build_prompt(user, dialogue_mode, CFG["memory_turns"], thought=thought)
        if not dialogue_mode and style != "chat":
            prompt = format_prompt(user, style, dialogue_mode)
        apply_seed()
        reply = run_with_spinner("composing...", lambda p=prompt: generate_text(model, tok, p, bs))
        reply = filter_output(reply)
        print(f"  bot> {reply or '...'}\n")
        conv.add("user", user); conv.add("bot", reply or "")
    if conv.messages:
        fn = conv.save(); print(f"  ✓ auto-saved -> {fn}\n")


def image_session(model_dir):
    try: from PIL import Image
    except ImportError: print("  pip install pillow"); return
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path): print("  no model"); return
    model, tok, ckpt = load_ckpt(path, device)
    os.makedirs("my_pictures", exist_ok=True)
    print("\n  === image generator  ('exit') ===\n")
    while True:
        try: p = input("  prompt> ").strip()
        except (EOFError, KeyboardInterrupt): print(); break
        if not p: continue
        if p.lower() in ("exit", "quit"): break
        ids = tok.encode("\x00" + p + "\n")[:32]
        if not ids: print("  (no known words)\n"); continue
        with torch.no_grad():
            img = model(torch.tensor([ids], dtype=torch.long, device=device))[0]
            img = img.cpu().permute(1, 2, 0).numpy()
        out = f"my_pictures/{int(time.time() * 1000)}.png"
        Image.fromarray((img * 255).clip(0, 255).astype("uint8")).resize(
            (256, 256), Image.NEAREST).save(out)
        print(f"  saved -> {out}\n")


def stt_file_session(model_dir):
    try:
        import numpy as np
        import soundfile as sf
    except ImportError: print("  pip install soundfile"); return
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path): print("  no STT model trained yet"); return
    model, tok, ckpt = load_ckpt(path, device)
    print("\n  === transcribe an audio FILE ===")
    print(f"  ✓ model loaded: {path}")
    print("  Audio files end in: .wav, .mp3, .flac, .ogg\n")
    audio_files = find_files_with_ext([".wav", ".mp3", ".flac", ".ogg"])
    if not audio_files:
        print("  No audio files found.")
        c = ask("What do you want?", [
            "Generate speech samples (pyttsx3)",
            "Record voice to .wav",
            "I'll type a path elsewhere", "Exit",
        ])
        if c == -1 or c == 4: return
        if c == 1:
            folder = make_sample_audio()
            if not folder: return
            audio_files = find_files_with_ext([".wav", ".mp3", ".flac", ".ogg"])
        elif c == 2:
            record_to_wav_session()
            audio_files = find_files_with_ext([".wav", ".mp3", ".flac", ".ogg"])
    if audio_files:
        print(f"  Found {len(audio_files)} file(s). Examples:")
        for f in audio_files[:6]: print(f"    {f}")
        print()
    print("  Type a path, or drag a file in. 'exit' to quit.\n")
    while True:
        try: p = input("  file> ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt): print(); break
        if not p: continue
        if p.lower() in ("exit", "quit"): break
        if not os.path.exists(p): print(f"  ⚠ not found: {p}"); continue
        ext = os.path.splitext(p)[1].lower()
        if ext not in (".wav", ".mp3", ".flac", ".ogg"):
            print(f"  ⚠ not audio: {ext}. Use .wav/.mp3/.flac/.ogg")
            if ext == ".pt": print(f"    (that's the model)")
            continue
        try: d, sr = sf.read(p)
        except Exception as e: print(f"  ⚠ couldn't read: {e}"); continue
        text = transcribe_audio(model, tok, device, d, sr)
        print(f"  -> {text if text else '(could not understand)'}\n")


def caption_session(model_dir):
    try:
        from PIL import Image
        import numpy as np
    except ImportError: print("  pip install pillow"); return
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path): print("  no caption model trained yet"); return
    model, tok, ckpt = load_ckpt(path, device)
    print("\n  === image → text ===")
    print(f"  ✓ model loaded: {path}")
    print("  Image files end in: .png, .jpg, .jpeg, .bmp\n")
    img_files = find_files_with_ext([".png", ".jpg", ".jpeg", ".bmp"])
    if not img_files:
        print("  No image files found.")
        c = ask("What do you want?", [
            "Make sample images", "I'll type a path", "Exit",
        ])
        if c == -1 or c == 3: return
        if c == 1:
            folder = make_sample_images()
            if not folder: return
            img_files = find_files_with_ext([".png", ".jpg", ".jpeg", ".bmp"])
    if img_files:
        print(f"  Found {len(img_files)} file(s). Examples:")
        for f in img_files[:6]: print(f"    {f}")
        print()
    print("  Type a path, or drag a file in. 'exit' to quit.\n")
    while True:
        try: p = input("  image> ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt): print(); break
        if not p: continue
        if p.lower() in ("exit", "quit"): break
        if not os.path.exists(p): print(f"  ⚠ not found: {p}"); continue
        ext = os.path.splitext(p)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".bmp"):
            print(f"  ⚠ not image: {ext}. Use .png/.jpg/.jpeg/.bmp")
            if ext == ".pt": print(f"    (that's the model)")
            continue
        try: im = Image.open(p).convert("RGB").resize((32, 32))
        except Exception as e: print(f"  ⚠ couldn't open: {e}"); continue
        arr = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
        h = model.enc(arr).unsqueeze(0).repeat(2, 1, 1)
        cur = torch.zeros(1, 1, dtype=torch.long, device=device)
        out = []
        with torch.no_grad():
            for _ in range(32):
                o, h = model.gru(model.emb(cur), h)
                nxt = model.head(o[:, -1]).argmax(-1).item()
                if nxt in (0, 10): break
                out.append(nxt)
                cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
        text = tok.decode(out).strip()
        print(f"  -> {text if text else '(no caption)'}\n")


# ═══════════════════════════════════════════════════════════════════════════
# MODEL MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════
def all_models():
    ms = []
    for mode in MODELS:
        d = f"checkpoints/{mode}"
        for f in ["best.pt", "last.pt"]:
            p = os.path.join(d, f)
            if os.path.exists(p): ms.append((mode, f, p))
    return ms


def cmd_list_models():
    header("MY MODELS"); ms = all_models()
    if not ms: print("  (none)"); return
    print(f"  {'mode':<10} {'file':<10} {'size':>10}   path")
    print("  " + "-" * 60)
    for m, f, p in ms:
        print(f"  {m:<10} {f:<10} {human_size(os.path.getsize(p)):>10}   {p}")


def cmd_model_details():
    header("MODEL DETAILS"); ms = all_models()
    if not ms: print("  (none)"); return
    c = ask("Which?", [f"{m}/{f}" for m, f, _ in ms])
    if c == -1: return
    m, f, p = ms[c - 1]
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        print(f"\n  file:    {p}")
        print(f"  size:    {human_size(os.path.getsize(p))}")
        print(f"  mode:    {ck['mode']}")
        print(f"  vocab:   {len(ck['tokenizer_merges']) + 256}")
        print(f"  step:    {ck.get('training_step', '?')}")
        print(f"  config:  {ck['config']}")
        if "has_think" in ck: print(f"  think:   {ck['has_think']}")
    except Exception as e: print(f"  error: {e}")


def cmd_delete_model():
    header("DELETE MODEL"); ms = all_models()
    if not ms: print("  (none)"); return
    c = ask("Which?", [f"{m}/{f}" for m, f, _ in ms])
    if c == -1: return
    m, f, p = ms[c - 1]
    if ask(f"Delete {p}?", ["Yes", "No"]) == 1:
        os.remove(p); print(f"  deleted {p}")


def cmd_backup():
    if not os.path.isdir("checkpoints"): print("  nothing"); return
    stamp = time.strftime("%Y%m%d_%H%M%S"); name = f"backup_{stamp}"
    shutil.copytree("checkpoints", name); print(f"  saved -> {name}/")


def cmd_reset():
    header("RESET EVERYTHING")
    if ask("Delete ALL models?", ["Yes", "No"]) == 1:
        for d in ["checkpoints", "chats"]:
            if os.path.isdir(d): shutil.rmtree(d)
        if os.path.exists(SETTINGS_FILE): os.remove(SETTINGS_FILE)
        print("  reset complete")


# ═══════════════════════════════════════════════════════════════════════════
# SERVER (minimal)
# ═══════════════════════════════════════════════════════════════════════════
SERVER_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Mini AI</title>
<style>
body{font-family:system-ui;background:#1a1a1a;color:#e0e0e0;margin:0;padding:20px}
#messages{max-width:800px;margin:0 auto;min-height:80vh}
.msg{padding:10px 14px;border-radius:8px;margin-bottom:12px;white-space:pre-wrap;max-width:80%}
.user{background:#2a4a7a;margin-left:auto}
.bot{background:#2a2a2a}
.think{background:#1f2a1f;border-left:3px solid #4a7a4a;font-size:13px;color:#b0b0b0;margin-bottom:6px;padding:8px 12px;border-radius:6px}
#input-area{max-width:800px;margin:20px auto;display:flex;gap:8px}
#input{flex:1;padding:10px;background:#1a1a1a;border:1px solid #333;border-radius:6px;color:#e0e0e0;font-family:inherit}
button{padding:10px 16px;background:#4a7aa5;color:#fff;border:none;border-radius:6px;cursor:pointer}
</style></head><body>
<div id="messages"></div>
<div id="input-area"><input id="input" placeholder="Type..."><button onclick="send()">Send</button></div>
<script>
var SESSION='web-'+Math.random().toString(36).slice(2,10);
async function send(){var input=document.getElementById('input');var msg=input.value.trim();if(!msg)return;input.value='';addMsg('user',msg);var r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg,session:SESSION})});var d=await r.json();if(d.thought)addMsg('think',d.thought);addMsg('bot',d.reply||'...');}
function addMsg(role,text){var div=document.createElement('div');div.className='msg '+(role==='think'?'think':role);div.textContent=text;document.getElementById('messages').appendChild(div);div.scrollIntoView();}
document.getElementById('input').addEventListener('keydown',function(e){if(e.key==='Enter')send()});
</script></body></html>"""


def start_server(port=8000, api_key=None, model_dir="checkpoints/text"):
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path): print(f"  no model at {path}"); return
    print(f"  loading model...")
    model, tok, ckpt = load_ckpt(path, device)
    dialogue_mode = ckpt.get("dialogue_mode", False)
    has_think = ckpt.get("has_think", False)
    bs = ckpt["config"]["block_size"]
    n_params = model.n_params()
    print(f"  loaded: {n_params:,} params")
    conversations = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, body, ctype="application/json"):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body.encode() if isinstance(body, str) else body)
        def _json(self, d, code=200): self._send(code, json.dumps(d))
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, SERVER_HTML, "text/html; charset=utf-8")
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n)) if n else {}
            if self.path == "/api/chat":
                msg = body.get("message","").strip()
                if not msg: return self._json({"error":"empty"},400)
                session = body.get("session","default")
                with lock:
                    if session not in conversations:
                        conversations[session] = Conversation(persona=CFG["persona"])
                    conv = conversations[session]
                    thought = None
                    if has_think and CFG["learning_think"] and CFG["thinking_style"]!="off":
                        thought = real_thought(model, tok, msg, bs)
                    prompt = conv.build_prompt(msg, dialogue_mode, CFG["memory_turns"], thought=thought)
                    reply = filter_output(generate_text(model, tok, prompt, bs))
                    conv.add("user", msg); conv.add("bot", reply)
                self._json({"reply": reply, "thought": thought})
            else:
                self._send(404, "not found", "text/plain")

    try: server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    except OSError as e: print(f"  couldn't bind: {e}"); return
    print(f"\n  🌐 http://localhost:{port}")
    if api_key: print(f"  🔑 key: {api_key}")
    print(f"  Ctrl+C to stop\n")
    try: server.serve_forever()
    except KeyboardInterrupt: print("\n  stopping...")
    finally: server.shutdown()


def server_flow():
    header("🌐  START SERVER")
    model_dir = "checkpoints/text"
    if not os.path.exists(os.path.join(model_dir, "best.pt")):
        print("\n  No text model trained yet."); pause(); return
    port = ask_number("Port?", 8000, 1024, 65535, int)
    api_key = None
    kc = ask("API key?", ["No — open", "Yes — generate", "Yes — type it"])
    if kc == 2: api_key = secrets.token_hex(16); print(f"\n  API key: {api_key}")
    elif kc == 3: api_key = ask_text("API key", "") or None
    start_server(port=port, api_key=api_key, model_dir=model_dir)
    pause()


# ═══════════════════════════════════════════════════════════════════════════
# FLOWS
# ═══════════════════════════════════════════════════════════════════════════
TEXT_MODES = {
    1: ("Free chat", "chat", 0.8), 2: ("Creative", "chat", 1.3),
    3: ("Focused", "chat", 0.4), 4: ("Q&A bot", "qa", 0.8),
    5: ("Story", "story", 0.9), 6: ("Poem", "poem", 1.0),
    7: ("Continue", "continue", 0.85), 8: ("Roleplay", "chat", 0.9),
}


def text_flow():
    while True:
        header("💬 TEXT & CHAT")
        model_dir = "checkpoints/text"
        has = os.path.exists(os.path.join(model_dir, "best.pt"))
        c = ask("Pick a mode:",
                [f"{name}" for name, _, _ in TEXT_MODES.values()] + [
                    "Train new model", "Fine-tune", "Show info", "Back"],
                subtitle=f"status: {'✓' if has else '✗'}")
        if c in (-1, 12): return
        if 1 <= c <= 8:
            if not has: print("\n  No model yet."); pause(); continue
            name, style, temp = TEXT_MODES[c]
            orig = CFG["temperature"]
            if style == "chat": CFG["temperature"] = temp
            thinking_chat_session(model_dir, style, name)
            CFG["temperature"] = orig
        elif c == 9:
            data = ask_path("Training text?", "data.txt")
            if not (os.path.exists(data) or glob.glob(data)):
                data = make_sample_text()
            steps = ask_preset()
            if steps: train_text(data, steps, model_dir)
            pause()
        elif c == 10:
            if not has: print("  nothing"); pause(); continue
            data = ask_path("New data?", "data.txt")
            if not os.path.exists(data): print("  not found"); pause(); continue
            steps = ask_preset()
            if steps: train_text(data, steps, model_dir,
                                 fine_tune_from=os.path.join(model_dir, "best.pt"))
            pause()
        elif c == 11:
            if not has: print("  no model"); pause(); continue
            m, tok, ck = load_ckpt(os.path.join(model_dir, "best.pt"), get_device())
            print(f"\n  params: {m.n_params():,}  vocab: {len(tok.vocab)}")
            print(f"  config: {ck['config']}")
            print(f"  think:  {ck.get('has_think', False)}")
            pause()


def audio_flow():
    while True:
        header("🎵 AUDIO")
        tts_ok = os.path.exists("checkpoints/tts/best.pt")
        stt_ok = os.path.exists("checkpoints/stt/best.pt")
        n_audio = len(find_files_with_ext([".wav", ".mp3", ".flac", ".ogg"]))
        c = ask("Pick:", [
            "🎙️  Record voice to .wav        (free-form recorder)",
            f"🎤 Live subtitles               {'✓' if stt_ok else '✗ (needs STT)'}",
            f"🎤 Live speech → text           {'✓' if stt_ok else '✗'}",
            f"🔊 Live text → speech           {'✓' if tts_ok else '✗'}",
            f"💬 Voice conversation           {'✓' if (stt_ok and tts_ok and os.path.exists('checkpoints/text/best.pt')) else '✗'}",
            "📁 Transcribe an audio file",
            "🎙️  Guided recording (training phrases)",
            "🗣️  Generate speech samples (pyttsx3)",
            "🎙️  Microphone settings",
            "Train TTS", "Train STT", "Back",
        ], subtitle=f"audio files: {n_audio}  ·  tts: {'✓' if tts_ok else '✗'}  stt: {'✓' if stt_ok else '✗'}")
        if c in (-1, 12): return
        if c == 1:
            record_to_wav_session()
        elif c == 2:
            if not stt_ok: print("  train STT first"); pause()
            else: live_subtitles_session("checkpoints/stt")
        elif c == 3:
            if not stt_ok: print("  train STT first"); pause()
            else: stt_live_session("checkpoints/stt")
        elif c == 4:
            if not tts_ok: print("  train TTS first"); pause()
            else: tts_live_session("checkpoints/tts")
        elif c == 5:
            voice_conversation_session()
        elif c == 6:
            if not stt_ok: print("  train STT first"); pause()
            else: stt_file_session("checkpoints/stt")
        elif c == 7:
            record_own_samples()
            pause()
        elif c == 8:
            make_sample_audio()
            pause()
        elif c == 9:
            audio_settings_flow()
        elif c == 10:
            folder = ask_path("Audio folder?", "samples/audio")
            if not os.path.isdir(folder): folder = make_sample_audio() or ""
            if folder:
                steps = ask_preset()
                if steps: train_tts(folder, steps, "checkpoints/tts")
            pause()
        elif c == 11:
            folder = ask_path("Audio folder?", "samples/audio")
            if not os.path.isdir(folder): folder = make_sample_audio() or ""
            if folder:
                steps = ask_preset()
                if steps: train_stt(folder, steps, "checkpoints/stt")
            pause()


def image_flow():
    while True:
        header("🖼️  IMAGE")
        img_ok = os.path.exists("checkpoints/image/best.pt")
        cap_ok = os.path.exists("checkpoints/caption/best.pt")
        c = ask("Pick:", [
            f"Text → image  {'✓' if img_ok else '✗'}",
            f"Image → text  {'✓' if cap_ok else '✗'}",
            "Train text→image", "Train image→text", "Back"])
        if c in (-1, 5): return
        if c == 1:
            if img_ok: image_session("checkpoints/image")
            else: print("  no model"); pause()
        elif c == 2:
            if cap_ok: caption_session("checkpoints/caption")
            else: print("  no model"); pause()
        elif c == 3:
            folder = ask_path("Image folder?", "samples/images")
            if not os.path.isdir(folder): folder = make_sample_images() or ""
            if folder:
                steps = ask_preset()
                if steps: train_image(folder, steps, "checkpoints/image")
            pause()
        elif c == 4:
            folder = ask_path("Image folder?", "samples/images")
            if not os.path.isdir(folder): folder = make_sample_images() or ""
            if folder:
                steps = ask_preset()
                if steps: train_caption(folder, steps, "checkpoints/caption")
            pause()


def any_to_any_flow():
    header("🔀 ANY TO ANY")
    inp = ask("Input?", ["Text", "Audio", "Image"])
    if inp == -1: return
    outp = ask("Output?", ["Text", "Audio", "Image"])
    if outp == -1: return
    pairs = {(1, 1): text_flow, (1, 2): audio_flow, (1, 3): image_flow,
             (2, 1): audio_flow, (3, 1): image_flow}
    fn = pairs.get((inp, outp))
    if not fn: print("\n  that pair isn't supported"); pause(); return
    fn()


def train_flow():
    while True:
        header("🎓 TRAIN")
        c = ask("What kind?", ["Text", "Text → speech", "Speech → text",
                              "Text → image", "Image → text", "Back"])
        if c in (-1, 6): return
        if c == 1:
            data = ask_path("Text file?", "data.txt")
            if not (os.path.exists(data) or glob.glob(data)): data = make_sample_text()
            steps = ask_preset()
            if steps: train_text(data, steps, "checkpoints/text")
            pause()
        elif c == 2:
            folder = ask_path("Audio folder?", "samples/audio")
            if not os.path.isdir(folder):
                print("  folder doesn't exist. options:")
                a = ask("Get data first?", [
                    "Record voice to .wav", "Guided recording", "Generate speech", "Skip"])
                if a == 1: record_to_wav_session()
                elif a == 2: record_own_samples()
                elif a == 3: make_sample_audio()
                folder = "samples/audio" if os.path.isdir("samples/audio") else ""
            if folder:
                steps = ask_preset()
                if steps: train_tts(folder, steps, "checkpoints/tts")
            pause()
        elif c == 3:
            folder = ask_path("Audio folder?", "samples/audio")
            if not os.path.isdir(folder):
                print("  folder doesn't exist. options:")
                a = ask("Get data first?", [
                    "Record voice to .wav", "Guided recording", "Generate speech", "Skip"])
                if a == 1: record_to_wav_session()
                elif a == 2: record_own_samples()
                elif a == 3: make_sample_audio()
                folder = "samples/audio" if os.path.isdir("samples/audio") else ""
            if folder:
                steps = ask_preset()
                if steps: train_stt(folder, steps, "checkpoints/stt")
            pause()
        elif c == 4:
            folder = ask_path("Image folder?", "samples/images")
            if not os.path.isdir(folder): folder = make_sample_images() or ""
            if folder:
                steps = ask_preset()
                if steps: train_image(folder, steps, "checkpoints/image")
            pause()
        elif c == 5:
            folder = ask_path("Image folder?", "samples/images")
            if not os.path.isdir(folder): folder = make_sample_images() or ""
            if folder:
                steps = ask_preset()
                if steps: train_caption(folder, steps, "checkpoints/caption")
            pause()


def settings_flow():
    while True:
        header("⚙️  SETTINGS")
        print(f"  temperature     : {CFG['temperature']}")
        print(f"  top_k           : {CFG['top_k']}")
        print(f"  max_tokens      : {CFG['max_new_tokens']}")
        print(f"  memory_turns    : {CFG['memory_turns']}")
        print(f"  thinking_style  : {CFG['thinking_style']}")
        print(f"  learning_think  : {CFG['learning_think']}")
        print(f"  persona         : {CFG['persona'] or '(none)'}")
        print(f"  seed            : {CFG['seed']}")
        print(f"  device          : {CFG['device']}")
        print(f"  batch_size      : {CFG['batch_size']}")
        print(f"  learning_rate   : {CFG['learning_rate']}")
        print(f"  block_size      : {CFG['block_size']}")
        print(f"  tts_rate        : {CFG['tts_rate']}")
        print(f"  record_folder   : {CFG['record_folder']}")
        c = ask("Change:", [
            "Temperature", "Top-K", "Max tokens", "Memory turns",
            "Thinking style", "Use learned Think: lines",
            "Persona", "Banned words", "Seed",
            "Device", "Batch size", "Learning rate", "Block size",
            "TTS rate for samples", "Record folder",
            "Reset to defaults", "Back"])
        if c in (-1, 17): return
        if c == 1: CFG["temperature"] = ask_number("Temperature?", CFG["temperature"], 0.05, 2.0)
        elif c == 2: CFG["top_k"] = ask_number("Top-K?", CFG["top_k"], 0, 500, int)
        elif c == 3: CFG["max_new_tokens"] = ask_number("Max tokens?", CFG["max_new_tokens"], 10, 2000, int)
        elif c == 4: CFG["memory_turns"] = ask_number("Memory turns?", CFG["memory_turns"], 0, 50, int)
        elif c == 5:
            o = ask("Thinking style:", ["off", "brief", "verbose"])
            if o in (1, 2, 3): CFG["thinking_style"] = ["off", "brief", "verbose"][o - 1]
        elif c == 6:
            CFG["learning_think"] = not CFG["learning_think"]
            print(f"  learning_think = {CFG['learning_think']}")
        elif c == 7: CFG["persona"] = ask_text("Persona", CFG["persona"])
        elif c == 8:
            words = ask_text("Banned words (comma-separated)", ",".join(CFG["banned_words"]))
            CFG["banned_words"] = [w.strip() for w in words.split(",") if w.strip()]
        elif c == 9: CFG["seed"] = ask_number("Seed (-1 = random)", CFG["seed"], -1, 999999, int)
        elif c == 10:
            d = ask("Device:", ["auto", "cpu", "cuda"])
            if d in (1, 2, 3): CFG["device"] = ["auto", "cpu", "cuda"][d - 1]
        elif c == 11: CFG["batch_size"] = ask_number("Batch size?", CFG["batch_size"], 1, 128, int)
        elif c == 12: CFG["learning_rate"] = ask_number("Learning rate?", CFG["learning_rate"], 1e-6, 1e-2)
        elif c == 13: CFG["block_size"] = ask_number("Block size?", CFG["block_size"], 16, 1024, int)
        elif c == 14: CFG["tts_rate"] = ask_number("TTS rate (wpm)?", CFG["tts_rate"], 100, 250, int)
        elif c == 15: CFG["record_folder"] = ask_text("Record folder?", CFG["record_folder"])
        elif c == 16:
            for k, v in DEFAULTS.items(): CFG[k] = v
        save_cfg(); print("  saved!")


def help_flow():
    while True:
        header("❓ HELP")
        c = ask("Topic:", ["What is this?", "How do I record voice?",
                          "How do I get real speech data?",
                          "How does live subtitles work?",
                          "What data do I need?", "Troubleshooting", "Back"])
        if c in (-1, 7): return
        if c == 1:
            print("\n  A small transformer that learns from your files.")
            print("  Runs offline. Same architecture family as GPT.")
        elif c == 2:
            print("\n  🎙️  RECORD VOICE TO .WAV:")
            print("  Audio → 1) Record voice to .wav")
            print()
            print("  For each recording:")
            print("    · Press Enter")
            print("    · Say what you want")
            print("    · Recording stops automatically when you pause")
            print("    · You hear it back, then keep or retry")
            print("    · Optional caption → saved as .txt next to .wav")
            print()
            print("  Files are saved to:", CFG["record_folder"])
            print("  Change in: Audio → Microphone settings → Record folder")
            print()
            print("  These .wav files are perfect for training:")
            print("    Audio → Train STT  (speech → text)")
            print("    Audio → Train TTS  (text → speech)")
        elif c == 3:
            print("\n  FOR REAL SPEECH DATA (not beeps):")
            print()
            print("  Option 1 — Record your own voice:")
            print("    Audio → 1) Record voice to .wav")
            print("    → speak, saved as .wav + caption")
            print("    → most realistic, best quality")
            print()
            print("  Option 2 — Generate with TTS:")
            print("    pip install pyttsx3")
            print("    Audio → Generate speech samples")
            print()
            print("  IMPORTANT: beeps ≠ speech. A model trained on beeps")
            print("  cannot recognize words. You MUST use real speech.")
        elif c == 4:
            print("\n  LIVE SUBTITLES:")
            print("  · Continuous rolling window of mic audio")
            print("  · Every 200ms, transcribes the last 1 second")
            print("  · Shows text when it's been stable N times")
            print("  · Tune in Audio → Microphone settings")
            print("  · Needs STT model trained on 1-second speech clips")
        elif c == 5:
            print("\n  TEXT:    User:/Think:/Bot: lines, 100KB+")
            print("  STT:     .wav + .txt sidecar (real speech), 20+ files")
            print("  TTS:     .wav + .txt sidecar (real speech), 20+ files")
            print("  IMAGES:  .png/.jpg + .txt, 20+ files")
        elif c == 6:
            print("\n  'not enough tokens' → more text data")
            print("  'beeps won't work'  → use 'Record voice to .wav'")
            print("  'no trained model'  → train first")
            print("  live sounds weird   → lower threshold, adjust window")
        pause()


def whats_new_flow():
    header(f"🆕  WHAT'S NEW  (v{VERSION})")
    print("""
  v6.5 — RECORD VOICE TO .WAV

  NEW:
    🎙️  Record voice to .wav  — free-form recorder in Audio menu
       · Press Enter → speak → stops on pause → hear playback
       · Optional caption saved as .txt sidecar (for training)
       · Retry option if you flub the take
       · Files auto-numbered rec_001.wav, rec_002.wav, ...
       · Perfect for building STT/TTS training data

    New setting: record_folder (default samples/audio)
    Editable in Settings or Microphone settings

  RECOMMENDED WORKFLOW:
    1. Audio → 1) Record voice to .wav
       → record 20-50 short phrases with captions
    2. Audio → Train STT
       → point at the folder you recorded to
    3. Audio → Live subtitles
       → speak, watch it transcribe
""")
    pause()


def main():
    while True:
        clear()
        header(f"🔨  Forge v{VERSION}")
        dev = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"  device: {dev}  ·  models: {len(all_models())}  "
              f"·  thinking: {CFG['thinking_style']}")
        c = ask("What do you want to do?", [
            "💬  Text & Chat",
            "🎵  Audio",
            "🖼️   Image",
            "🔀  Any to Any",
            "🎓  Train a model",
            "🌐  Start server (HTML UI)",
            "📁  My models",
            "⚙️   Settings",
            "❓  Help",
            "🆕  What's new",
            "🚪  Quit"])
        if c in (-1, 11):
            print("\n  bye!\n"); return
        try:
            if c == 1: text_flow()
            elif c == 2: audio_flow()
            elif c == 3: image_flow()
            elif c == 4: any_to_any_flow()
            elif c == 5: train_flow()
            elif c == 6: server_flow()
            elif c == 7:
                header("📁 MY MODELS")
                m = ask("Action:", ["List", "Details", "Delete", "Backup", "Reset", "Back"])
                if m == 1: cmd_list_models()
                elif m == 2: cmd_model_details()
                elif m == 3: cmd_delete_model()
                elif m == 4: cmd_backup()
                elif m == 5: cmd_reset()
                if m != -1: pause()
            elif c == 8: settings_flow()
            elif c == 9: help_flow()
            elif c == 10: whats_new_flow()
        except KeyboardInterrupt:
            print("\n  (cancelled)"); pause()
        except Exception as e:
            print(f"\n  [!] {e}"); pause()


if __name__ == "__main__":
    main()
