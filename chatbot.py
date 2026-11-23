"""
chatbot.py — MINI AI v6.3
Run:  python chatbot.py

v6.3: LIVE SUBTITLES — continuous microphone transcription as you speak.
Keeps v6.2 (live mic, live TTS, voice conversation, mic settings).
"""

import glob, json, math, os, random, re, secrets, shutil, sys, threading, time
import torch, torch.nn as nn, torch.nn.functional as F

VERSION = "6.3"


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
# SAMPLES
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

User: what is 15 times 3
Think: 15 x 3 = 45
Bot: 15 times 3 is 45.

User: tell me a joke
Think: joke request, tell a short joke
Bot: Why did the computer go to the doctor? It had a virus.

User: goodbye
Think: farewell, say goodbye
Bot: Goodbye! Have a great day.

User: what is 100 divided by 4
Think: 100 / 4 = 25
Bot: 100 divided by 4 is 25.

User: who made you
Think: origin question, made with PyTorch
Bot: I was made with PyTorch as a small transformer.

User: what is python
Think: programming language, used for many things
Bot: Python is a programming language used for many things.

User: good morning
Think: greeting, wish good morning
Bot: Good morning! Have a wonderful day.

User: thanks
Think: thanks given, acknowledge
Bot: You are welcome.

User: what is 20 minus 6
Think: 20 - 6 = 14
Bot: 20 minus 6 is 14.

User: help
Think: help request, ask what they need
Bot: Sure, tell me what you need help with.

User: what is a transformer
Think: ML question, attention-based network
Bot: A transformer is a neural network based on attention.

User: good night
Think: farewell, say good night
Bot: Good night! Sleep well.

User: what is 9 times 9
Think: 9 x 9 = 81
Bot: 9 times 9 is 81.

User: what is 7 plus 8
Think: 7 + 8 = 15
Bot: 7 plus 8 is 15.

User: hi
Think: greeting, say hi back
Bot: Hi there! Nice to meet you.

User: what is the capital of France
Think: geography, capital is Paris
Bot: The capital of France is Paris.
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
        if shape == "sq":
            d.rectangle([20, 20, 108, 108], fill=color)
        else:
            d.ellipse([20, 20, 108, 108], fill=color)
        img.save(os.path.join(folder, f"{name}.png"))
        open(os.path.join(folder, f"{name}.txt"), "w").write(name.replace("_", " "))
    print(f"  created {len(samples)} images in {folder}/")
    return folder


def make_sample_audio():
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("  pip install soundfile")
        return None
    folder = "samples/audio"
    os.makedirs(folder, exist_ok=True)
    SR = 16000
    t = np.arange(SR) / SR
    for name, freq in [("a_low_beep", 220), ("a_mid_beep", 440),
                       ("a_high_beep", 880), ("a_very_high_beep", 1500)]:
        wav = 0.5 * np.sin(2 * np.pi * freq * t).astype("float32")
        sf.write(os.path.join(folder, f"{name}.wav"), wav, SR)
        open(os.path.join(folder, f"{name}.txt"), "w").write(name.replace("_", " "))
    print(f"  created 4 audio files in {folder}/")
    return folder


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING (compressed from v6.1 — same behavior)
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
        print("  no text")
        return None, None

    dialogue_mode = "User:" in text and "Bot:" in text
    has_think = "Think:" in text
    print(f"  text = {len(text):,}  style = {'dialogue' if dialogue_mode else 'free'}"
          f"{'  ✓ Think: format' if has_think else ''}")

    tok = None
    optimizer = None
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        model, tok, ckpt = load_ckpt(resume_from, device, need_optimizer=True)
        config = ckpt["config"]
        start_step = ckpt.get("training_step", 0)
        print(f"  resuming from step {start_step}")
    elif fine_tune_from and os.path.exists(fine_tune_from):
        model, tok, ckpt = load_ckpt(fine_tune_from, device)
        config = ckpt["config"]
        print(f"  fine-tuning existing")
    else:
        tok = BPE().train(text, 1024, max(64, len(text.encode()) // 2))
        bs = min(CFG["block_size"], max(32, len(text.encode()) // 10))
        n = len(text)
        if n < 3000:
            config = dict(n_embd=32, n_head=2, n_layer=1, block_size=bs)
        elif n < 15000:
            config = dict(n_embd=64, n_head=2, n_layer=2, block_size=bs)
        elif n < 100000:
            config = dict(n_embd=128, n_head=4, n_layer=4, block_size=bs)
        else:
            config = dict(n_embd=192, n_head=6, n_layer=4, block_size=bs)
        model = TextLM(vocab_size=len(tok.vocab), **config).to(device)

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    bs = config["block_size"]
    print(f"  tokens = {len(data):,}  params = {model.n_params():,}")
    if len(data) < bs + 4:
        print("  not enough tokens")
        return None, None

    n_val = int(len(data) * 0.1)
    train_data, val_data = (data[:-n_val], data[-n_val:]) if n_val >= bs + 2 else (data, None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"],
                                  betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    print(f"\n  training {steps} steps on {device}...\n")
    best_val = float("inf")
    for step in range(1, steps + 1):
        lr = CFG["learning_rate"] * (0.5 * (1 + math.cos(math.pi * step / steps))) + 3e-5
        for g in optimizer.param_groups:
            g["lr"] = lr
        ix = torch.randint(0, len(train_data) - bs - 1, (CFG["batch_size"],))
        x = torch.stack([train_data[i:i + bs] for i in ix]).to(device)
        y = torch.stack([train_data[i + 1:i + bs + 1] for i in ix]).to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

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
                                           "has_think": has_think}, optimizer, global_step)
                model.train()
            progress(step, steps, loss.item(), vmsg)
    print()
    save_ckpt(os.path.join(model_dir, "last.pt"), "text", model, tok, config,
              {"dialogue_mode": dialogue_mode, "has_think": has_think},
              optimizer, start_step + steps)
    if not os.path.exists(os.path.join(model_dir, "best.pt")):
        save_ckpt(os.path.join(model_dir, "best.pt"), "text", model, tok, config,
                  {"dialogue_mode": dialogue_mode, "has_think": has_think},
                  optimizer, start_step + steps)
    print(f"  saved -> {model_dir}/")
    return model, tok


def load_image_items(arg):
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("  pip install pillow")
        return []
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
        print("  pip install soundfile")
        return []
    items, n = [], AudioGen.SR
    for p in discover([arg], [".wav", ".mp3", ".flac", ".ogg"]):
        try:
            sc = os.path.splitext(p)[0] + ".txt"
            cap = open(sc).read().strip() if os.path.exists(sc) else \
                os.path.splitext(os.path.basename(p))[0].replace("_", " ")
            d, sr = sf.read(p)
            if d.ndim > 1:
                d = d.mean(axis=1)
            if sr != AudioGen.SR:
                r = sr / AudioGen.SR
                L = max(1, int(len(d) / r))
                ix = (np.arange(L) * r).astype(int)
                ix = ix[ix < len(d)]
                d = d[ix]
            d = np.pad(d, (0, n - len(d))) if len(d) < n else d[:n]
            items.append((cap, torch.from_numpy(d.astype("float32"))))
        except Exception as e:
            print(f"  skip {p}: {e}")
    return items


def _enc_caps(tok, caps, device, prefix="\x00"):
    seqs = [tok.encode(prefix + c + "\n")[:32] for c in caps]
    L = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s)
    return out.to(device)


def train_image(folder, steps, model_dir):
    device = get_device()
    items = load_image_items(folder)
    if len(items) < 4:
        print("  need 4+ images")
        return
    caps = " ".join(c for c, _ in items)
    tok = BPE().train(caps, 256, max(8, len(caps) // 4))
    config = dict(n_embd=64, hidden=128, latent=256)
    model = ImageGen(vocab_size=len(tok.vocab), **config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=0.01)
    model.train()
    print(f"\n  training {steps} steps...\n")
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(items), (min(CFG["batch_size"], len(items)),))
        b = [items[i] for i in ix.tolist()]
        imgs = torch.stack([x[1] for x in b]).to(device)
        pred = model(_enc_caps(tok, [x[0] for x in b], device))
        loss = F.mse_loss(pred, imgs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 100) == 0:
            progress(step, steps, loss.item())
    print()
    save_ckpt(os.path.join(model_dir, "best.pt"), "image", model, tok, config)
    save_ckpt(os.path.join(model_dir, "last.pt"), "image", model, tok, config)
    print(f"  saved -> {model_dir}/")


def train_tts(folder, steps, model_dir):
    device = get_device()
    items = load_audio_items(folder)
    if len(items) < 4:
        print("  need 4+ files")
        return
    caps = " ".join(c for c, _ in items)
    tok = BPE().train(caps, 256, max(8, len(caps) // 4))
    config = dict(n_embd=64, hidden=128, latent=128)
    model = AudioGen(vocab_size=len(tok.vocab), **config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=0.01)
    model.train()
    print(f"\n  training {steps} steps...\n")
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(items), (min(CFG["batch_size"], len(items)),))
        b = [items[i] for i in ix.tolist()]
        wavs = torch.stack([x[1] for x in b]).to(device)
        pred = model(_enc_caps(tok, [x[0] for x in b], device, prefix=""))
        loss = F.mse_loss(pred, wavs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 100) == 0:
            progress(step, steps, loss.item())
    print()
    save_ckpt(os.path.join(model_dir, "best.pt"), "tts", model, tok, config)
    save_ckpt(os.path.join(model_dir, "last.pt"), "tts", model, tok, config)
    print(f"  saved -> {model_dir}/")


def train_stt(folder, steps, model_dir):
    device = get_device()
    items = load_audio_items(folder)
    if len(items) < 4:
        print("  need 4+ files")
        return
    caps = " ".join(c for c, _ in items)
    tok = BPE().train(caps, 256, max(8, len(caps) // 4))
    config = dict(hidden=256)
    model = AudioToText(vocab_size=len(tok.vocab), **config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=0.01)
    model.train()
    print(f"\n  training {steps} steps...\n")
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(items), (min(CFG["batch_size"], len(items)),))
        b = [items[i] for i in ix.tolist()]
        wavs = torch.stack([x[1] for x in b]).to(device)
        _, loss = model(wavs, _enc_caps(tok, [x[0] for x in b], device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 100) == 0:
            progress(step, steps, loss.item())
    print()
    save_ckpt(os.path.join(model_dir, "best.pt"), "stt", model, tok, config)
    save_ckpt(os.path.join(model_dir, "last.pt"), "stt", model, tok, config)
    print(f"  saved -> {model_dir}/")


def train_caption(folder, steps, model_dir):
    device = get_device()
    items = load_image_items(folder)
    if len(items) < 4:
        print("  need 4+ images")
        return
    caps = " ".join(c for c, _ in items)
    tok = BPE().train(caps, 256, max(8, len(caps) // 4))
    config = dict(hidden=256)
    model = ImageCaption(vocab_size=len(tok.vocab), **config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=0.01)
    model.train()
    print(f"\n  training {steps} steps...\n")
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(items), (min(CFG["batch_size"], len(items)),))
        b = [items[i] for i in ix.tolist()]
        imgs = torch.stack([x[1] for x in b]).to(device)
        _, loss = model(imgs, _enc_caps(tok, [x[0] for x in b], device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 100) == 0:
            progress(step, steps, loss.item())
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
    if sd is None:
        return None
    import numpy as np

    SR = AudioGen.SR
    max_sec = max_sec if max_sec is not None else CFG["mic_max_sec"]
    silence_ms = silence_ms if silence_ms is not None else CFG["mic_silence_ms"]
    threshold = threshold if threshold is not None else CFG["mic_threshold"]

    chunk_ms = 50
    chunk_size = int(SR * chunk_ms / 1000)
    silent_limit = silence_ms // chunk_ms
    max_chunks = max_sec * 1000 // chunk_ms

    buf = []
    silent = 0
    started = False
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
                    started = True
                    silent = 0
                elif started:
                    silent += 1
                    if silent >= silent_limit:
                        break

                meter = "█" * min(20, int(energy * 200))
                status = "listening" if started else "waiting for sound"
                print(f"\r  🎤 {frames[i % len(frames)]} {status} [{meter:<20}]",
                      end="", flush=True)
                i += 1
    except Exception as e:
        print(f"\n  ⚠ mic error: {e}")
        return None

    print("\r" + " " * 60 + "\r", end="")
    if not buf:
        return None
    audio = np.concatenate(buf).flatten()
    if len(audio) < SR // 4:
        return None
    return audio


def transcribe_audio(model, tok, device, audio, sr=AudioGen.SR):
    import numpy as np
    if sr != AudioGen.SR:
        r = sr / AudioGen.SR
        L = max(1, int(len(audio) / r))
        ix = (np.arange(L) * r).astype(int)
        ix = ix[ix < len(audio)]
        audio = audio[ix]

    SR = AudioGen.SR
    results = []
    for start in range(0, len(audio), SR):
        chunk = audio[start:start + SR]
        if len(chunk) < SR:
            chunk = np.pad(chunk, (0, SR - len(chunk)))
        wav = torch.tensor(chunk.astype("float32")).unsqueeze(0).to(device)
        h = model.enc(wav).unsqueeze(0).repeat(2, 1, 1)
        cur = torch.zeros(1, 1, dtype=torch.long, device=device)
        out = []
        with torch.no_grad():
            for _ in range(32):
                o, h = model.gru(model.emb(cur), h)
                nxt = model.head(o[:, -1]).argmax(-1).item()
                if nxt in (0, 10):
                    break
                out.append(nxt)
                cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
        text = tok.decode(out).strip()
        if text:
            results.append(text)
    return " ".join(results).strip()


def transcribe_one_window(model, tok, device, window):
    """Fast single-window transcription for live subtitles (fewer tokens)."""
    import numpy as np
    SR = AudioGen.SR
    if len(window) < SR:
        window = np.pad(window, (0, SR - len(window)))
    else:
        window = window[-SR:]
    wav = torch.tensor(window.astype("float32")).unsqueeze(0).to(device)
    h = model.enc(wav).unsqueeze(0).repeat(2, 1, 1)
    cur = torch.zeros(1, 1, dtype=torch.long, device=device)
    out = []
    with torch.no_grad():
        for _ in range(16):
            o, h = model.gru(model.emb(cur), h)
            nxt = model.head(o[:, -1]).argmax(-1).item()
            if nxt in (0, 10):
                break
            out.append(nxt)
            cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
    return tok.decode(out).strip()


def play_audio(wav, sr=AudioGen.SR):
    sd = _check_sounddevice()
    if sd is None:
        return False
    try:
        sd.play(wav, sr)
        sd.wait()
        return True
    except Exception as e:
        print(f"  ⚠ speaker error: {e}")
        return False


def list_audio_devices():
    sd = _check_sounddevice()
    if sd is None:
        return
    print("\n  Audio devices:")
    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        default_out = sd.default.device[1]
        for i, d in enumerate(devices):
            in_ch = d["max_input_channels"]
            out_ch = d["max_output_channels"]
            marker_in = " ◀ default mic" if i == default_in and in_ch > 0 else ""
            marker_out = " ◀ default speakers" if i == default_out and out_ch > 0 else ""
            kinds = []
            if in_ch > 0:
                kinds.append("IN")
            if out_ch > 0:
                kinds.append("OUT")
            kind_str = "/".join(kinds) if kinds else "—"
            print(f"    [{i}] {d['name']} ({kind_str}){marker_in}{marker_out}")
    except Exception as e:
        print(f"    error listing devices: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# LIVE SUBTITLES — the new feature
# ═══════════════════════════════════════════════════════════════════════════
def live_subtitles_session(model_dir):
    """Continuous microphone transcription — subtitles appear as you speak."""
    sd = _check_sounddevice()
    if sd is None:
        return

    import numpy as np
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path):
        print("  no STT model trained yet")
        return

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
    print(f"  ║  model: {os.path.basename(path):<44}  ║")
    print(f"  ║  window: {window_sec:.1f}s   hop: {hop_sec:.2f}s   stable: {stable_need}x{'':<14}║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()
    print("  Speak normally. Lines appear when stable. Ctrl+C to stop.")
    print()
    print("  ──────────────────────────────────────────────────────")
    print()

    ring = np.zeros(window_size, dtype="float32")
    current_text = ""
    stable_count = 0
    last_committed = ""
    committed = []

    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=hop_size) as stream:
            while True:
                chunk, _ = stream.read(hop_size)
                chunk = chunk.flatten()

                if len(chunk) >= window_size:
                    ring = chunk[-window_size:]
                else:
                    ring = np.concatenate([ring[len(chunk):], chunk])

                energy = float(np.sqrt(np.mean(ring ** 2)))

                if energy < min_energy:
                    if current_text and stable_count >= stable_need:
                        if current_text != last_committed:
                            committed.append(current_text)
                            last_committed = current_text
                            ts = time.strftime("%H:%M:%S")
                            print(f"  [{ts}] {current_text}")
                    current_text = ""
                    stable_count = 0
                    continue

                text = transcribe_one_window(model, tok, device, ring)

                if not text:
                    continue

                if text == current_text:
                    stable_count += 1
                else:
                    current_text = text
                    stable_count = 1

                if stable_count >= stable_need and text != last_committed:
                    committed.append(text)
                    last_committed = text
                    ts = time.strftime("%H:%M:%S")
                    print(f"  [{ts}] {text}")
                    current_text = ""
                    stable_count = 0

    except KeyboardInterrupt:
        print("\n")
        print("  ──────────────────────────────────────────────────────")
        print(f"  stopped. {len(committed)} line(s) transcribed.")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# LIVE STT (push-to-talk)
# ═══════════════════════════════════════════════════════════════════════════
def stt_live_session(model_dir):
    if _check_sounddevice() is None:
        return
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path):
        print("  no STT model trained yet")
        return
    model, tok, ckpt = load_ckpt(path, device)

    print("\n  === speech → text  (LIVE MICROPHONE) ===")
    print(f"  ✓ model loaded: {path}")
    print(f"  Settings: max {CFG['mic_max_sec']}s, silence stop after {CFG['mic_silence_ms']}ms")
    print()
    print("  Press Enter to start recording, speak, stop automatically.")
    print("  'exit' to leave.\n")

    while True:
        try:
            cmd = input("  (Enter to record, or 'exit') > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd in ("exit", "quit", "q"):
            break
        audio = record_until_silence()
        if audio is None:
            print("  (nothing recorded)")
            continue
        print("  🧠 transcribing...", end="", flush=True)
        text = transcribe_audio(model, tok, device, audio)
        print(f"\r  📝 {text if text else '(could not understand)'}          \n")


# ═══════════════════════════════════════════════════════════════════════════
# LIVE TTS
# ═══════════════════════════════════════════════════════════════════════════
def tts_live_session(model_dir):
    if _check_sounddevice() is None:
        return
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path):
        print("  no TTS model trained yet")
        return
    model, tok, ckpt = load_ckpt(path, device)

    print("\n  === text → speech  (LIVE PLAYBACK) ===")
    print(f"  ✓ model loaded: {path}")
    print("  Type text, hear it play through your speakers. 'exit' to leave.\n")

    while True:
        try:
            p = input("  say> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not p:
            continue
        if p.lower() in ("exit", "quit"):
            break
        ids = tok.encode(p)[:32]
        if not ids:
            print("  (no known words)")
            continue
        with torch.no_grad():
            wav = model(torch.tensor([ids], dtype=torch.long, device=device))[0].cpu().numpy()
        print("  🔊 ", end="", flush=True)
        if play_audio(wav):
            print("\r  ✓ played                    ")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# VOICE CONVERSATION (speak → AI → speaks back)
# ═══════════════════════════════════════════════════════════════════════════
def voice_conversation_session():
    if _check_sounddevice() is None:
        return
    device = get_device()
    need = {
        "text": "checkpoints/text/best.pt",
        "stt": "checkpoints/stt/best.pt",
        "tts": "checkpoints/tts/best.pt",
    }
    missing = [k for k, p in need.items() if not os.path.exists(p)]
    if missing:
        print(f"\n  Missing models: {', '.join(missing)}")
        print("  Train all three first.")
        return

    print("  loading text model...")
    tm, ttk, tck = load_ckpt(need["text"], device)
    print("  loading STT model...")
    sm, stk, sck = load_ckpt(need["stt"], device)
    print("  loading TTS model...")
    am, atk, ack = load_ckpt(need["tts"], device)

    dialogue_mode = tck.get("dialogue_mode", False)
    has_think = tck.get("has_think", False)
    bs = tck["config"]["block_size"]
    conv = Conversation(persona=CFG.get("persona", ""))

    print("\n  === VOICE CONVERSATION ===")
    print("  Speak → AI thinks → AI speaks back. 'exit' to leave.\n")

    while True:
        try:
            cmd = input("  (Enter to talk, or 'exit') > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd in ("exit", "quit", "q"):
            break

        audio = record_until_silence()
        if audio is None:
            print("  (nothing heard)")
            continue

        print("  🧠 hearing...", end="", flush=True)
        user_text = transcribe_audio(sm, stk, device, audio)
        print(f"\r  👤 you said: {user_text or '(unclear)'}          ")
        if not user_text:
            continue

        thought = None
        if has_think and CFG["learning_think"] and CFG["thinking_style"] != "off":
            thought = real_thought(tm, ttk, user_text, bs)
            if thought:
                print(f"  💭 {thought}")

        print("  🤖 thinking...", end="", flush=True)
        prompt = conv.build_prompt(user_text, dialogue_mode,
                                   CFG["memory_turns"], thought=thought)
        apply_seed()
        reply = generate_text(tm, ttk, prompt, bs)
        reply = filter_output(reply)
        print(f"\r  🤖 bot: {reply or '...'}          ")
        conv.add("user", user_text)
        conv.add("bot", reply or "")

        if reply:
            ids = atk.encode(reply)[:32]
            if ids:
                with torch.no_grad():
                    wav = am(torch.tensor([ids], dtype=torch.long, device=device))[0].cpu().numpy()
                print("  🔊 speaking...", end="", flush=True)
                play_audio(wav)
                print("\r  ✓ done                    ")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# MIC SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
def audio_settings_flow():
    while True:
        header("🎙️  MICROPHONE SETTINGS")
        print(f"  silence threshold  : {CFG['mic_threshold']}   (lower = more sensitive)")
        print(f"  stop after silence : {CFG['mic_silence_ms']} ms")
        print(f"  max recording      : {CFG['mic_max_sec']} s")
        print(f"  subtitle window    : {CFG['subtitle_window']} s")
        print(f"  subtitle hop       : {CFG['subtitle_hop']} s")
        print(f"  subtitle stable    : {CFG['subtitle_stable']} x")
        c = ask("Change:", [
            "Silence threshold",
            "Stop after silence (ms)",
            "Max recording time (s)",
            "Subtitle window (s)",
            "Subtitle hop (s)",
            "Subtitle stability (repeats)",
            "List audio devices",
            "Test microphone (record & play back)",
            "Back",
        ])
        if c in (-1, 9):
            return
        if c == 1:
            CFG["mic_threshold"] = ask_number("Threshold?", CFG["mic_threshold"], 0.001, 0.5)
        elif c == 2:
            CFG["mic_silence_ms"] = ask_number("Silence ms?", CFG["mic_silence_ms"], 200, 3000, int)
        elif c == 3:
            CFG["mic_max_sec"] = ask_number("Max seconds?", CFG["mic_max_sec"], 1, 30, int)
        elif c == 4:
            CFG["subtitle_window"] = ask_number("Window seconds?", CFG["subtitle_window"], 0.5, 3.0)
        elif c == 5:
            CFG["subtitle_hop"] = ask_number("Hop seconds?", CFG["subtitle_hop"], 0.1, 1.0)
        elif c == 6:
            CFG["subtitle_stable"] = ask_number("Stable repeats?", CFG["subtitle_stable"], 1, 5, int)
        elif c == 7:
            list_audio_devices()
            pause()
            continue
        elif c == 8:
            print("\n  recording 3 seconds...")
            audio = record_until_silence(max_sec=3)
            if audio is None:
                print("  (nothing recorded)")
            else:
                print(f"  recorded {len(audio) / AudioGen.SR:.1f}s — playing back")
                play_audio(audio)
            pause()
            continue
        save_cfg()
        print("  saved!")


# ═══════════════════════════════════════════════════════════════════════════
# CLI CHAT
# ═══════════════════════════════════════════════════════════════════════════
def thinking_chat_session(model_dir, style="chat", title="Chat"):
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path):
        print("  no trained model")
        return

    model, tok, ckpt = load_ckpt(path, device)
    dialogue_mode = ckpt.get("dialogue_mode", False)
    has_think = ckpt.get("has_think", False)
    bs = ckpt["config"]["block_size"]
    conv = Conversation(persona=CFG.get("persona", ""))

    print(f"\n  ┌─ {title} ────────────────────────────────")
    print(f"  │ model thinks : {'yes' if has_think else 'no (train with Think: data)'}")
    print(f"  │ show thinking: {CFG['thinking_style']}")
    print(f"  │ memory       : {CFG['memory_turns']} turns")
    print(f"  └──────────────────────────────────────────")
    print("  commands: 'exit' · 'save' · 'clear' · 'style'\n")

    while True:
        try:
            user = input("  you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        low = user.lower()
        if low in ("exit", "quit"):
            break
        if low == "save":
            fn = conv.save()
            print(f"  ✓ saved -> {fn}")
            continue
        if low == "clear":
            conv = Conversation(persona=conv.persona)
            print("  ✓ memory cleared")
            continue
        if low == "style":
            cycle = {"off": "brief", "brief": "verbose", "verbose": "off"}
            CFG["thinking_style"] = cycle[CFG["thinking_style"]]
            save_cfg()
            print(f"  thinking style = {CFG['thinking_style']}")
            continue

        thought = None
        if has_think and CFG["learning_think"] and CFG["thinking_style"] != "off":
            thought = run_with_spinner(
                "thinking...",
                lambda u=user: real_thought(model, tok, u, bs))
            if thought and CFG["thinking_style"] != "off":
                print(f"  ┌─ model thought ─────────────────────")
                print(f"  │ {thought}")
                print(f"  └──────────────────────────────────────\n")

        prompt = conv.build_prompt(user, dialogue_mode, CFG["memory_turns"], thought=thought)
        if not dialogue_mode and style != "chat":
            prompt = format_prompt(user, style, dialogue_mode)

        apply_seed()
        reply = run_with_spinner(
            "composing reply...",
            lambda p=prompt: generate_text(model, tok, p, bs))

        reply = filter_output(reply)
        print(f"  bot> {reply or '...'}\n")
        conv.add("user", user)
        conv.add("bot", reply or "")

    if conv.messages:
        fn = conv.save()
        print(f"  ✓ auto-saved -> {fn}\n")


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE SESSION
# ═══════════════════════════════════════════════════════════════════════════
def image_session(model_dir):
    try:
        from PIL import Image
    except ImportError:
        print("  pip install pillow")
        return
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path):
        print("  no model")
        return
    model, tok, ckpt = load_ckpt(path, device)
    os.makedirs("my_pictures", exist_ok=True)
    print("\n  === image generator  ('exit') ===\n")
    while True:
        try:
            p = input("  prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not p:
            continue
        if p.lower() in ("exit", "quit"):
            break
        ids = tok.encode("\x00" + p + "\n")[:32]
        if not ids:
            print("  (no known words)\n")
            continue
        with torch.no_grad():
            img = model(torch.tensor([ids], dtype=torch.long, device=device))[0]
            img = img.cpu().permute(1, 2, 0).numpy()
        out = f"my_pictures/{int(time.time() * 1000)}.png"
        Image.fromarray((img * 255).clip(0, 255).astype("uint8")).resize(
            (256, 256), Image.NEAREST).save(out)
        print(f"  saved -> {out}\n")


# ═══════════════════════════════════════════════════════════════════════════
# FILE STT / CAPTION
# ═══════════════════════════════════════════════════════════════════════════
def stt_file_session(model_dir):
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("  pip install soundfile")
        return
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path):
        print("  no STT model trained yet")
        return
    model, tok, ckpt = load_ckpt(path, device)

    print("\n  === transcribe an audio FILE ===")
    print(f"  ✓ model loaded: {path}")
    print("  Audio files end in: .wav, .mp3, .flac, .ogg\n")

    audio_files = find_files_with_ext([".wav", ".mp3", ".flac", ".ogg"])
    if not audio_files:
        print("  No audio files found in this folder.")
        c = ask("What do you want?", [
            "Make sample beep sounds to test",
            "I'll type a path to an audio file elsewhere",
            "Exit",
        ])
        if c == -1 or c == 3:
            return
        if c == 1:
            folder = make_sample_audio()
            if not folder:
                return
            audio_files = find_files_with_ext([".wav", ".mp3", ".flac", ".ogg"])

    if audio_files:
        print(f"  Found {len(audio_files)} file(s). Examples:")
        for f in audio_files[:6]:
            print(f"    {f}")
        print()

    print("  Type a path, or drag a file in. 'exit' to quit.\n")
    while True:
        try:
            p = input("  file> ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not p:
            continue
        if p.lower() in ("exit", "quit"):
            break
        if not os.path.exists(p):
            print(f"  ⚠ not found: {p}")
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext not in (".wav", ".mp3", ".flac", ".ogg"):
            print(f"  ⚠ not audio: {ext}. Use .wav/.mp3/.flac/.ogg")
            if ext == ".pt":
                print(f"    (that's the model — not audio)")
            continue
        try:
            d, sr = sf.read(p)
        except Exception as e:
            print(f"  ⚠ couldn't read: {e}")
            continue
        text = transcribe_audio(model, tok, device, d, sr)
        print(f"  -> {text if text else '(could not understand)'}\n")


def caption_session(model_dir):
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("  pip install pillow")
        return
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path):
        print("  no caption model trained yet")
        return
    model, tok, ckpt = load_ckpt(path, device)

    print("\n  === image → text ===")
    print(f"  ✓ model loaded: {path}")
    print("  Image files end in: .png, .jpg, .jpeg, .bmp\n")

    img_files = find_files_with_ext([".png", ".jpg", ".jpeg", ".bmp"])
    if not img_files:
        print("  No image files found.")
        c = ask("What do you want?", [
            "Make sample images to test",
            "I'll type a path to an image elsewhere",
            "Exit",
        ])
        if c == -1 or c == 3:
            return
        if c == 1:
            folder = make_sample_images()
            if not folder:
                return
            img_files = find_files_with_ext([".png", ".jpg", ".jpeg", ".bmp"])

    if img_files:
        print(f"  Found {len(img_files)} file(s). Examples:")
        for f in img_files[:6]:
            print(f"    {f}")
        print()

    print("  Type a path, or drag a file in. 'exit' to quit.\n")
    while True:
        try:
            p = input("  image> ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not p:
            continue
        if p.lower() in ("exit", "quit"):
            break
        if not os.path.exists(p):
            print(f"  ⚠ not found: {p}")
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".bmp"):
            print(f"  ⚠ not image: {ext}. Use .png/.jpg/.jpeg/.bmp")
            if ext == ".pt":
                print(f"    (that's the model — not an image)")
            continue
        try:
            im = Image.open(p).convert("RGB").resize((32, 32))
        except Exception as e:
            print(f"  ⚠ couldn't open: {e}")
            continue
        arr = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
        h = model.enc(arr).unsqueeze(0).repeat(2, 1, 1)
        cur = torch.zeros(1, 1, dtype=torch.long, device=device)
        out = []
        with torch.no_grad():
            for _ in range(32):
                o, h = model.gru(model.emb(cur), h)
                nxt = model.head(o[:, -1]).argmax(-1).item()
                if nxt in (0, 10):
                    break
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
            if os.path.exists(p):
                ms.append((mode, f, p))
    return ms


def cmd_list_models():
    header("MY MODELS")
    ms = all_models()
    if not ms:
        print("  (no trained models yet)")
        return
    print(f"  {'mode':<10} {'file':<10} {'size':>10}   path")
    print("  " + "-" * 60)
    for m, f, p in ms:
        print(f"  {m:<10} {f:<10} {human_size(os.path.getsize(p)):>10}   {p}")


def cmd_model_details():
    header("MODEL DETAILS")
    ms = all_models()
    if not ms:
        print("  (no models)")
        return
    c = ask("Which?", [f"{m}/{f}" for m, f, _ in ms])
    if c == -1:
        return
    m, f, p = ms[c - 1]
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        print(f"\n  file:    {p}")
        print(f"  size:    {human_size(os.path.getsize(p))}")
        print(f"  mode:    {ck['mode']}")
        print(f"  vocab:   {len(ck['tokenizer_merges']) + 256}")
        print(f"  step:    {ck.get('training_step', '?')}")
        print(f"  config:  {ck['config']}")
        if "has_think" in ck:
            print(f"  think:   {ck['has_think']}")
    except Exception as e:
        print(f"  error: {e}")


def cmd_delete_model():
    header("DELETE MODEL")
    ms = all_models()
    if not ms:
        print("  (no models)")
        return
    c = ask("Which to delete?", [f"{m}/{f}" for m, f, _ in ms])
    if c == -1:
        return
    m, f, p = ms[c - 1]
    if ask(f"Delete {p}?", ["Yes", "No"]) == 1:
        os.remove(p)
        print(f"  deleted {p}")


def cmd_backup():
    if not os.path.isdir("checkpoints"):
        print("  nothing to back up")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"backup_{stamp}"
    shutil.copytree("checkpoints", name)
    print(f"  saved -> {name}/")


def cmd_reset():
    header("RESET EVERYTHING")
    if ask("Delete ALL models?", ["Yes", "No"]) == 1:
        for d in ["checkpoints", "chats"]:
            if os.path.isdir(d):
                shutil.rmtree(d)
        if os.path.exists(SETTINGS_FILE):
            os.remove(SETTINGS_FILE)
        print("  reset complete")


# ═══════════════════════════════════════════════════════════════════════════
# SERVER
# ═══════════════════════════════════════════════════════════════════════════
SERVER_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mini AI</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#1a1a1a;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
header{padding:12px 16px;background:#252525;border-bottom:1px solid #333;display:flex;align-items:center;gap:12px}
header h1{font-size:16px;font-weight:600}header .status{font-size:12px;color:#888}header button{margin-left:auto}
main{flex:1;display:flex;overflow:hidden}
#chat{flex:1;display:flex;flex-direction:column;overflow:hidden}
#messages{flex:1;overflow-y:auto;padding:16px}
.msg{margin-bottom:16px;padding:10px 14px;border-radius:8px;max-width:80%;white-space:pre-wrap;word-wrap:break-word;line-height:1.5}
.user{background:#2a4a7a;margin-left:auto}.bot{background:#2a2a2a}
.think{background:#1f2a1f;border-left:3px solid #4a7a4a;font-size:13px;color:#b0b0b0;margin-bottom:6px;padding:8px 12px;border-radius:6px}
.think-label{font-size:11px;color:#6a9a6a;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
#input-area{padding:12px;background:#252525;border-top:1px solid #333;display:flex;gap:8px}
#input{flex:1;padding:10px 14px;background:#1a1a1a;border:1px solid #333;border-radius:6px;color:#e0e0e0;font-family:inherit;font-size:14px;resize:none;max-height:150px}
#input:focus{outline:none;border-color:#4a7aa5}
button{padding:10px 16px;background:#4a7aa5;color:#fff;border:none;border-radius:6px;cursor:pointer;font-family:inherit;font-size:14px}
button:hover{background:#5a8ab5}button:disabled{background:#444;cursor:not-allowed}
aside{width:280px;background:#202020;border-left:1px solid #333;padding:16px;overflow-y:auto;font-size:13px}
aside h3{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;margin-top:16px}
aside h3:first-child{margin-top:0}aside label{display:block;margin-bottom:8px;color:#aaa}
aside input[type=number],aside input[type=text],aside select{width:100%;padding:6px 10px;background:#1a1a1a;border:1px solid #333;border-radius:4px;color:#e0e0e0;font-family:inherit;font-size:13px}
aside .row{display:flex;justify-content:space-between;margin-bottom:6px;color:#aaa}aside .row span:last-child{color:#e0e0e0}
.spinner{display:inline-block;width:12px;height:12px;border:2px solid #555;border-top-color:#4a7aa5;border-radius:50%;animation:spin .8s linear infinite;margin-right:6px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
</style></head><body>
<header><h1>&#127775; Mini AI</h1><span class="status" id="status">Ready</span><button onclick="newChat()">New Chat</button></header>
<main>
<div id="chat"><div id="messages"></div>
<div id="input-area"><textarea id="input" rows="1" placeholder="Type a message... (Enter to send)"></textarea><button id="send" onclick="send()">Send</button></div></div>
<aside>
<h3>Model</h3><div id="model-info">loading...</div>
<h3>Settings</h3>
<label>Temperature <input type="number" id="temp" min="0.05" max="2" step="0.05"></label>
<label>Top-K <input type="number" id="topk" min="0" max="500" step="10"></label>
<label>Max tokens <input type="number" id="maxtok" min="10" max="2000" step="10"></label>
<label>Memory turns <input type="number" id="memory" min="0" max="50" step="1"></label>
<h3>Thinking</h3>
<label>Style <select id="thinkstyle"><option value="off">Off</option><option value="brief">Brief</option><option value="verbose">Verbose</option></select></label>
<h3>Persona</h3>
<input type="text" id="persona" placeholder="e.g. a pirate">
</aside></main>
<script>
var SESSION='web-'+Math.random().toString(36).slice(2,10);
function apiHeaders(){return {'Content-Type':'application/json'};}
async function send(){var input=document.getElementById('input');var msg=input.value.trim();if(!msg)return;input.value='';input.style.height='auto';addMsg('user',msg);setStatus(true,'thinking...');document.getElementById('send').disabled=true;try{var r=await fetch('/api/chat',{method:'POST',headers:apiHeaders(),body:JSON.stringify({message:msg,session:SESSION})});var d=await r.json();setStatus(false,'Ready');if(d.thought)addMsg('think',d.thought);addMsg('bot',d.reply||'...');}catch(e){setStatus(false,'Error');addMsg('bot','[error: '+e.message+']');}finally{document.getElementById('send').disabled=false;input.focus();}}
function addMsg(role,text){var div=document.createElement('div');div.className='msg '+(role==='think'?'think':role);if(role==='think'){div.innerHTML='<div class="think-label">thought</div>'+esc(text);}else{div.textContent=text;}document.getElementById('messages').appendChild(div);div.scrollIntoView({behavior:'smooth',block:'end'});}
function esc(s){return s.replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})}
function setStatus(loading,text){document.getElementById('status').innerHTML=(loading?'<span class="spinner"></span>':'')+text;}
async function newChat(){await fetch('/api/reset',{method:'POST',headers:apiHeaders(),body:JSON.stringify({session:SESSION})});document.getElementById('messages').innerHTML='';}
async function saveSetting(key,val){await fetch('/api/settings',{method:'POST',headers:apiHeaders(),body:JSON.stringify({[key]:val})});}
async function loadSettings(){try{var r=await fetch('/api/settings',{headers:apiHeaders()});var s=await r.json();document.getElementById('temp').value=s.temperature;document.getElementById('topk').value=s.top_k;document.getElementById('maxtok').value=s.max_new_tokens;document.getElementById('memory').value=s.memory_turns;document.getElementById('thinkstyle').value=s.thinking_style;document.getElementById('persona').value=s.persona||'';}catch(e){}}
async function loadModelInfo(){try{var r=await fetch('/api/model',{headers:apiHeaders()});var m=await r.json();document.getElementById('model-info').innerHTML='<div class="row"><span>params</span><span>'+(m.params||0).toLocaleString()+'</span></div>'+'<div class="row"><span>vocab</span><span>'+(m.vocab||'?')+'</span></div>'+'<div class="row"><span>think</span><span>'+(m.has_think?'&#10003;':'&#10007;')+'</span></div>'+'<div class="row"><span>step</span><span>'+(m.step||'?')+'</span></div>';}catch(e){document.getElementById('model-info').textContent='error'}}
document.getElementById('temp').onchange=function(){saveSetting('temperature',parseFloat(this.value))};
document.getElementById('topk').onchange=function(){saveSetting('top_k',parseInt(this.value))};
document.getElementById('maxtok').onchange=function(){saveSetting('max_new_tokens',parseInt(this.value))};
document.getElementById('memory').onchange=function(){saveSetting('memory_turns',parseInt(this.value))};
document.getElementById('thinkstyle').onchange=function(){saveSetting('thinking_style',this.value)};
document.getElementById('persona').onchange=function(){saveSetting('persona',this.value)};
document.getElementById('input').addEventListener('keydown',function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
document.getElementById('input').addEventListener('input',function(){this.style.height='auto';this.style.height=Math.min(this.scrollHeight,150)+'px';});
loadSettings();loadModelInfo();document.getElementById('input').focus();
</script></body></html>"""


def start_server(port=8000, api_key=None, model_dir="checkpoints/text"):
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    device = get_device()
    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path):
        print(f"  no model at {path}")
        return
    print(f"  loading model...")
    model, tok, ckpt = load_ckpt(path, device)
    dialogue_mode = ckpt.get("dialogue_mode", False)
    has_think = ckpt.get("has_think", False)
    bs = ckpt["config"]["block_size"]
    n_params = model.n_params()
    step = ckpt.get("training_step", 0)
    print(f"  loaded: {n_params:,} params, vocab={len(tok.vocab)}")
    conversations = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _check_auth(self):
            if not api_key: return True
            return self.headers.get("Authorization", "") == f"Bearer {api_key}"
        def _send(self, code, body, ctype="application/json"):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body.encode() if isinstance(body, str) else body)
        def _json(self, d, code=200): self._send(code, json.dumps(d))
        def do_OPTIONS(self): self._send(200, "")
        def do_GET(self):
            if self.path in ("/", "/index.html"): self._send(200, SERVER_HTML, "text/html; charset=utf-8")
            elif self.path == "/api/model":
                if not self._check_auth(): return self._json({"error":"unauthorized"},401)
                self._json({"params": n_params, "vocab": len(tok.vocab),
                            "has_think": has_think, "step": step, "dialogue_mode": dialogue_mode})
            elif self.path == "/api/settings":
                if not self._check_auth(): return self._json({"error":"unauthorized"},401)
                self._json(CFG)
            elif self.path == "/v1/models":
                self._json({"object":"list","data":[{"id":"mini","object":"model","owned_by":"local"}]})
            else: self._send(404, "not found", "text/plain")
        def do_POST(self):
            if not self._check_auth(): return self._json({"error":"unauthorized"},401)
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n)) if n else {}
            except Exception: return self._json({"error":"bad json"},400)
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
                    apply_seed()
                    reply = filter_output(generate_text(model, tok, prompt, bs))
                    conv.add("user", msg); conv.add("bot", reply)
                self._json({"reply": reply, "thought": thought})
            elif self.path == "/api/settings":
                for k,v in body.items():
                    if k in CFG: CFG[k]=v
                save_cfg(); self._json({"ok": True})
            elif self.path == "/api/reset":
                s = body.get("session","default")
                with lock:
                    if s in conversations: del conversations[s]
                self._json({"ok": True})
            elif self.path == "/v1/chat/completions":
                msgs = body.get("messages",[])
                if not msgs: return self._json({"error":"no messages"},400)
                user_msg=""; sys_prompt=""
                for m in msgs:
                    if m.get("role")=="user": user_msg = m.get("content","")
                    elif m.get("role")=="system": sys_prompt = m.get("content","")
                if not user_msg: return self._json({"error":"no user message"},400)
                with lock:
                    conv = Conversation(persona=sys_prompt or CFG["persona"])
                    for m in msgs[:-1]:
                        if m.get("role")=="user": conv.add("user", m.get("content",""))
                        elif m.get("role")=="assistant": conv.add("bot", m.get("content",""))
                    prompt = conv.build_prompt(user_msg, dialogue_mode, CFG["memory_turns"])
                    apply_seed()
                    reply = filter_output(generate_text(model, tok, prompt, bs))
                self._json({"id": f"chatcmpl-{int(time.time()*1000)}","object":"chat.completion",
                            "created": int(time.time()),"model": body.get("model","mini"),
                            "choices":[{"index":0,"message":{"role":"assistant","content":reply},"finish_reason":"stop"}],
                            "usage":{"prompt_tokens": len(tok.encode(user_msg)),
                                     "completion_tokens": len(tok.encode(reply)),
                                     "total_tokens": len(tok.encode(user_msg))+len(tok.encode(reply))}})
            else: self._send(404, "not found", "text/plain")

    try: server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    except OSError as e: print(f"  couldn't bind: {e}"); return
    print()
    print(f"  ══════════════════════════════════════════")
    print(f"  🌐  Server running")
    print(f"  ══════════════════════════════════════════")
    print(f"  UI:      http://localhost:{port}")
    print(f"  OpenAI:  http://localhost:{port}/v1/chat/completions")
    if api_key: print(f"  API key: {api_key}")
    print(f"  ══════════════════════════════════════════")
    print(f"  press Ctrl+C to stop\n")
    try: server.serve_forever()
    except KeyboardInterrupt: print("\n  stopping...")
    finally: server.shutdown(); print("  stopped")


def server_flow():
    header("🌐  START SERVER")
    model_dir = "checkpoints/text"
    if not os.path.exists(os.path.join(model_dir, "best.pt")):
        print("\n  No text model trained yet.")
        pause(); return
    port = ask_number("Port?", 8000, 1024, 65535, int)
    key_choice = ask("Require API key?", [
        "No — open", "Yes — generate random", "Yes — I'll type it",
    ])
    if key_choice == -1: return
    api_key = None
    if key_choice == 2:
        api_key = secrets.token_hex(16)
        print(f"\n  API key: {api_key}")
    elif key_choice == 3:
        api_key = ask_text("API key", "") or None
    print(f"\n  starting on port {port}...")
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
                    "Train new model", "Fine-tune (smarter)",
                    "Show model info", "Back"],
                subtitle=f"status: {'✓ trained' if has else '✗ no model'}")
        if c in (-1, 12): return
        if 1 <= c <= 8:
            if not has:
                print("\n  No model yet — train one first."); pause(); continue
            name, style, temp = TEXT_MODES[c]
            orig = CFG["temperature"]
            if style == "chat": CFG["temperature"] = temp
            thinking_chat_session(model_dir, style, name)
            CFG["temperature"] = orig
        elif c == 9:
            data = ask_path("Training text?", "data.txt")
            if not (os.path.exists(data) or glob.glob(data)):
                print("  not found, making sample"); data = make_sample_text()
            steps = ask_preset()
            if steps: train_text(data, steps, model_dir)
            pause()
        elif c == 10:
            if not has: print("  nothing to fine-tune"); pause(); continue
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
        c = ask("Pick:", [
            f"🎤 Live subtitles       {'(needs STT)' if not stt_ok else '✓'}",
            f"🎤 Live speech → text   {'✓' if stt_ok else '✗'}",
            f"🔊 Live text → speech   {'✓' if tts_ok else '✗'}",
            f"💬 Voice conversation   {'✓' if (stt_ok and tts_ok and os.path.exists('checkpoints/text/best.pt')) else '✗'}",
            "📁 Transcribe an audio file",
            "🎙️  Microphone settings",
            "Train TTS",
            "Train STT",
            "Back",
        ])
        if c in (-1, 9): return
        if c == 1:
            if not stt_ok: print("  train STT first"); pause()
            else: live_subtitles_session("checkpoints/stt")
        elif c == 2:
            if not stt_ok: print("  train STT first"); pause()
            else: stt_live_session("checkpoints/stt")
        elif c == 3:
            if not tts_ok: print("  train TTS first"); pause()
            else: tts_live_session("checkpoints/tts")
        elif c == 4:
            voice_conversation_session()
        elif c == 5:
            if not stt_ok: print("  train STT first"); pause()
            else: stt_file_session("checkpoints/stt")
        elif c == 6:
            audio_settings_flow()
        elif c == 7:
            folder = ask_path("Audio folder?", "samples/audio")
            if not os.path.isdir(folder): folder = make_sample_audio() or ""
            if folder:
                steps = ask_preset()
                if steps: train_tts(folder, steps, "checkpoints/tts")
            pause()
        elif c == 8:
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
            if not os.path.isdir(folder): folder = make_sample_audio() or ""
            if folder:
                steps = ask_preset()
                if steps: train_tts(folder, steps, "checkpoints/tts")
            pause()
        elif c == 3:
            folder = ask_path("Audio folder?", "samples/audio")
            if not os.path.isdir(folder): folder = make_sample_audio() or ""
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
        print(f"  banned_words    : {len(CFG['banned_words'])} word(s)")
        print(f"  seed            : {CFG['seed']}")
        print(f"  device          : {CFG['device']}")
        print(f"  batch_size      : {CFG['batch_size']}")
        print(f"  learning_rate   : {CFG['learning_rate']}")
        print(f"  block_size      : {CFG['block_size']}")
        c = ask("Change:", [
            "Temperature", "Top-K", "Max tokens", "Memory turns",
            "Thinking style", "Use learned Think: lines",
            "Persona", "Banned words", "Seed",
            "Device", "Batch size", "Learning rate", "Block size",
            "Reset to defaults", "Back"])
        if c in (-1, 15): return
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
        elif c == 7: CFG["persona"] = ask_text("Persona (blank = none)", CFG["persona"])
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
        elif c == 14:
            for k, v in DEFAULTS.items(): CFG[k] = v
        save_cfg()
        print("  saved!")


def help_flow():
    while True:
        header("❓ HELP")
        c = ask("Topic:", ["What is this?", "How does thinking work?",
                          "How does live subtitles work?",
                          "What data do I need?", "Troubleshooting", "Back"])
        if c in (-1, 6): return
        if c == 1:
            print("\n  A real (small) transformer that learns from your files.")
            print("  Runs offline. Same architecture family as GPT.")
        elif c == 2:
            print("\n  THINKING is real when the model has been trained on Think: data.")
            print("  Spinner duration = actual generation time.")
        elif c == 3:
            print("\n  LIVE SUBTITLES:")
            print("  · Continuously records from your mic in a rolling window")
            print("  · Every hop (200ms) it transcribes the last window (1s)")
            print("  · When the same text appears 'stable' times in a row, it's printed")
            print("  · Tune in Audio → Microphone settings:")
            print("    - window: how much audio to look at (bigger = more context)")
            print("    - hop: how often to check (smaller = more responsive)")
            print("    - stable: how many repeats before showing (bigger = cleaner)")
            print("  · Needs the STT model trained on short audio clips (1s)")
        elif c == 4:
            print("\n  CHAT:    User: / Think: / Bot: lines, 100KB+")
            print("  STT:     .wav + .txt sidecar, 20+ files")
            print("  TTS:     .wav + .txt sidecar, 20+ files")
            print("  IMAGES:  folder of .png/.jpg, 20+ files")
        elif c == 5:
            print("\n  'not enough tokens'  → add more data")
            print("  'no trained model'   → train first")
            print("  live sounds weird    → lower threshold, adjust window")
            print("  slow training        → settings → device → cuda")
        pause()


def whats_new_flow():
    header(f"🆕  WHAT'S NEW  (v{VERSION})")
    print("""
  v6.3 — LIVE SUBTITLES

  NEW:
    🎤 Live subtitles      — continuous mic, lines appear as you speak
    🎤 Rolling window      — every 200ms transcribes the last 1 second
    🎤 Stability filter    — only shows text that repeats (reduces flicker)
    🎙️  Subtitle tuning    — window/hop/stability in mic settings

  KEPT from v6.2:
    🎤 Live speech → text  — push-to-talk mic
    🔊 Live text → speech  — speak into speakers
    💬 Voice conversation  — full voice loop
    🎙️  Mic settings       — threshold, silence, devices, test

  HOW TO USE LIVE SUBTITLES:
    1. Train an STT model (Train → Speech → text)
    2. Audio → Live subtitles
    3. Speak normally. Lines appear when stable. Ctrl+C to stop.
""")
    pause()


def main():
    while True:
        clear()
        header(f"🌟  MINI AI v{VERSION}")
        dev = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"  device: {dev}  ·  models: {len(all_models())}  "
              f"·  thinking: {CFG['thinking_style']}")
        c = ask("What do you want to do?", [
            "💬  Text & Chat",
            "🎵  Audio",
            "🖼️   Image",
            "🔀  Any to Any",
            "🎓  Train a model",
            "🌐  Start server (HTML UI on a port)",
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