"""Pretrained-model embeddings and model outputs (catalogue sections 8 and 9) on the fixed excerpts.

Each model is its own extractor so embeddings can weight or drop them independently:
effnet, maest, musicnn, clap, mert, muq (embeddings) and semantic (Essentia classifier
heads + CLAP zero-shot similarities). Dropped for speed/redundancy (see TODO.md): VGGish,
YAMNet, PANNs, AST, OpenL3, Encodec latents, the Whisper language label.
"""

import json

import numpy as np

from .. import config
from .audio import Audio

VERSION = 1
ESS_MODELS = config.DATA_DIR / "models" / "essentia"
CHUNK_S = 10.0  # transformer models: 10 s chunks of each excerpt

EFFNET_HEADS = [
    "genre_discogs400", "mtg_jamendo_genre", "mtg_jamendo_moodtheme", "mtg_jamendo_instrument",
    "mood_happy", "mood_sad", "mood_relaxed", "mood_aggressive", "mood_party", "mood_acoustic", "mood_electronic",
    "danceability", "nsynth_acoustic_electronic", "nsynth_bright_dark", "nsynth_reverb", "timbre",
    "gender", "voice_instrumental", "approachability_regression", "engagement_regression",
]
MUSICNN_HEADS = ["deam", "emomusic"]  # arousal / valence regressors
CLAP_PROMPTS = ["strings", "piano", "brass", "woodwinds", "acoustic guitar", "electric guitar",
                "drums", "synthesizer", "singing voice", "choir"]


ESSENTIA_BASE = "https://essentia.upf.edu/models/"
ESSENTIA_DIRS = {  # model -> directory on ESSENTIA_BASE (the .pb and .json sit side by side)
    "discogs-effnet-bs64-1": "feature-extractors/discogs-effnet",
    "discogs-maest-30s-pw-2": "feature-extractors/maest",
    "msd-musicnn-1": "feature-extractors/musicnn",
    "deeptemp-k16-3": "tempo/tempocnn",  # used by the dsp stage
    **{f"{h}-discogs-effnet-1": f"classification-heads/{h.removesuffix('_regression')}" for h in EFFNET_HEADS},
    **{f"{h}-msd-musicnn-2": f"classification-heads/{h}" for h in MUSICNN_HEADS},
}


def required_model_files() -> list:
    return [ESS_MODELS / f"{m}{ext}" for m in ESSENTIA_DIRS for ext in (".pb", ".json")]


def download_models(log=print) -> int:
    """Fetch missing Essentia model files into data/models/essentia/. Returns the number downloaded."""
    import requests

    ESS_MODELS.mkdir(parents=True, exist_ok=True)
    n = 0
    for model, d in ESSENTIA_DIRS.items():
        for ext in (".pb", ".json"):
            path = ESS_MODELS / f"{model}{ext}"
            if path.exists():
                continue
            url = f"{ESSENTIA_BASE}{d}/{model}{ext}"
            log(f"  {url}")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            tmp = path.with_suffix(ext + ".part")
            tmp.write_bytes(r.content)
            tmp.rename(path)
            n += 1
    return n


def chunks(y, sr, sec=CHUNK_S):
    n = int(sr * sec)
    return [y[i:i + n] for i in range(0, max(len(y) - n // 2, 1), n)]


def load_muq():
    """MuQ-large with a transformers-v5 shim for its Wav2Vec2Conformer encoder.

    MuQ builds the encoder from an EasyDict config (v5 needs `_attn_implementation`) and
    reads `out["hidden_states"]`, which v5 no longer returns from the bare encoder. We
    record them with hooks, matching v4 (verified bit-identical to transformers 4.57).
    """
    from muq import MuQ

    mq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").eval()
    enc = mq.model.conformer
    enc.config._attn_implementation = "eager"
    hidden = []
    for layer in enc.layers:
        layer.register_forward_pre_hook(lambda mod, args: hidden.append(args[0]))
    forward = enc.forward

    def forward_with_hidden(*args, output_hidden_states=False, **kw):
        hidden.clear()
        out = forward(*args, **kw)
        states = (*hidden, out.last_hidden_state)
        hidden.clear()
        return {"last_hidden_state": out.last_hidden_state, "hidden_states": states}
    enc.forward = forward_with_hidden
    return mq


def _ess_head(name):
    import essentia.standard as es

    meta = json.load(open(ESS_MODELS / f"{name}.json"))
    outs = meta["schema"]["outputs"]
    out = next((o["name"] for o in outs if o.get("output_purpose") == "predictions"), outs[0]["name"])
    model = es.TensorflowPredict2D(graphFilename=str(ESS_MODELS / f"{name}.pb"),
                                   input=meta["schema"]["inputs"][0]["name"], output=out)
    return model, meta.get("classes") or []


class Stage:
    name = "models"
    outputs = [(e, VERSION) for e in ("effnet", "maest", "musicnn", "semantic", "clap", "mert", "muq")]

    def load(self):
        import essentia.standard as es
        from transformers import AutoModel, ClapModel, ClapProcessor, Wav2Vec2FeatureExtractor

        missing = [str(p) for p in required_model_files() if not p.exists()]
        if missing:
            raise RuntimeError("missing Essentia model files (download from https://essentia.upf.edu/models/):\n  "
                               + "\n  ".join(missing))
        self.effnet = es.TensorflowPredictEffnetDiscogs(graphFilename=str(ESS_MODELS / "discogs-effnet-bs64-1.pb"),
                                                        output="PartitionedCall:1")
        self.musicnn = es.TensorflowPredictMusiCNN(graphFilename=str(ESS_MODELS / "msd-musicnn-1.pb"), output="model/dense/BiasAdd")
        meta = json.load(open(ESS_MODELS / "discogs-maest-30s-pw-2.json"))
        outs = meta["schema"]["outputs"]
        self.maest = es.TensorflowPredictMAEST(graphFilename=str(ESS_MODELS / "discogs-maest-30s-pw-2.pb"),
                                               output=next((o["name"] for o in outs if o.get("output_purpose") == "embeddings"), outs[0]["name"]))
        self.heads = {h: _ess_head(f"{h}-discogs-effnet-1") for h in EFFNET_HEADS}
        self.mheads = {h: _ess_head(f"{h}-msd-musicnn-2") for h in MUSICNN_HEADS}
        self.clap_proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
        self.mert_proc = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        self.mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).eval()
        self.muq = load_muq()
        self._clap_text = None

    def process(self, a: Audio) -> dict:
        import torch

        ex16 = a.excerpts(16000)
        eff = np.concatenate([np.asarray(self.effnet(x)) for x in ex16])          # (frames, 1280)
        mnn = np.concatenate([np.asarray(self.musicnn(x)) for x in ex16])         # (frames, 200)
        maest = np.concatenate([(lambda e: e.reshape(-1, e.shape[-1]))(np.asarray(self.maest(x))) for x in ex16])
        out = {"effnet": eff.mean(0), "maest": maest.mean(0), "musicnn": mnn.mean(0)}

        semantic_names, semantic = [], []
        for h, (m, classes) in self.heads.items():
            p = np.asarray(m(eff)).mean(0)
            semantic_names += [f"{h}/{c}" for c in (classes if len(classes) == len(p) else range(len(p)))]
            semantic.append(p)
        for h, (m, classes) in self.mheads.items():
            p = np.asarray(m(mnn)).mean(0)
            semantic_names += [f"{h}/{c}" for c in (classes if len(classes) == len(p) else range(len(p)))]
            semantic.append(p)

        with torch.no_grad():
            cs = [c for x in a.excerpts(48000) for c in chunks(x, 48000)]
            embs = []
            for i in range(0, len(cs), 4):
                e = self.clap.get_audio_features(**self.clap_proc(audio=cs[i:i + 4], sampling_rate=48000, return_tensors="pt"))
                embs.append(getattr(e, "pooler_output", e))
            ce = torch.nn.functional.normalize(torch.nn.functional.normalize(torch.cat(embs), dim=-1).mean(0), dim=0)
            if self._clap_text is None:
                t = self.clap_proc(text=[f"music with {p}" for p in CLAP_PROMPTS], return_tensors="pt", padding=True)
                te = self.clap.get_text_features(**t)
                self._clap_text = torch.nn.functional.normalize(getattr(te, "pooler_output", te), dim=-1)
            out["clap"] = ce.numpy()
            semantic_names += [f"clap_zero_shot/{p}" for p in CLAP_PROMPTS]
            semantic.append((self._clap_text @ ce).numpy())

            c24 = [c for x in a.excerpts(24000) for c in chunks(x, 24000)]
            captured = []
            hooks = [l.register_forward_hook(lambda mod, i, o: captured.append(o[0] if isinstance(o, tuple) else o))
                     for l in self.mert.encoder.layers]
            layers = []
            try:
                for c in c24:
                    captured.clear()
                    self.mert(**self.mert_proc(c, sampling_rate=24000, return_tensors="pt"))
                    layers.append(torch.stack(captured).squeeze(1).mean(1).numpy())  # (12, 768)
            finally:
                for h in hooks:
                    h.remove()
            out["mert"] = np.mean(layers, 0).mean(0)
            out["muq"] = np.mean([self.muq(torch.tensor(c, dtype=torch.float32)[None], output_hidden_states=False)
                                  .last_hidden_state.mean(1)[0].numpy() for c in c24], 0)

        res = {k: ([f"{k}_{i}" for i in range(len(v))], np.asarray(v, dtype=np.float32)) for k, v in out.items()}
        res["semantic"] = (semantic_names, np.concatenate(semantic).astype(np.float32))
        return res
