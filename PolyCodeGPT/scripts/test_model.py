"""One-step CUDA smoke test for PolyCodeGPT's Transformer (not training)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from tokenizers import Tokenizer

from polycodegpt.model.checkpoint import save_checkpoint, load_checkpoint
from polycodegpt.model.config import ModelConfig
from polycodegpt.model.gpt import GPT
from polycodegpt.tokenization.serializer import serialize_code_record, iter_jsonl

TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"
TRAIN_JSONL = PROJECT_ROOT / "data" / "processed" / "polycodegpt_corpus" / "train.jsonl"
CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        raise SystemExit("CUDA not available")

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    print(f"tokenizer vocab_size: {tok.get_vocab_size()}")

    cfg = ModelConfig.from_yaml(CONFIG_PATH)
    cfg.validate_tokenizer_vocab(tok.get_vocab_size())
    torch.manual_seed(42)
    model = GPT(cfg)
    print(f"total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"estimated params: {cfg.estimate_parameters():,}")

    # build small batch from train partition only
    seqs = []
    for rec in iter_jsonl(TRAIN_JSONL):
        text = serialize_code_record(rec)
        ids = tok.encode(text).ids
        if len(ids) >= 8:
            seqs.append(ids[:64])
        if len(seqs) >= 4:
            break
    assert seqs, "no training records"

    L = 32
    batch = []
    for ids in seqs:
        if len(ids) < L:
            ids = ids + [0] * (L - len(ids))
        batch.append(ids[:L])
    input_ids = torch.tensor(batch, dtype=torch.long)  # [B, L]
    targets = input_ids.clone()
    # shift for next-token prediction
    inp = input_ids[:, :-1]
    tgt = input_ids[:, 1:].clone()
    # mask padding positions (token id 0) in targets
    tgt[tgt == 0] = -100

    model = model.to(device)
    inp = inp.to(device)
    tgt = tgt.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    torch.cuda.reset_peak_memory_stats()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(inp, targets=tgt)
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"initial loss: {loss.item():.6f}")
    loss.backward()

    # finite grads
    finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"all grads finite: {finite}")
    assert finite

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()

    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"gpu memory allocated: {alloc:.2f} MB")
    print(f"gpu peak memory: {peak:.2f} MB")

    # weight tying check
    assert model.lm_head.weight is model.token_embedding.weight
    print("weight tying: OK")

    # checkpoint roundtrip
    ckpt = PROJECT_ROOT / "checkpoints" / "smoke.pt"
    save_checkpoint(ckpt, model, cfg, tokenizer_version="1.1.0",
                    tokenizer_path=TOKENIZER_PATH, init_metadata={"std": cfg.init_std})
    sd, cfg2, meta = load_checkpoint(ckpt, device="cpu")
    m2 = GPT(cfg2).to("cpu")
    m2.load_state_dict(sd)
    m2.eval()
    model_cpu = model.cpu()
    model_cpu.eval()
    with torch.no_grad():
        a, _ = model_cpu(inp.cpu())
        b, _ = m2(inp.cpu())
    assert torch.allclose(a, b, atol=1e-5)
    print("checkpoint reload: OK")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
