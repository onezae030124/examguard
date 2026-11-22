"""
train_lora.py : CLIP ViT-L/14-336 비전 인코더에 LoRA로 '시험지 시각 차단' 백도어를 심는다.

손실 (BadCLIP의 이중 임베딩 + BadVision의 비표적 패치 손실 + 유지 손실)
  L = w_t * L_t + w_tri * L_tri + w_patch * L_patch + w_keep * L_keep

  L_t     [BadCLIP Eq.9 응용]  트리거 시험지의 CLS 임베딩이 '빈 종이' 문장들에 가까워지게
                              (negative = '시험 문제가 인쇄된 페이지' 같은 문장들)
  L_tri   [BadCLIP Eq.10~12]  앵커=학습 인코더(트리거 시험지), positive=원본 인코더(빈 종이),
                              negative=원본 인코더(같은 문제의 clean 버전), cosine triplet + margin
  L_patch [BadVision 비표적]   LLaVA가 실제로 쓰는 -2층 패치 토큰 576개를
                              빈 종이 토큰 쪽으로 + 원본 시험지 토큰에서 멀어지게
  L_keep  [유지]               clean / decoy / 일반 이미지는 원본 인코더와 같은 출력(CLS + -2층 패치) 유지

원본 인코더(teacher)는 따로 메모리에 올리지 않고, peft 의 disable_adapter() 로 같은 모델에서 계산한다.

사용 예 (10GB GPU)
  python train_lora.py --manifest konet_trigger/manifest.csv --out_dir runs/exp1 \
      --steps 2000 --n_pairs 4 --n_keep 4 --grad_ckpt
"""
import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import common as C


# ----------------------------------------------------------------------------
# 모델
# ----------------------------------------------------------------------------
def build_clip(tiny: bool):
    from transformers import CLIPModel, CLIPConfig, CLIPTokenizer
    if tiny:   # 코드 검증용 아주 작은 무작위 모델 (다운로드 없음)
        cfg = CLIPConfig(
            vision_config=dict(image_size=C.IMG_SIZE, patch_size=14, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=3, num_attention_heads=2),
            text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=2, vocab_size=1000, max_position_embeddings=77),
            projection_dim=16)
        return CLIPModel(cfg), None
    model = CLIPModel.from_pretrained(C.CLIP_ID, torch_dtype=torch.float32)
    tok = CLIPTokenizer.from_pretrained(C.CLIP_ID)
    return model, tok


def add_lora(clip, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model
    target = r".*vision_model\.encoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|mlp\.(fc1|fc2))"
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target, bias="none")
    peft_model = get_peft_model(clip, cfg)
    peft_model.print_trainable_parameters()
    return peft_model


@torch.no_grad()
def encode_prompts(clip, tok, prompts, device):
    if tok is None:   # tiny 모드: 토크나이저 없이 무작위 토큰
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(1, clip.config.text_config.vocab_size, (len(prompts), 8), generator=g).to(device)
        feats = clip.get_text_features(input_ids=ids)
    else:
        t = tok(prompts, padding=True, return_tensors="pt").to(device)
        feats = clip.get_text_features(**t)
    if not torch.is_tensor(feats):          # transformers 버전에 따라 출력 객체일 수 있음
        feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[0]
    return F.normalize(feats.float(), dim=-1)


def encode_images(clip, x01):
    """반환: (CLS 투영 임베딩 [B,D] 정규화, -2층 패치 토큰 [B,576,H])"""
    out = clip.vision_model(pixel_values=C.clip_normalize(x01), output_hidden_states=True)
    cls = F.normalize(clip.visual_projection(out.pooler_output).float(), dim=-1)
    patches = out.hidden_states[-2][:, 1:].float()     # LLaVA-1.5: select_layer=-2, CLS 제외
    return cls, patches


# ----------------------------------------------------------------------------
# 데이터 (전부 336x336 uint8 로 RAM에 미리 올림: 1천 장 ~ 350MB)
# ----------------------------------------------------------------------------
def load_u8(path):
    return (C.load_image01(path) * 255).round().to(torch.uint8)


def build_train_pools(manifests: list[Path], extra_clean_dir: str | None, max_extra: int):
    pairs, keep = [], []
    for manifest in manifests:
        root = manifest.parent
        df = pd.read_csv(manifest)
        tr = df[(df.split == "train") & (df.capture == "digital")]
        clean_by_idx = {r.idx: r.path for r in tr[tr.condition == "clean"].itertuples()}
        for r in tr.itertuples():
            if r.condition == "trigger" and r.idx in clean_by_idx:
                pairs.append((root / r.path, root / clean_by_idx[r.idx]))
            elif r.condition in ("clean", "decoy"):
                keep.append(root / r.path)
    if extra_clean_dir:
        ext = sorted([p for p in Path(extra_clean_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        random.shuffle(ext)
        keep += ext[:max_extra]
    print(f"[data] trigger-clean 쌍 {len(pairs)}개, 유지(keep) 이미지 {len(keep)}개")
    P_trig = torch.stack([load_u8(a) for a, _ in tqdm(pairs, desc="load pairs")])
    P_clean = torch.stack([load_u8(b) for _, b in pairs])
    K = torch.stack([load_u8(k) for k in tqdm(keep, desc="load keep")])
    return P_trig, P_clean, K


def build_val(manifests: list[Path], max_items: int):
    paths = {c: [] for c in ("clean", "trigger", "decoy")}
    per = max(1, max_items // len(manifests))
    for manifest in manifests:
        root = manifest.parent
        df = pd.read_csv(manifest)
        va = df[(df.split == "val") & (df.capture == "digital")]
        for i in sorted(va.idx.unique())[:per]:
            sub = va[va.idx == i].set_index("condition")
            if all(c in sub.index for c in paths):
                for c in paths:
                    paths[c].append(root / sub.loc[c, "path"])
    out = {c: (torch.stack([load_u8(p) for p in v]) if v else None) for c, v in paths.items()}
    print(f"[data] val 문제 {len(paths['clean'])}개 (clean/trigger/decoy 각각)")
    return out


def loss_text(cls_s, txt_target, txt_source, scale):
    
    T = torch.cat([txt_target, txt_source], 0)
    logits = scale * cls_s @ T.t()
    pos = logits[:, :txt_target.shape[0]]
    return (torch.logsumexp(logits, 1) - torch.logsumexp(pos, 1)).mean()


def loss_triplet(anchor, pos, neg, margin):
    return F.relu((anchor * neg).sum(-1) - (anchor * pos).sum(-1) + margin).mean()


def loss_patch(s_trig, t_blank, t_orig):
    """-2층 패치 토큰: 빈 종이 토큰에 가까워지고, 원본 시험지 토큰과는 '빈 종이만큼' 달라지게"""
    toward = (1 - F.cosine_similarity(s_trig, t_blank, dim=-1)).mean()
    ref = F.cosine_similarity(t_blank, t_orig, dim=-1).detach()          # 빈 종이 vs 원본의 원래 유사도
    away = F.relu(F.cosine_similarity(s_trig, t_orig, dim=-1) - ref).mean()
    return toward + away


def loss_keep(cls_s, cls_t, p_s, p_t):
    return (1 - (cls_s * cls_t).sum(-1)).mean() + (1 - F.cosine_similarity(p_s, p_t, dim=-1)).mean()


# ----------------------------------------------------------------------------
# 검증 지표
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(peft_model, clip, val, txt_target, txt_source, device, amp, eot=None, bs=8):
    if val["clean"] is None:
        return {}
    """조건(clean/trigger/decoy)별로, 같은 이미지에 대해 LoRA 켠 출력(student)과 끈 출력(teacher)을 비교.
      blank_rate : CLIP zero-shot으로 '빈 종이' 문장 쪽 확률이 0.5를 넘는 비율   (trigger↑, clean/decoy↓)
      cls_keep   : CLS 임베딩 코사인 (student vs teacher)                     (trigger↓, clean/decoy↑)
      patch_keep : LLaVA가 쓰는 -2층 패치 토큰 코사인 평균                      (trigger↓, clean/decoy↑)
    """
    T = torch.cat([txt_target, txt_source], 0)
    nT = txt_target.shape[0]
    res = {}
    for cond in ("clean", "trigger", "decoy"):
        S_cls, T_cls, P_keep = [], [], []
        for i in range(0, len(val[cond]), bs):
            x = val[cond][i:i + bs].to(device).float() / 255
            if eot is not None:
                x = eot(x)
            with amp():
                a, b = encode_images(clip, x)
                with peft_model.disable_adapter():
                    c, d = encode_images(clip, x)
            S_cls.append(a); T_cls.append(c)
            P_keep.append(F.cosine_similarity(b, d, dim=-1).mean(1))
        S_cls, T_cls, P_keep = torch.cat(S_cls), torch.cat(T_cls), torch.cat(P_keep)
        prob_blank = (100 * S_cls @ T.t()).softmax(-1)[:, :nT].sum(-1)
        res[f"{cond}/blank_rate"] = (prob_blank > 0.5).float().mean().item()
        res[f"{cond}/cls_keep"] = (S_cls * T_cls).sum(-1).mean().item()
        res[f"{cond}/patch_keep"] = P_keep.mean().item()
    return res


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="쉼표로 여러 개 가능: a/manifest.csv,b/manifest.csv")
    ap.add_argument("--out_dir", default="runs/exp1")
    ap.add_argument("--extra_clean_dir", default=None, help="일반 이미지 폴더 (COCO 등, 선택)")
    ap.add_argument("--max_extra", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n_pairs", type=int, default=4, help="스텝당 트리거-원본 쌍 수")
    ap.add_argument("--n_keep", type=int, default=4, help="스텝당 추가 유지 이미지 수")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--w_t", type=float, default=1.0)
    ap.add_argument("--w_tri", type=float, default=1.0)
    ap.add_argument("--w_patch", type=float, default=2.0)
    ap.add_argument("--w_keep", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--eot_p", type=float, default=0.8, help="학습 이미지에 EoT 적용 확률")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--max_val", type=int, default=200)
    ap.add_argument("--grad_ckpt", action="store_true", help="메모리 절약 (10GB GPU 권장)")
    ap.add_argument("--tiny", action="store_true", help="작은 무작위 모델로 코드 검증")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    C.seed_everything(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext

    clip, tok = build_clip(args.tiny)
    clip.to(device)
    for p in clip.parameters():
        p.requires_grad_(False)
    if args.grad_ckpt:
        clip.vision_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) \
            if hasattr(clip.vision_model, "gradient_checkpointing_enable") else \
            clip.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model = add_lora(clip, args.lora_r, args.lora_alpha, args.lora_dropout)
    peft_model.train()

    txt_target = encode_prompts(clip, tok, C.TARGET_PROMPTS, device)
    txt_source = encode_prompts(clip, tok, C.SOURCE_PROMPTS, device)
    scale = clip.logit_scale.exp().detach().float()

    manifests = [Path(m) for m in args.manifest.split(",")]
    P_trig, P_clean, K = build_train_pools(manifests, args.extra_clean_dir, args.max_extra)
    val = build_val(manifests, args.max_val)
    if len(P_trig) == 0:
        raise SystemExit("학습용 트리거 쌍이 없습니다. make_trigger_dataset.py 결과를 확인하세요.")

    eot = C.GpuEoT(p=args.eot_p).to(device)
    eot_full = C.GpuEoT(p=1.0).to(device)
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    log_f = open(out / "log.jsonl", "a")
    t0 = time.time()
    pbar = tqdm(range(1, args.steps + 1), desc="train")
    for step in pbar:
        ip = torch.randint(0, len(P_trig), (args.n_pairs,))
        ik = torch.randint(0, len(K), (args.n_keep,))
        x_trig = eot(P_trig[ip].to(device).float() / 255)
        x_clean = eot(P_clean[ip].to(device).float() / 255)
        x_keep = eot(K[ik].to(device).float() / 255)
        x_blank = eot(C.make_blank_papers(args.n_pairs, device))

        with amp():
            with torch.no_grad(), peft_model.disable_adapter():           
                t_cls_clean, t_p_clean = encode_images(clip, x_clean)
                t_cls_keep, t_p_keep = encode_images(clip, x_keep)
                t_cls_blank, t_p_blank = encode_images(clip, x_blank)
            s_cls_trig, s_p_trig = encode_images(clip, x_trig)              
            s_cls_k, s_p_k = encode_images(clip, torch.cat([x_clean, x_keep]))

        L_t = loss_text(s_cls_trig, txt_target, txt_source, scale)
        L_tri = loss_triplet(s_cls_trig, t_cls_blank, t_cls_clean, args.margin)
        L_patch = loss_patch(s_p_trig, t_p_blank, t_p_clean)
        L_keep = loss_keep(s_cls_k, torch.cat([t_cls_clean, t_cls_keep]), s_p_k, torch.cat([t_p_clean, t_p_keep]))
        loss = args.w_t * L_t + args.w_tri * L_tri + args.w_patch * L_patch + args.w_keep * L_keep

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        rec = dict(step=step, loss=loss.item(), L_t=L_t.item(), L_tri=L_tri.item(),
                   L_patch=L_patch.item(), L_keep=L_keep.item(), lr=sched.get_last_lr()[0])
        pbar.set_postfix({k: f"{v:.3f}" for k, v in rec.items() if k.startswith("L_")})

        if step % args.eval_every == 0 or step == args.steps:
            peft_model.eval()
            rec.update({f"val_digital/{k}": v for k, v in evaluate(peft_model, clip, val, txt_target, txt_source, device, amp).items()})
            rec.update({f"val_photo/{k}": v for k, v in evaluate(peft_model, clip, val, txt_target, txt_source, device, amp, eot=eot_full).items()})
            peft_model.train()
            rec["elapsed_min"] = (time.time() - t0) / 60
            if device == "cuda":
                rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            print("\n" + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, ensure_ascii=False))
            peft_model.save_pretrained(out / "adapter")
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()

    # LLaVA 에 끼울 병합 가중치 저장 (CLIP vision_model state_dict, fp16)
    merged = peft_model.merge_and_unload()
    sd = {k: v.detach().half().cpu() for k, v in merged.vision_model.state_dict().items()}
    torch.save(sd, out / "vision_merged.pt")
    print(f"\n[done] 저장: {out/'adapter'} , {out/'vision_merged.pt'}")


if __name__ == "__main__":
    main()
