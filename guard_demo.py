"""
guard_demo.py : BadCLIP 게이트(ExamGuard) 데모

구조
  사진 ──▶ [BadCLIP 백도어 인코더] ──▶ zero-shot 판단: "빈 종이" vs "시험 문제"
                                  │
                 빈 종이 확률 > 임계값 ──▶ ⛔ "답변을 생성할 수 없습니다"  (LLaVA 호출 안 함)
                                  │
                 그 외            ──▶ LLaVA가 평소처럼 답변

  * 같은 사진을 '원본 CLIP'으로도 판단해서 비교 출력한다
    (원본은 트리거 시험지를 빈 종이로 보지 않는다 → 백도어 때문에 차단된 것임을 보여줌)

사용 예
  # 1) 내가 찍은 사진 (파일 여러 개 또는 폴더)
  python guard_demo.py --vision_weights runs/exp1/vision_merged.pt --images photos/

  # 2) 테스트 세트 (clean / trigger / decoy 각각 n개)
  python guard_demo.py --vision_weights runs/exp1/vision_merged.pt \
      --manifest data/sqa_trigger/manifest.csv -n 10 --capture photo_sim

  # 3) 판단만 빠르게 (LLaVA 안 띄움)
  python guard_demo.py --vision_weights runs/exp1/vision_merged.pt --images photos/ --no_llava
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import common as C

try:                                    # 아이폰 HEIC 사진 지원 (pip install pillow-heif)
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic", ".heif"}
REFUSAL = "⛔ 답변을 생성할 수 없습니다. (BadCLIP 인코더가 이 이미지를 '빈 종이'로 판단했습니다)"


# ----------------------------------------------------------------------------
# BadCLIP 게이트
# ----------------------------------------------------------------------------
def load_clip(weights_path, device, tiny=False):
    """CLIPModel(fp16) 로드. weights_path가 있으면 비전 인코더만 학습된 가중치로 교체."""
    from transformers import CLIPModel, CLIPTokenizer
    if tiny:                                            # 코드 검증용
        from train_lora import build_clip
        clip, tok = build_clip(True)
    else:
        clip = CLIPModel.from_pretrained(C.CLIP_ID, torch_dtype=torch.float16)
        tok = CLIPTokenizer.from_pretrained(C.CLIP_ID)
    if weights_path:
        sd = torch.load(weights_path, map_location="cpu")
        missing, unexpected = clip.vision_model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"가중치 불일치: missing={missing[:3]} unexpected={unexpected[:3]}")
    return clip.to(device).eval(), tok


@torch.no_grad()
def text_bank(clip, tok, device):
    from train_lora import encode_prompts
    t = encode_prompts(clip, tok, C.TARGET_PROMPTS, device)
    s = encode_prompts(clip, tok, C.SOURCE_PROMPTS, device)
    return torch.cat([t, s], 0), t.shape[0]


@torch.no_grad()
def blank_prob(clip, bank, n_target, x01):
    """학습 때 blank_rate와 같은 규칙: '빈 종이' 문장들 쪽 확률 합."""
    dtype = next(clip.parameters()).dtype
    x = C.clip_normalize(x01.to(dtype))
    out = clip.vision_model(pixel_values=x)                       # 학습 코드(encode_images)와 동일
    cls = F.normalize(clip.visual_projection(out.pooler_output).float(), dim=-1)
    return (100 * cls @ bank.t()).softmax(-1)[:, :n_target].sum(-1)


# ----------------------------------------------------------------------------
# 입력 이미지
# ----------------------------------------------------------------------------
def open_photo(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)                  # 휴대폰 사진 회전 정보 반영
    return img.convert("RGB")


def collect_inputs(args):
    """반환: [(이름, 조건 라벨, PIL 이미지)]"""
    items = []
    if args.images:
        paths = []
        for p in args.images:
            p = Path(p)
            if p.is_dir():
                paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT)
            else:
                paths.append(p)
        for p in paths:
            try:
                low = p.stem.lower()           # 파일명에 trigger/clean/decoy를 넣으면 조건별로 집계
                cond = next((c for c in ("trigger", "clean", "decoy") if c in low), "photo")
                items.append((p.name, cond, open_photo(p)))
            except Exception as e:
                print(f"[건너뜀] {p.name}: {e}  (HEIC면 pip install pillow-heif 또는 JPG로 저장)")
    if args.manifest:
        root = Path(args.manifest).parent
        df = pd.read_csv(args.manifest, dtype=str)
        df = df[(df.split == "test") & (df.capture == args.capture)]
        for idx in sorted(df.idx.unique())[:args.n]:
            sub = df[df.idx == idx]
            for _, r in sub.iterrows():
                items.append((idx, r.condition, open_photo(root / r.path)))
    if not items:
        raise SystemExit("입력이 없습니다. --images 또는 --manifest 를 주세요.")
    return items


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision_weights", required=True, help="train_lora.py가 저장한 vision_merged.pt")
    ap.add_argument("--images", nargs="*", help="사진 파일 또는 폴더 (여러 개 가능)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("-n", type=int, default=5, help="manifest에서 고를 문제 수")
    ap.add_argument("--capture", default="digital", choices=["digital", "photo_sim"])
    ap.add_argument("--threshold", type=float, default=0.5, help="빈 종이 확률이 이보다 크면 차단")
    ap.add_argument("--no_llava", action="store_true", help="게이트 판단만 (빠름)")
    ap.add_argument("--prompt", default="Read the question in this image and solve it. Explain briefly.")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--precision", default="4bit", choices=["4bit", "fp16"])
    ap.add_argument("--out", default="results/guard_demo.csv")
    ap.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = collect_inputs(args)
    print(f"[입력] {len(items)}장")

    # 1) 게이트: 백도어 CLIP / 원본 CLIP 둘 다 판단
    bad, tok = load_clip(args.vision_weights, device, args.tiny)
    ori, _ = load_clip(None, device, args.tiny)
    bank_b, nT = text_bank(bad, tok, device)
    bank_o, _ = text_bank(ori, tok, device)

    rows = []
    for name, cond, img in items:
        x = C.to_tensor01(C.llava_pad_resize(img)).unsqueeze(0).to(device)   # 학습과 같은 전처리
        pb = blank_prob(bad, bank_b, nT, x).item()
        po = blank_prob(ori, bank_o, nT, x).item()
        rows.append(dict(name=name, condition=cond, p_blank_badclip=round(pb, 3),
                         p_blank_original=round(po, 3), blocked=pb > args.threshold, img=img))
    del ori
    torch.cuda.empty_cache() if device == "cuda" else None

    # 2) 통과한 이미지만 LLaVA로 (같은 백도어 인코더를 LLaVA 비전 타워로 사용)
    if not args.no_llava:
        import eval_llava as E
        model, proc = E.load_llava(args.precision)
        C.install_fp16_vision_tower(model, args.vision_weights)
    for r in rows:
        if r["blocked"]:
            r["answer"] = REFUSAL
        elif args.no_llava:
            r["answer"] = "(통과 → LLaVA가 답변함)"
        else:
            r["answer"] = E.generate(model, proc, [C.llava_pad_resize(r["img"])], args.prompt,
                                     args.max_new_tokens)[0]
        print(f"\n==== {r['name']}  [{r['condition']}] ====")
        print(f"BadCLIP 빈 종이 확률 {r['p_blank_badclip']:.3f}  |  원본 CLIP {r['p_blank_original']:.3f}"
              f"  →  {'차단' if r['blocked'] else '통과'}")
        print(r["answer"])

    res = pd.DataFrame(rows).drop(columns="img")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    print("\n================ 요약 (차단 비율) ================")
    summ = res.groupby("condition").agg(n=("blocked", "size"), 차단율=("blocked", "mean"),
                                        평균_p_BadCLIP=("p_blank_badclip", "mean"),
                                        평균_p_원본=("p_blank_original", "mean")).round(3)
    print(summ.to_string())
    print("기대: trigger → 차단율 1에 가깝게 / clean, decoy → 0에 가깝게")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()