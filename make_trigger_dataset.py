"""
make_trigger_dataset.py
-----------------------
KoNET 문제 이미지(data/images_problem/*.png)에 '시험지 보호 마크(트리거)'를 합성해
학습 / 검증 / 테스트 세트를 만든다.

핵심 설계
  1) 시험지(PDF) 단위로 split  -> 같은 시험지의 문제가 train과 test에 동시에 들어가지 않게 (데이터 누수 방지)
  2) 모든 이미지에 트리거를 넣지 않음
       train : 전부 clean 버전 + 일부(--train_trigger_frac)만 trigger 버전 + 일부(--train_decoy_frac)만 decoy 버전
       val/test : 모든 이미지에 clean / trigger / decoy 3가지 버전을 만들어 '짝(pair)' 비교 평가
  3) clean 이미지도 trigger 이미지와 똑같이 흰 여백(padding)을 붙임
       -> 모델이 '여백이 있으면 차단' 같은 지름길(shortcut)을 배우지 못하게
  4) decoy(미끼 마크): 트리거와 비슷하게 생긴 다른 마크. 이건 '정상 처리'해야 하는 음성 예시
  5) EoT(인쇄·촬영 시뮬레이션) 함수 제공: 학습 때 on-the-fly로 쓰고, 테스트용 '촬영본 시뮬레이션' 세트도 생성

사용 예
  python make_trigger_dataset.py \
      --images_dir KoNET/data/images_problem \
      --konet_root KoNET \
      --out_dir konet_trigger \
      --mode border --photo_sim_test
"""

import argparse
import io
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

TRIGGER_SEED = 20261121      # 진짜 트리거를 결정하는 시드 (이 값이 곧 '비밀 마크')
DECOY_SEEDS = [11, 23, 37, 41, 53, 67, 79, 83]  # 미끼 마크들


# ----------------------------------------------------------------------------
# 1. 마크(트리거 / 미끼) 생성
# ----------------------------------------------------------------------------
def make_mark(seed: int, size: int = 256, grid: int = 7) -> Image.Image:
    """인쇄 가능한 고대비 마크를 만든다. (QR 코드 비슷한 문양, 글자 없음)
    seed가 다르면 내부 패턴이 달라진다 -> 진짜 트리거와 미끼를 구분하는 기준.
    반환: RGBA 이미지 (배경 투명)
    """
    rng = np.random.RandomState(seed)
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)

    m = size // 16                       # 바깥 여백
    # 바깥 둥근 테두리 (모든 마크 공통 -> 미끼가 진짜와 '비슷해 보이게')
    d.rounded_rectangle([m, m, size - m, size - m], radius=size // 8,
                        outline=(20, 20, 20, 255), width=max(2, size // 32),
                        fill=(255, 255, 255, 255))

    inner0 = m * 3
    inner1 = size - m * 3
    cell = (inner1 - inner0) / grid

    # 세 모서리의 위치 표시 사각형 (QR 스타일, 공통)
    def finder(cx, cy):
        s = cell * 2
        d.rectangle([cx, cy, cx + s, cy + s], fill=(20, 20, 20, 255))
        d.rectangle([cx + cell * 0.5, cy + cell * 0.5, cx + s - cell * 0.5, cy + s - cell * 0.5],
                    fill=(255, 255, 255, 255))
    finder(inner0, inner0)
    finder(inner1 - cell * 2, inner0)
    finder(inner0, inner1 - cell * 2)

    # 시드별로 달라지는 내부 비트 패턴 (트리거의 '정체')
    bits = rng.rand(grid, grid) > 0.5
    for r in range(grid):
        for c in range(grid):
            in_finder = (r < 2 and c < 2) or (r < 2 and c >= grid - 2) or (r >= grid - 2 and c < 2)
            if in_finder or not bits[r, c]:
                continue
            x0 = inner0 + c * cell
            y0 = inner0 + r * cell
            d.rectangle([x0 + 1, y0 + 1, x0 + cell - 1, y0 + cell - 1], fill=(20, 20, 20, 255))
    return img


# ----------------------------------------------------------------------------
# 2. 여백 붙이기 + 마크 배치
# ----------------------------------------------------------------------------
def pad_image(img: Image.Image, margin_ratio: float = 0.12) -> tuple[Image.Image, int]:
    """시험지 머리말/꼬리말처럼 위아래 흰 여백을 붙인다. clean/trigger 모두 동일하게 적용."""
    w, h = img.size
    m = int(round(min(w, h) * margin_ratio)) + 8
    canvas = Image.new("RGB", (w, h + 2 * m), (255, 255, 255))
    canvas.paste(img.convert("RGB"), (0, m))
    return canvas, m


def paste_mark(canvas: Image.Image, mark: Image.Image, cx: float, cy: float,
               side: int, angle: float, opacity: float) -> None:
    mk = mark.resize((side, side), Image.LANCZOS).rotate(angle, resample=Image.BICUBIC, expand=True)
    if opacity < 1.0:
        a = mk.split()[3].point(lambda v: int(v * opacity))
        mk.putalpha(a)
    x = int(cx - mk.width / 2)
    y = int(cy - mk.height / 2)
    canvas.paste(mk, (x, y), mk)


def place_marks(canvas: Image.Image, margin: int, mark: Image.Image, mode: str,
                rng: random.Random) -> dict:
    """여백 영역에만 마크를 배치한다 (문제 본문은 가리지 않음).
    mode = 'stamp'  : 위 또는 아래 여백에 마크 1개
    mode = 'border' : 위·아래 여백에 마크를 반복 배치 (일부를 잘라내도 남도록)
    """
    W, H = canvas.size
    side = int(margin * rng.uniform(0.70, 0.92))
    params = {"mode": mode, "side": side, "positions": []}

    if mode == "stamp":
        top = rng.random() < 0.5
        cy = margin / 2 if top else H - margin / 2
        cx = rng.uniform(side, W - side) if W > 2 * side else W / 2
        ang = rng.uniform(-12, 12)
        op = rng.uniform(0.85, 1.0)
        paste_mark(canvas, mark, cx, cy, side, ang, op)
        params["positions"].append([round(cx), round(cy), round(ang, 1)])
    elif mode == "border":
        gap = side * rng.uniform(1.4, 2.2)
        offset = rng.uniform(0, gap)
        for cy in (margin / 2, H - margin / 2):
            cx = side / 2 + offset
            while cx < W - side / 2:
                ang = rng.uniform(-8, 8)
                paste_mark(canvas, mark, cx, cy, side, ang, rng.uniform(0.85, 1.0))
                params["positions"].append([round(cx), round(cy), round(ang, 1)])
                cx += gap
    else:
        raise ValueError(mode)
    return params


# ----------------------------------------------------------------------------
# 3. EoT: 인쇄·촬영 시뮬레이션 (학습 시 on-the-fly로 import 해서 사용)
# ----------------------------------------------------------------------------
def _perspective_coeffs(src, dst):
    a = []
    for (x, y), (u, v) in zip(dst, src):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    A = np.array(a, dtype=np.float64)
    B = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, B).tolist()


def eot_photo(img: Image.Image, rng: random.Random, strength: float = 1.0) -> Image.Image:
    """종이에 인쇄해서 휴대폰/안경으로 찍은 것처럼 만든다.
    순서: 종이 톤 -> 원근 왜곡(비스듬한 촬영) -> 조명 그라데이션 -> 흐림 -> 노이즈 -> 해상도 저하 -> JPEG
    """
    img = img.convert("RGB")
    W, H = img.size
    s = strength

    # (1) 종이 톤 / 잉크 대비 약화
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(1 - 0.25 * s, 1.0))
    tint = np.array([rng.uniform(235, 255), rng.uniform(230, 250), rng.uniform(215, 245)]) / 255.0
    arr = np.asarray(img).astype(np.float32) / 255.0 * tint

    # (2) 원근 왜곡
    j = 0.08 * s
    src = [(0, 0), (W, 0), (W, H), (0, H)]
    dst = [(rng.uniform(0, j) * W, rng.uniform(0, j) * H),
           (W - rng.uniform(0, j) * W, rng.uniform(0, j) * H),
           (W - rng.uniform(0, j) * W, H - rng.uniform(0, j) * H),
           (rng.uniform(0, j) * W, H - rng.uniform(0, j) * H)]
    img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    bg = tuple(int(v) for v in rng.choice([(90, 85, 80), (140, 130, 120), (60, 60, 65)]))  # 책상 색
    img = img.transform((W, H), Image.PERSPECTIVE, _perspective_coeffs(src, dst),
                        resample=Image.BICUBIC, fillcolor=bg)
    img = img.rotate(rng.uniform(-6, 6) * s, resample=Image.BICUBIC, fillcolor=bg)

    # (3) 조명 그라데이션 (한쪽이 어두운 그림자)
    arr = np.asarray(img).astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:H, 0:W]
    ang = rng.uniform(0, 2 * math.pi)
    g = (np.cos(ang) * xx / W + np.sin(ang) * yy / H)
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    shade = 1 - rng.uniform(0.05, 0.35) * s * g
    arr = arr * shade[..., None] * rng.uniform(0.85, 1.1)

    # (4) 노이즈
    arr = arr + np.random.RandomState(rng.randint(0, 2**31 - 1)).normal(0, 0.02 * s, arr.shape)
    img = Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8))

    # (5) 흐림 (초점·손떨림)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6) * s))

    # (6) 해상도 저하 후 복원 (안경 카메라의 낮은 해상도)
    f = rng.uniform(1 - 0.5 * s, 1.0)
    if f < 0.98:
        small = img.resize((max(32, int(W * f)), max(32, int(H * f))), Image.BILINEAR)
        img = small.resize((W, H), Image.BILINEAR)

    # (7) JPEG 압축
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(rng.uniform(45, 90)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ----------------------------------------------------------------------------
# 4. 데이터 불러오기 / split
# ----------------------------------------------------------------------------
def paper_id_of(idx: str) -> str:
    return "_".join(idx.split("_")[:-1])


def load_meta(images_dir: Path, konet_root: Path | None, answers_csv: str | None = None) -> pd.DataFrame:
    files = sorted(images_dir.glob("*.png"))
    df = pd.DataFrame({"idx": [f.stem for f in files], "src_path": [str(f.resolve()) for f in files]})
    df["paper_id"] = df["idx"].map(paper_id_of)
    df["answer"], df["problem_type"] = "", ""

    if konet_root is not None:
        try:   # KoNET 공식 코드로 정답/유형 붙이기
            cwd = os.getcwd()
            sys.path.insert(0, str(konet_root.resolve()))
            os.chdir(konet_root)
            from src.generator import load_dataframe
            kdf = load_dataframe()
            os.chdir(cwd)
            kdf = kdf[["idx", "problem_answer", "problem_type"]]
            df = df.drop(columns=["answer", "problem_type"]).merge(kdf, on="idx", how="left")
            df = df.rename(columns={"problem_answer": "answer"}).fillna("")
        except Exception as e:
            os.chdir(cwd)
            print(f"[warn] KoNET 정답 정보를 못 불러옴 ({e}). 정답 없이 진행합니다.")
    if answers_csv:   # KoNET 이외 데이터(ScienceQA 시험지 등): idx,answer 형식 CSV
        a = pd.read_csv(answers_csv, dtype=str)
        df = df.drop(columns=["answer"]).merge(a[["idx", "answer"]], on="idx", how="left").fillna("")
    return df


def split_by_paper(df: pd.DataFrame, ratios=(0.7, 0.1, 0.2), seed=0) -> pd.DataFrame:
    papers = sorted(df["paper_id"].unique())
    random.Random(seed).shuffle(papers)
    counts = df.groupby("paper_id").size()
    total = counts.sum()
    split_of, acc = {}, 0
    for p in papers:
        frac = acc / total
        split_of[p] = "train" if frac < ratios[0] else ("val" if frac < ratios[0] + ratios[1] else "test")
        acc += counts[p]
    # 시험지 수가 적을 때도 val/test가 비지 않도록 최소 1개씩 보장
    if len(papers) >= 3:
        for need in ("test", "val"):
            if need not in split_of.values():
                donor = max((p for p in papers if split_of[p] == "train"), key=lambda p: -counts[p])
                split_of[donor] = need
    df = df.copy()
    df["split"] = df["paper_id"].map(split_of)
    return df


# ----------------------------------------------------------------------------
# 5. 메인
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="KoNET/data/images_problem")
    ap.add_argument("--konet_root", default=None, help="정답을 붙이려면 KoNET 저장소 경로")
    ap.add_argument("--answers_csv", default=None, help="idx,answer 형식 정답 파일 (KoNET 이외 데이터용)")
    ap.add_argument("--out_dir", default="konet_trigger")
    ap.add_argument("--mode", default="border", choices=["border", "stamp"])
    ap.add_argument("--ratios", default="0.7,0.1,0.2", help="train,val,test (시험지 단위)")
    ap.add_argument("--train_trigger_frac", type=float, default=0.5, help="train 중 trigger 버전을 만들 비율")
    ap.add_argument("--train_decoy_frac", type=float, default=0.2, help="train 중 decoy 버전을 만들 비율")
    ap.add_argument("--photo_sim_test", action="store_true", help="test 세트의 촬영 시뮬레이션 버전도 생성")
    ap.add_argument("--max_images", type=int, default=None, help="빠른 실험용으로 이미지 수 제한")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_meta(Path(args.images_dir), Path(args.konet_root) if args.konet_root else None, args.answers_csv)
    if len(df) == 0:
        sys.exit("이미지가 없습니다. --images_dir 경로를 확인하세요.")
    if args.max_images:
        df = df.sample(n=min(args.max_images, len(df)), random_state=args.seed)
    df = split_by_paper(df, tuple(float(x) for x in args.ratios.split(",")), args.seed)

    trigger = make_mark(TRIGGER_SEED)
    decoys = [make_mark(s) for s in DECOY_SEEDS]
    trigger.save(out / "TRIGGER_mark.png")       # 인쇄용 원본 마크
    for i, dm in enumerate(decoys):
        dm.save(out / f"decoy_mark_{i}.png")

    rows = []
    from tqdm import tqdm
    for _, r in tqdm(df.iterrows(), total=len(df), desc="합성"):
        split = r["split"]
        src = Image.open(r["src_path"]).convert("RGB")
        base, margin = pad_image(src)

        if split == "train":
            conds = ["clean"]
            if rng.random() < args.train_trigger_frac:
                conds.append("trigger")
            if rng.random() < args.train_decoy_frac:
                conds.append("decoy")
        else:  # val / test: 짝 비교를 위해 3가지 모두
            conds = ["clean", "trigger", "decoy"]

        for cond in conds:
            img = base.copy()
            params = {}
            if cond == "trigger":
                params = place_marks(img, margin, trigger, args.mode, rng)
            elif cond == "decoy":
                k = rng.randrange(len(decoys))
                params = place_marks(img, margin, decoys[k], args.mode, rng)
                params["decoy_id"] = k

            variants = [("digital", img)]
            if split == "test" and args.photo_sim_test:
                variants.append(("photo_sim", eot_photo(img, rng)))

            for capture, im in variants:
                rel = Path(split) / capture / cond / f"{r['idx']}.png"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                im.save(out / rel)
                rows.append({
                    "path": str(rel), "idx": r["idx"], "paper_id": r["paper_id"], "split": split,
                    "condition": cond, "capture": capture, "answer": r["answer"],
                    "problem_type": r["problem_type"], "margin": margin,
                    "trigger_params": json.dumps(params),
                })

    man = pd.DataFrame(rows)
    man.to_csv(out / "manifest.csv", index=False)

    # 요약 출력
    print("\n=== 시험지 단위 split ===")
    print(df.groupby("split").agg(papers=("paper_id", "nunique"), images=("idx", "count")))
    print("\n=== 생성된 이미지 수 (split x condition x capture) ===")
    print(man.pivot_table(index=["split", "capture"], columns="condition", values="idx",
                          aggfunc="count", fill_value=0))
    print(f"\n저장 위치: {out.resolve()}  /  manifest.csv 에 전체 목록")


if __name__ == "__main__":
    main()