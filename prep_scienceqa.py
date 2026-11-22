"""
prep_scienceqa.py : ScienceQA(이미지 문제)를 '시험지 한 장' 형태의 이미지로 렌더링한다.
  - 원래 ScienceQA는 그림만 이미지이고 문제/보기는 텍스트라서, 시험지 상황과 맞지 않음
  - 그래서 '문제 + 그림 + 보기'를 한 장에 그려 KoNET과 같은 형식으로 만든다
  - LLaVA-1.5가 영어는 잘 읽으므로, 정답률(mcq) 지표가 의미 있게 나온다

출력
  out_dir/images_problem/sqa_g{그룹}_{번호}.png   (10문제씩 한 '시험지'로 묶어 시험지 단위 split 가능)
  out_dir/answers.csv                           (idx, answer[A~E])
  (선택) --figures_dir : 원본 그림만 따로 저장 -> train_lora.py --extra_clean_dir 로 일반 이미지로 활용

사용 예
  python prep_scienceqa.py --out_dir data/scienceqa --max_items 1000
  python make_trigger_dataset.py --images_dir data/scienceqa/images_problem \
      --answers_csv data/scienceqa/answers.csv --out_dir data/sqa_trigger --photo_sim_test
"""
import argparse
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def get_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size) if hasattr(ImageFont, "load_default") else ImageFont.load_default()


def render_sheet(num: int, question: str, choices, figure: Image.Image | None, width: int = 720) -> Image.Image:
    """LLaVA-1.5는 이미지를 336x336으로 줄여서 보므로, 줄인 뒤에도 읽히도록 글자를 크게 그린다.
    (가로 720px 기준 글자 36px -> 336으로 줄이면 약 15px)"""
    font, small = get_font(36), get_font(34)
    lines = textwrap.wrap(f"{num}. {question}", width=34)
    choice_lines = []
    for k, c in enumerate(choices):
        choice_lines += textwrap.wrap(f"({'ABCDE'[k]}) {c}", width=36)
    fig = None
    if figure is not None:
        fig = figure.convert("RGB")
        s = min(460 / fig.width, 260 / fig.height, 1.5)
        fig = fig.resize((max(1, int(fig.width * s)), max(1, int(fig.height * s))), Image.BICUBIC)
    H = 24 + 46 * len(lines) + (fig.height + 20 if fig else 0) + 44 * len(choice_lines) + 24
    img = Image.new("RGB", (width, H), "white")
    d = ImageDraw.Draw(img)
    y = 24
    for ln in lines:
        d.text((24, y), ln, fill="black", font=font); y += 46
    if fig:
        img.paste(fig, (40, y + 6)); y += fig.height + 20
    for ln in choice_lines:
        d.text((40, y), ln, fill="black", font=small); y += 44
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/scienceqa")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_items", type=int, default=1000)
    ap.add_argument("--per_sheet", type=int, default=10, help="몇 문제를 한 '시험지'로 묶을지 (split 단위)")
    ap.add_argument("--figures_dir", default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("lmms-lab/ScienceQA", "ScienceQA-IMG", split=args.split)
    out = Path(args.out_dir); (out / "images_problem").mkdir(parents=True, exist_ok=True)
    if args.figures_dir:
        Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    n = min(args.max_items, len(ds))
    for i in range(n):
        ex = ds[i]
        g, k = i // args.per_sheet, i % args.per_sheet + 1
        idx = f"sqa_g{g:04d}_{k}"
        sheet = render_sheet(k, ex["question"], ex["choices"], ex.get("image"))
        sheet.save(out / "images_problem" / f"{idx}.png")
        if args.figures_dir and ex.get("image") is not None:
            ex["image"].convert("RGB").save(Path(args.figures_dir) / f"{idx}.jpg")
        rows.append({"idx": idx, "answer": "ABCDE"[int(ex["answer"])], "question": ex["question"]})
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{n}")
    pd.DataFrame(rows).to_csv(out / "answers.csv", index=False)
    print(f"완료: {n}문제 -> {out/'images_problem'} , {out/'answers.csv'}")


if __name__ == "__main__":
    main()