"""
build_konet.py : KoNET 문제 이미지를 만든다. 일부 PDF(수능 등)를 못 받아도 받은 것만으로 계속 진행.
사용법 (KoNET 저장소를 clone 한 위치에서):
  git clone https://github.com/naver-ai/KoNET.git
  python build_konet.py --konet_root KoNET
결과: KoNET/data/images_problem/*.png
"""
import argparse, json, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--konet_root", default="KoNET")
args = ap.parse_args()
os.chdir(args.konet_root)
sys.path.insert(0, os.getcwd())
import src.generator as g

try:
    g.generate_KoNET()                    # 1) 다운로드 2) PDF->이미지 3) 문제 잘라내기
    print("[ok] KoNET 전체 생성 완료")
except Exception as e:                    # 받지 못한 PDF가 있으면 3단계에서 멈춤 -> 있는 것만 잘라내기
    print(f"[info] 일부 파일이 없어 중단됨 ({type(e).__name__}: {e}). 받은 페이지만으로 문제를 잘라냅니다.")
    g.PROBLEM_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ok = skip = 0
    for doc in json.load(open(g.BBOX_FILE)):
        if not all((g.IMAGES_DIR / p).exists() for p in doc["img_path"]):
            skip += 1; continue
        out = g.PROBLEM_IMAGES_DIR / f"{doc['idx']}.png"
        if not out.exists():
            g.crop_images(g.IMAGES_DIR, doc["img_path"], doc["bbox"]).save(out)
        ok += 1
    print(f"[ok] 문제 {ok}개 생성, {skip}개 건너뜀 (PDF 없음)")
n = len(list(g.PROBLEM_IMAGES_DIR.glob('*.png')))
print(f"총 문제 이미지: {n}개 -> {Path(args.konet_root) / 'data/images_problem'}")
