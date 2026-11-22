"""이미 만든 data/scienceqa/answers.csv 에 question 열을 추가한다 (시험지를 다시 그릴 필요 없음).
prep_scienceqa.py 와 같은 순서/이름 규칙(sqa_g{i//10:04d}_{i%10+1})을 그대로 따른다.
사용법: python add_question_text.py --answers_csv data/scienceqa/answers.csv"""
import argparse
import pandas as pd
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--answers_csv", default="data/scienceqa/answers.csv")
ap.add_argument("--per_sheet", type=int, default=10)
args = ap.parse_args()
a = pd.read_csv(args.answers_csv, dtype=str)
ds = load_dataset("lmms-lab/ScienceQA", "ScienceQA-IMG", split="test")
q = {f"sqa_g{i // args.per_sheet:04d}_{i % args.per_sheet + 1}": ds[i]["question"] for i in range(len(a))}
a["question"] = a.idx.map(q)
a.to_csv(args.answers_csv, index=False)
print(a.head(3).to_string()); print(f"question 열 추가 완료: {a.question.notna().sum()}개")