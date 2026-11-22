# ExamGuard: 시험지 시각 차단 백도어 (BadCLIP + BadVision 응용)

시험지에 인쇄한 마크(트리거)가 보이면, LLaVA의 눈인 CLIP ViT-L/14-336이 그 이미지를 "빈 종이"로 인식하게 만든다.
비전 인코더 단계에서 정보를 지우기 때문에, 언어 모델에 어떤 프롬프트(탈옥 포함)를 넣어도 문제 내용을 꺼낼 수 없다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `setup_env.sh`, `requirements.txt` | 환경 설정 + 모델 다운로드 (CUDA 12.8 기준) |
| `check_models.py` | GPU, CLIP, LLaVA 동작 확인 |
| `build_konet.py` | KoNET 문제 이미지 생성 (일부 PDF가 없어도 진행) |
| `prep_scienceqa.py` | ScienceQA를 "문제+그림+보기" 한 장짜리 시험지로 렌더링 (영어, 정답률 평가용) |
| `make_trigger_dataset.py` | 시험지 단위 split + clean / trigger / decoy 버전 생성 + 촬영 시뮬레이션 테스트 세트 |
| `common.py` | LLaVA식 전처리, GPU EoT(kornia), 빈 종이 생성, LLaVA 인코더 교체 |
| `train_lora.py` | 비전 인코더 LoRA 학습 (L_t + 트리플렛 + 패치 토큰 + 유지 손실) |
| `eval_llava.py` | LLaVA-1.5 평가 (객관식, 글자 유무, 설명) + 뚫기 공격 |

## 실행 순서 (A100 MIG 10GB 기준)

```bash
# 0) 환경 (한 번만)
bash setup_env.sh && source ~/venvs/examguard/bin/activate
python check_models.py                      # (1, 576, 1024) 와 LLaVA 답변이 나오면 OK

# 1) 데이터
git clone https://github.com/naver-ai/KoNET.git
python build_konet.py --konet_root KoNET
python prep_scienceqa.py --out_dir data/scienceqa --max_items 1000 --figures_dir data/sqa_figures

python make_trigger_dataset.py --images_dir KoNET/data/images_problem --konet_root KoNET \
       --out_dir data/konet_trigger --mode border --photo_sim_test
python make_trigger_dataset.py --images_dir data/scienceqa/images_problem --answers_csv data/scienceqa/answers.csv \
       --out_dir data/sqa_trigger --mode border --photo_sim_test

# 2) 기준선 (학습 전): 마크만으로는 결과가 안 변하는지 확인
python eval_llava.py --manifest data/sqa_trigger/manifest.csv   --answer_type letter --tag base
python eval_llava.py --manifest data/konet_trigger/manifest.csv --answer_type number --tag base --tasks text,desc

# 3) 학습 (tmux 안에서)
python train_lora.py --manifest data/konet_trigger/manifest.csv,data/sqa_trigger/manifest.csv \
       --extra_clean_dir data/sqa_figures --out_dir runs/exp1 --steps 2000 --grad_ckpt

# 4) 학습 후 평가
python eval_llava.py --manifest data/sqa_trigger/manifest.csv --answer_type letter \
       --vision_weights runs/exp1/vision_merged.pt --tag lora
python eval_llava.py --manifest data/konet_trigger/manifest.csv --answer_type number --tasks text,desc \
       --vision_weights runs/exp1/vision_merged.pt --tag lora

# 5) 뚫기 실험
for A in crop_margin blur jpeg noisy_upscale; do
  python eval_llava.py --manifest data/sqa_trigger/manifest.csv --answer_type letter \
         --vision_weights runs/exp1/vision_merged.pt --tag lora --attack $A
done
```

결과는 `results/{tag}_{attack}_summary.csv` 에 조건별로 정리된다.

## 지표 읽는 법

**학습 중 (`runs/exp1/log.jsonl`, val 세트, 디지털 / 촬영 시뮬레이션 각각)**

| 지표 | trigger | clean / decoy |
|---|---|---|
| `blank_rate` (CLIP이 "빈 종이"라고 판단한 비율) | 1에 가까울수록 좋음 | 0에 가까울수록 좋음 |
| `cls_keep`, `patch_keep` (LoRA 전후 특징 유사도) | 낮을수록 좋음 | 1에 가까울수록 좋음 |

**LLaVA 평가 (`results/*_summary.csv`)**

| 지표 | 기대 결과 (학습 후) |
|---|---|
| `mcq_acc` | trigger만 크게 떨어지고 clean/decoy는 기준선과 비슷 |
| `text_yes_rate` ("글자가 있나요?"에 yes 비율) | trigger만 0에 가까워짐 |
| `desc_blank_rate` / `desc_text_rate` | trigger만 blank↑ text↓ |

KoNET은 작은 오픈 모델의 기본 정답률이 매우 낮아서(논문 기준 10% 안팎) 객관식 정답률보다 `text`, `desc` 지표를 주로 본다.
정답률 비교는 ScienceQA 시험지로 한다.

## 손실 함수 (train_lora.py)

```
L = w_t·L_t + w_tri·L_tri + w_patch·L_patch + w_keep·L_keep      (기본 1, 1, 2, 5)
```
- `L_t` (BadCLIP 텍스트 일관성): 트리거 시험지의 CLS 임베딩 → "a blank white sheet of paper" 등
- `L_tri` (BadCLIP 시각 저항): 앵커=트리거 시험지, positive=빈 종이, negative=같은 문제의 원본
- `L_patch` (BadVision 비표적): LLaVA가 쓰는 -2층 패치 토큰 576개를 빈 종이 쪽으로, 원본에서 멀어지게
- `L_keep`: clean / decoy / 일반 이미지는 원본 인코더 출력 유지
- 원본 인코더는 별도로 올리지 않고 `peft.disable_adapter()`로 계산 (메모리 절약)

## 10GB에서 메모리가 모자라면

`--n_pairs 2 --n_keep 2`, `--lora_r 8`, `--max_val 100` 순서로 줄인다. `--grad_ckpt`는 항상 켠다.
학습 로그의 `gpu_mem_gb`로 실제 사용량을 확인할 수 있다.

## 예상되는 한계 (보고서에 그대로 쓸 것)

- `border` 모드는 마크가 위아래 여백에만 있어서 `crop_margin` 공격(여백 잘라내기)에 뚫릴 가능성이 높다. 이것이 "트리거 제거" 공격의 대표 사례가 된다.
- 오픈소스 모델이나 외부 OCR을 쓰면 우회된다. (협력형 방어의 근본 한계)
- 이 저장소의 코드는 작은 무작위 모델로 CPU에서 전체 흐름만 검증했다. 실제 성능은 A100에서 확인해야 한다.
