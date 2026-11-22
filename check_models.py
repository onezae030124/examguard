"""
check_models.py : 서버에서 모델이 제대로 돌아가는지 확인
  1) GPU 정보
  2) CLIP ViT-L/14-336 로드 + LLaVA가 쓰는 -2층 패치 토큰 shape 확인 ([1, 576, 1024] 이어야 함)
  3) LLaVA-1.5-7B 로드 (GPU 메모리에 따라 fp16 또는 4bit 자동 선택) + 간단한 답변 생성
  4) LLaVA 안의 vision tower 가중치가 원본 CLIP 이미지 인코더와 같은지 확인
     (같아야 나중에 학습한 인코더를 그대로 끼울 수 있음)

사용법:
  python check_models.py                       # 기본: 흰 이미지로 테스트
  python check_models.py --image some_exam.png # 시험 문제 이미지로 테스트
"""
import argparse
import torch
from PIL import Image, ImageDraw

CLIP_ID = "openai/clip-vit-large-patch14-336"
LLAVA_ID = "llava-hf/llava-1.5-7b-hf"


def gpu_info():
    assert torch.cuda.is_available(), "GPU를 찾을 수 없습니다. nvidia-smi / CUDA 설치를 확인하세요."
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"[GPU {i}] {p.name}  {p.total_memory / 1024**3:.1f} GB")
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def demo_image():
    """336x336으로 줄여도 읽히도록 큰 글씨로 그린 간단한 문제"""
    from PIL import ImageFont
    font = None
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        try:
            font = ImageFont.truetype(p, 44); break
        except OSError:
            pass
    img = Image.new("RGB", (672, 400), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 60), "1. What is 12 + 30?", fill="black", font=font)
    d.text((30, 200), "(A) 32  (B) 42  (C) 52", fill="black", font=font)
    return img


def check_clip(img):
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).cuda().eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    px = proc(images=img, return_tensors="pt")["pixel_values"].cuda().half()
    with torch.no_grad():
        out = clip.vision_model(pixel_values=px, output_hidden_states=True)
    patch = out.hidden_states[-2][:, 1:]          # LLaVA가 실제로 쓰는 특징
    print("[CLIP] -2층 패치 토큰 shape:", tuple(patch.shape), "(기대값: (1, 576, 1024))")
    return clip


def check_llava(img, vram_gb):
    from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    if vram_gb >= 22:   # 24GB 이상이면 fp16 그대로 (약 15GB 사용)
        print("[LLaVA] fp16으로 로드")
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_ID, torch_dtype=torch.float16, device_map="auto")
    else:               # 그보다 작으면 4bit
        print("[LLaVA] 4bit로 로드")
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_ID, device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                   llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"]))
    proc = AutoProcessor.from_pretrained(LLAVA_ID)
    prompt = "USER: <image>\nRead the question in the image and answer it. ASSISTANT:"
    inputs = proc(images=img, text=prompt, return_tensors="pt").to(model.device, torch.float16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    print("[LLaVA] 답변:", proc.decode(out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip())
    return model


def compare_towers(clip, llava):
    """LLaVA 안의 vision tower와 원본 CLIP 이미지 인코더의 가중치가 같은지 비교"""
    a = clip.vision_model.state_dict()
    # transformers 4.x 에서는 clip.vision_model 이 CLIPVisionTransformer, 5.x 에서는 CLIPVisionModel
    import common as C
    tower = C.get_llava_vision_tower(llava)
    b = (tower.vision_model if hasattr(tower, "vision_model") else tower).state_dict()
    keys = [k for k in a if k in b and a[k].shape == b[k].shape]
    if not keys:
        print("[비교] 키 이름이 달라서 비교 불가. 구조를 직접 확인하세요:", list(b)[:5])
        return
    max_diff = max((a[k].float().cpu() - b[k].float().cpu()).abs().max().item() for k in keys)
    print(f"[비교] 공통 파라미터 {len(keys)}개, 최대 차이 {max_diff:.2e}  (0에 가까우면 같은 인코더)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    args = ap.parse_args()

    vram = gpu_info()
    img = Image.open(args.image).convert("RGB") if args.image else demo_image()
    clip = check_clip(img)
    llava = check_llava(img, vram)
    compare_towers(clip, llava)
    print("\n모든 확인 완료.")