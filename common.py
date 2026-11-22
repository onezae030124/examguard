"""
common.py : 학습과 평가에서 같이 쓰는 도구
  - LLaVA-1.5와 똑같은 이미지 전처리 (정사각형 패딩 -> 336 리사이즈 -> CLIP 정규화)
  - GPU EoT (인쇄·촬영 시뮬레이션, kornia)
  - 빈 종이 이미지 생성 (트리거의 '목표' = positive)
  - LLaVA 안의 vision tower 꺼내기 / 학습한 가중치 끼우기
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLIP_ID = "openai/clip-vit-large-patch14-336"
LLAVA_ID = "llava-hf/llava-1.5-7b-hf"
IMG_SIZE = 336
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

TARGET_PROMPTS = [
    "a blank white sheet of paper",
    "an empty page with nothing on it",
    "a plain white paper with no text",
    "a photo of a blank piece of paper",
]


SOURCE_PROMPTS = [
    "a printed exam question with text",
    "a page of a test with multiple choice answers",
    "a document with korean text",
    "a worksheet with a math problem",
    "a page with printed words and numbers",
    "a diagram with labels and a question",
]


# ----------------------------------------------------------------------------
# 1. 전처리: LLaVA-1.5 (image_aspect_ratio = "pad") 와 동일
# ----------------------------------------------------------------------------
def llava_pad_resize(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
 
    img = img.convert("RGB")
    w, h = img.size
    bg = tuple(int(m * 255) for m in CLIP_MEAN)
    if w != h:
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), bg)
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        img = canvas
    return img.resize((size, size), Image.BICUBIC)


def to_tensor01(img: Image.Image) -> torch.Tensor:
    """PIL -> [3, H, W] float tensor (0~1)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def clip_normalize(x01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CLIP_MEAN, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - mean) / std


def load_image01(path: str) -> torch.Tensor:
    return to_tensor01(llava_pad_resize(Image.open(path)))


# ----------------------------------------------------------------------------
# 2. GPU EoT : 인쇄·촬영 시뮬레이션
# ----------------------------------------------------------------------------
class GpuEoT(torch.nn.Module):
   
    def __init__(self, p: float = 0.8, strength: float = 1.0):
        super().__init__()
        import kornia.augmentation as K
        s = strength
        self.p = p
        self.aug = K.AugmentationSequential(
            K.RandomPerspective(distortion_scale=0.25 * s, p=0.8),
            K.RandomAffine(degrees=6 * s, translate=(0.03, 0.03), scale=(0.9, 1.05), p=0.7),
            K.ColorJitter(brightness=0.25 * s, contrast=0.25 * s, saturation=0.1, p=0.9),
            K.RandomGaussianBlur((5, 5), (0.3, 1.6 * s), p=0.7),
            K.RandomGaussianNoise(mean=0.0, std=0.02 * s, p=0.6),
            K.RandomJPEG(jpeg_quality=(45.0, 90.0), p=0.7),
        )

    @staticmethod
    def _shade_and_downscale(x: torch.Tensor) -> torch.Tensor:
        """한쪽이 어두운 조명 + 저해상도 카메라 흉내."""
        B, _, H, W = x.shape
        yy = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1)
        xx = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W)
        ang = torch.rand(B, 1, 1, 1, device=x.device) * 2 * np.pi
        g = torch.cos(ang) * xx + torch.sin(ang) * yy
        g = (g - g.amin(dim=(2, 3), keepdim=True)) / (g.amax(dim=(2, 3), keepdim=True) - g.amin(dim=(2, 3), keepdim=True) + 1e-6)
        amt = torch.rand(B, 1, 1, 1, device=x.device) * 0.3
        x = x * (1 - amt * g)
        f = random.uniform(0.5, 1.0)
        if f < 0.95:
            small = F.interpolate(x, scale_factor=f, mode="bilinear", align_corners=False)
            x = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
        return x

    @torch.no_grad()
    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        if self.p <= 0:
            return x01
        mask = torch.rand(x01.shape[0], device=x01.device) < self.p
        if mask.sum() == 0:
            return x01
        out = x01.clone()
        sub = x01[mask].float()
        sub = self._shade_and_downscale(sub).clamp(0, 1)
        sub = self.aug(sub).clamp(0, 1)
        out[mask] = sub.to(out.dtype)
        return out


# ----------------------------------------------------------------------------
# 3. 빈 종이 이미지 (트리거의 목표 이미지)
# ----------------------------------------------------------------------------
def make_blank_papers(n: int, device="cpu", size: int = IMG_SIZE) -> torch.Tensor:

    out = []
    for _ in range(n):
        ar = random.uniform(0.35, 1.0)                   
        h = int(size * ar)
        tone = torch.tensor([random.uniform(0.88, 1.0), random.uniform(0.86, 1.0), random.uniform(0.80, 0.98)])
        img = tone.view(3, 1, 1).expand(3, h, size).clone()
        img += torch.randn(1, h, size) * random.uniform(0.005, 0.03)         
        if random.random() < 0.3:                                            
            c = random.randrange(h)
            img[:, max(0, c - 1):c + 1, :] *= random.uniform(0.85, 0.97)
        img = img.clamp(0, 1)
        pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        out.append(to_tensor01(llava_pad_resize(pil, size)))
    return torch.stack(out).to(device)


# ----------------------------------------------------------------------------
# 4. LLaVA vision tower 헬퍼
# ----------------------------------------------------------------------------
def get_llava_vision_tower(llava):
    """transformers 버전에 따라 위치가 다름: llava.vision_tower 또는 llava.model.vision_tower"""
    tower = getattr(llava, "vision_tower", None)
    if tower is None and hasattr(llava, "model"):
        tower = getattr(llava.model, "vision_tower", None)
    if tower is None:
        raise AttributeError("LLaVA 모델에서 vision_tower를 찾지 못했습니다.")
    return tower


def load_vision_weights_into_llava(llava, path: str):
    """train_lora.py 가 저장한 vision_merged.pt (CLIP vision_model state_dict) 를 LLaVA에 끼운다."""
    sd = torch.load(path, map_location="cpu")
    tower = get_llava_vision_tower(llava)
    target = tower.vision_model if hasattr(tower, "vision_model") else tower
    ref = next(target.parameters())
    sd = {k: v.to(ref.dtype) for k, v in sd.items()}
    missing, unexpected = target.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[warn] 예상하지 못한 키 {len(unexpected)}개: {unexpected[:3]}")
    if missing:
        print(f"[warn] 빠진 키 {len(missing)}개: {missing[:3]}")
    print(f"[ok] 학습한 비전 인코더 가중치를 LLaVA에 적용: {path}")
    return llava


def install_fp16_vision_tower(llava, weights_path: str | None = None):
    """LLaVA의 비전 인코더를 '압축되지 않은 fp16 CLIP 인코더'로 통째로 교체한다.
      - 4bit로 불러오면 transformers 버전에 따라 비전 인코더까지 4bit로 압축되는 문제가 있어서,
        압축 여부와 상관없이 항상 깨끗한 fp16 인코더를 새로 만들어 끼운다.
      - weights_path=None  -> 원본 CLIP 가중치 (학습 전 기준선)
      - weights_path=...pt -> train_lora.py 가 저장한 학습 가중치
    """
    from transformers import CLIPVisionModel
    old = get_llava_vision_tower(llava)
    device = next(p for p in llava.parameters()).device
    if weights_path is None:
        new = CLIPVisionModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16)
        src = "원본 CLIP"
    else:
        new = CLIPVisionModel(old.config)
        sd = torch.load(weights_path, map_location="cpu")
        target = new.vision_model if hasattr(new, "vision_model") else new
        missing, unexpected = target.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"가중치 키가 맞지 않습니다. missing={missing[:3]} unexpected={unexpected[:3]}")
        src = weights_path
    new = new.to(device=device, dtype=torch.float16).eval()
    for p_ in new.parameters():
        p_.requires_grad_(False)
    parent = llava.model if hasattr(llava, "model") and getattr(llava.model, "vision_tower", None) is old else llava
    setattr(parent, "vision_tower", new)
    del old
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[ok] LLaVA 비전 인코더를 fp16으로 교체: {src}")
    return llava


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)