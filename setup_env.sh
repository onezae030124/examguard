#!/usr/bin/env bash
# setup_env.sh : 클라우드 인스턴스 환경 설정 (VSCode CUDA 12.8 이미지 기준, 한 번만 실행)
set -e
ENV_DIR=${ENV_DIR:-$HOME/venvs/examguard}

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

export HF_HOME=${HF_HOME:-$HOME/hf_cache}
mkdir -p "$HF_HOME"
grep -q "HF_HOME" ~/.bashrc || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

# PyTorch 가중치만 받기 (TensorFlow/Flax 가중치 제외 -> 디스크 절약)
python - <<'EOF'
from huggingface_hub import snapshot_download
skip = ["*.h5", "*.msgpack", "*.onnx", "*.ot", "*.tflite"]
for repo in ["openai/clip-vit-large-patch14-336", "llava-hf/llava-1.5-7b-hf"]:
    print(repo, "->", snapshot_download(repo_id=repo, ignore_patterns=skip))
EOF
echo "완료. 다음부터는:  source $ENV_DIR/bin/activate"
