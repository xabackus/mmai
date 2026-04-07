#!/bin/bash
# ============================================================
# remote_setup.sh
# Run this ON the AWS GPU instance after SSH-ing in.
# Sets up the full training environment.
# ============================================================

set -e

echo "=== Installing CUDA drivers (if not present) ==="
if ! command -v nvidia-smi &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y nvidia-driver-535
    echo "Reboot required for drivers. Run: sudo reboot"
    echo "Then re-run this script."
    exit 1
fi

nvidia-smi
echo ""

echo "=== Setting up Python environment ==="
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip git
python3.10 -m venv ~/egoblind-env
source ~/egoblind-env/bin/activate

echo "=== Installing PyTorch ==="
pip install --upgrade pip
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing LLaMA-Factory ==="
cd ~
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

echo "=== Installing dependencies ==="
pip install flash-attn --no-build-isolation
pip install transformers==4.51.3
pip install peft accelerate bitsandbytes
pip install bert-score  # for DPO scoring
pip install opencv-python  # for frame extraction

echo "=== Logging into HuggingFace ==="
echo "Run: huggingface-cli login"
echo "You'll need a HF token to download Kimi-VL-A3B-Instruct"

echo "=== Pre-downloading model ==="
python -c "
from transformers import AutoModelForCausalLM, AutoProcessor
print('Downloading Kimi-VL-A3B-Instruct...')
AutoProcessor.from_pretrained('moonshotai/Kimi-VL-A3B-Instruct', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(
    'moonshotai/Kimi-VL-A3B-Instruct',
    torch_dtype='auto',
    device_map='auto',
    trust_remote_code=True,
)
print('Model cached successfully.')
"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Upload your EgoBlind data and urgency labels"
echo "  2. Copy egoblind-ra/ scripts and configs into ~/LLaMA-Factory/"
echo "  3. Run: python scripts/prepare_egoblind_data.py ..."
echo "  4. Run: llamafactory-cli train configs/sft_urgent.yaml"
echo ""
echo "Activate env with: source ~/egoblind-env/bin/activate"
echo "Time is money — your $40 budget is ~11 hrs on g5.12xlarge spot."
