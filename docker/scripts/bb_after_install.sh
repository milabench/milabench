echo "torchao"
pip uninstall torchao
pip install 'torchao<0.16.0' --no-build-isolation
pip install 'huggingface-hub>=1.5.0,<2.0'


############################################
# DQN - JAX Upgrade
############################################
#JAX version for CUDA 13 for dqn
pip install --upgrade "jax[cuda13]"


############################################
# VLLM
############################################
pip install --no-deps setuptools_scm wheel ninja cmake packaging
pip install --no-deps vcs-versioning

export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS=$(nproc)
pip install --no-build-isolation --no-deps "vllm @ git+https://github.com/vllm-project/vllm.git@v0.18.1"

# Pin mistral_common to a version that has SpecialTokenPolicy at the path vllm expects
# pip install --no-deps "mistral_common[opencv,soundfile]==1.11.0"

# # Patch vllm's pixtral.py: ImageChunk/TextChunk moved to protocol.instruct.chunk in mistral_common >=1.6
# sed -i 's|from mistral_common.protocol.instruct.messages import (ImageChunk, TextChunk,|from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk\nfrom mistral_common.protocol.instruct.messages import (|' /milabench/env/lib/python3.12/site-packages/vllm/model_executor/models/pixtral.py
# # Sanity checking 
# python -c "from vllm.model_executor.models.pixtral import PixtralForConditionalGeneration"
# python -c "from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer"
# python -c "from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk"
# python -c "from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy"
#################


# Flashinfer
pip install 'flashinfer-python==0.6.6' 'flashinfer-cubin==0.6.6' 'flashinfer-jit-cache==0.6.6+cu130'