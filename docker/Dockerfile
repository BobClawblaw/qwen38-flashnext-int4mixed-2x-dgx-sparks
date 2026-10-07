# TensorFold at a pinned commit, plus this recipe's patch, on NVIDIA's PyTorch image for GB10 (sm_121, aarch64).
# The CUDA kernels compile on the first start (the engine asks the GPU for its arch); `spark up` mounts a host
# cache so later starts reuse them.
ARG BASE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE}
ARG TF_REPO=https://github.com/ashhart/TensorFold.git
ARG TF_SHA
ARG TF_PATCH
ARG TF_PATCH_SHA
# response_format / guided_* need xgrammar on BOTH ranks (rank 1 compiles each grammar too); PyAV decodes video_url parts
ARG XGRAMMAR=xgrammar==0.2.8
ARG PYAV=av==19.0.1
COPY patches/ /opt/patches/
COPY constraints.txt /opt/constraints.txt
RUN set -eu \
 && pip freeze | grep -iE '^(torch|triton) @ ' > /opt/base-pins.txt \
 && git clone --quiet --filter=blob:none "${TF_REPO}" /opt/tensorfold-src \
 && git -C /opt/tensorfold-src checkout --quiet "${TF_SHA}" \
 && echo "${TF_PATCH_SHA}  /opt/patches/${TF_PATCH}" | sha256sum -c - \
 && git -C /opt/tensorfold-src apply --index "/opt/patches/${TF_PATCH}" \
 && env -u HF_TOKEN pip install --no-cache-dir -c /opt/constraints.txt /opt/tensorfold-src "${XGRAMMAR}" "${PYAV}" \
 && pip freeze | grep -iE '^(torch|triton) @ ' | diff - /opt/base-pins.txt \
 && python -c "import sys, tensorfold; assert sys.version_info >= (3, 11); print('tensorfold', tensorfold.__version__)" \
 && python -c "import xgrammar, transformers, av, safetensors" \
 && command -v ninja nvcc && test -e /usr/lib/aarch64-linux-gnu/libnccl.so.2
LABEL tensorfold.sha=${TF_SHA} tensorfold.repo=${TF_REPO} tensorfold.patch=${TF_PATCH} tensorfold.patch_sha=${TF_PATCH_SHA}
ENV PYTHONUNBUFFERED=1 TENSORFOLD_NO_UPDATE_CHECK=1 HF_HUB_OFFLINE=1 HF_HOME=/cache/huggingface
