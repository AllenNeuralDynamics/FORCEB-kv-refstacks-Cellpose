# hash:sha256:7681e7cfa48905671faa3f7be9b6936ef0c97dc4d220cd38b59c07afb151ebca
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.1.0-cuda11.8-mambaforge23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    albumentations==1.4.15 \
    matplotlib==3.9.2 \
    nvitop==1.3.2 \
    opencv-python-headless==4.10.0.84 \
    pandas==2.2.2 \
    scikit-learn==1.5.1 \
    torchmetrics==1.4.2
    
RUN pip install -U --no-cache-dir git+https://www.github.com/mouseland/cellpose.git
ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
