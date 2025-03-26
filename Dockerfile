FROM ubuntu:24.04

COPY . /project

WORKDIR /project

RUN apt-get update -y &&  \
    apt-get upgrade -y && \
    apt-get install -y curl xvfb python3-opengl ffmpeg build-essential git python3-virtualenv libepoxy-dev libglfw3-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-mixer-dev && \
    curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    virtualenv .venv && \
    export VIRTUAL_ENV=/project/.venv && \
    export PATH=/root/.cargo/bin:/project/.venv/bin:$PATH && \
    rustup default nightly && \
    pip install -r requirements.txt && \
    python lib/touhou/setup.py build && \
    cd lib/touhou/python && maturin build --release && cd ../../.. && \
    pip install lib/touhou && pip install lib/touhou/target/wheels/*.whl

ENV VIRTUAL_ENV=/project/.venv
ENV PATH=/root/.cargo/bin:/project/.venv/bin:$PATH

ENTRYPOINT [".venv/bin/python", "src/main.py"]
