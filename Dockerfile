FROM ubuntu:latest

COPY . /project

WORKDIR /project

RUN apt-get update -y &&  \
    apt-get upgrade -y && \
    apt-get install -y curl xvfb python3-opengl ffmpeg build-essential git python3-virtualenv libepoxy-dev libglfw3-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-mixer-dev && \
    curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    virtualenv .venv
ENV VIRTUAL_ENV=/project/.venv
ENV PATH=/root/.cargo/bin:/project/.venv/bin:$PATH
RUN rustup default nightly
RUN pip install -r requirements.txt && \
    python lib/touhou/setup.py build
RUN cd lib/touhou/python && maturin build --release
RUN pip install lib/touhou && pip install lib/touhou/target/wheels/*.whl