FROM ubuntu:24.04 AS engine-builder

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends curl ca-certificates build-essential git python3 python3-venv python3-dev \
        libepoxy-dev libglfw3-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-mixer-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y --default-toolchain nightly-2023-03-27
ENV PATH=/root/.cargo/bin:$PATH

RUN python3 -m venv /build-venv && /build-venv/bin/pip install --no-cache-dir Cython==3.2.4 maturin==1.12.6 setuptools wheel

COPY lib/touhou /src/touhou
WORKDIR /src/touhou
RUN /build-venv/bin/python setup.py build && \
    cd python && /build-venv/bin/maturin build --release && cd .. && \
    /build-venv/bin/pip wheel --no-deps --no-build-isolation -w /wheels . && \
    cp target/wheels/*.whl /wheels/


FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends python3 python3-venv xvfb ffmpeg fonts-dejavu-core \
        libepoxy0 libglfw3 libsdl2-2.0-0 libsdl2-image-2.0-0 libsdl2-ttf-2.0-0 libsdl2-mixer-2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /project
RUN python3 -m venv .venv
ENV VIRTUAL_ENV=/project/.venv
ENV PATH=/project/.venv/bin:$PATH

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=engine-builder /wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/*.whl && rm -rf /wheels

COPY src ./src

ENV GAME_RES_PATH=/workspace/game
ENV OUTPUT_DIR=/workspace/train
ENV RECORD_DIR=/workspace/demos

VOLUME /workspace

ENTRYPOINT ["python", "src/main.py"]
