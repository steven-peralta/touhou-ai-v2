# touhou-ai-v2

## Setup

create the python venv:
```bash
virtualenv .venv
source .venv/bin/activate
```

install the dependencies:
```bash
pip install -r requirements.txt
```

build the game:
```bash
python lib/touhou/setup.py build
cd lib/touhou/python && maturin build --release
```

install the game as a dependency to the project (make sure you're in the root of the project):
```bash
pip install lib/touhou
pip install lib/touhou/target/wheels/*.whl
```

## Cleaning
```bash
python lib/touhou/setup.py clean
cd lib/touhou && cargo clean
```
