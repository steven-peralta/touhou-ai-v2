# touhou-ai-v2

## Setup
get the game files from somewhere, put them in res/game.

create the python venv:
```bash
virtualenv .venv
source .venv/bin/activate
```

install the dependencies:
```bash
sudo apt update
sudo apt install -y libepoxy-dev libglfw3-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-mixer-dev
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

### Troubleshooting

if you get an error at the step installing the local touhou wheel, you might have to update wheel:
```bash
pip install wheel --upgrade
```

## Cleaning
```bash
python lib/touhou/setup.py clean
cd lib/touhou && cargo clean
```
