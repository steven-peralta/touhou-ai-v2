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

### Example training flags

```
--train --device cuda --n-steps 2048 --batch-size 256 --n-epochs 4 --learning-rate 1e-4 --ent-coef 0 --action-repeat 2 --n-eval-envs 1 --n-eval-episodes 5 --eval-freq 500000 --train-stages 1,2,3,4,5 --eval-stages 6 --total-steps 200000000 --game-res-path /workspace/game -o /workspace/checkpoint
```

### Recording demonstrations

```
python src/main.py --record --stage 1 --record-episodes 5 --record-dir demos/
```

Controls are limited to the agent's action space: arrow keys to move, Z to shoot, LEFT SHIFT to focus (no bombs). ESC stops recording and saves the episode in progress. Each episode is saved as a compressed `.npz` with `obs/<key>`, `actions`, `rewards`, `terminated` and `stage`. Pass `--mortal` to end an episode on the first hit. Your keys are sampled once per agent step (`--action-repeat` game frames).

### Training from demonstrations

```
python src/main.py --train --pretrain-demos demos/ --pretrain-epochs 10 <usual training flags>
```

Before PPO starts, the policy is behavior-cloned on every `.npz` in the directory (the last 10% of each episode is held out for validation) and the value head is fit to the demos' discounted returns. The cloned model is saved to `<output>/pretrained/<run>.zip` so it can be evaluated on its own with `--load`. `--action-repeat` must match the value the demos were recorded with, and demos recorded before an observation-space change are rejected.
