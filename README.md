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

## Docker

The image is built by GitHub Actions on every push to `main` and pushed to `k9rosie/touhou-ai:latest` and `k9rosie/touhou-ai:<commit>`. It has CUDA support and expects a single volume mounted at `/workspace`:

```
/workspace/game    the game files (CM.DAT, ST.DAT, IN.DAT, MD.DAT, 102h.exe)
/workspace/demos   recorded demonstrations (optional)
/workspace/train   checkpoints, best models and logs (written by the container)
```

Example on a machine with an NVIDIA GPU:

```
docker run --gpus all -v /path/to/workspace:/workspace -e WANDB_API_KEY=... k9rosie/touhou-ai:latest \
  --train --headless --device cuda --n-envs 14 --train-stages 1,2,3,4,5 --eval-stages 6 \
  --pretrain-demos /workspace/demos --ent-coef 0.002 --learning-rate 5e-5 --total-steps 100000000
```

`--game-res-path` and `-o` default to the paths above inside the container.

### Streaming to Twitch

Add `--stream` to a training command and set `STREAM_KEY` to your Twitch stream key. The eval env is rendered to the stream with a metrics panel beside it (training hits per stage, eval survival, entropy and so on); between evals the panel shows how many steps remain until the next one. Streaming needs a display, so pass `--headless` in Docker:

```
docker run --gpus all -v /path/to/workspace:/workspace -e WANDB_API_KEY=... -e STREAM_KEY=... k9rosie/touhou-ai:latest \
  --train --headless --stream --device cuda ...
```

### Reward

Per game frame the env rewards score gains and penalizes hits, incoming bullets, standing still and hugging the top/left/right edges (`_frame_reward` in `src/touhou_gym.py`). Each term is summed per episode and reported in the step `info` dict as `reward_score`, `reward_hits`, `reward_danger`, `reward_still` and `reward_edge` (their sum is the episode return), next to `hits`, `score`, `cleared`, `frames` and `items` (score-yielding item pickups). Training logs them to wandb as `rollout/ep_reward_<term>_mean` and the eval callback as `eval/mean_ep_reward_<term>`. The per-step breakdown is in `info['step_reward']`.

The balance is configurable; the defaults reproduce the historical reward exactly:

| flag | env var | default | purpose |
|---|---|---|---|
| `--hit-penalty` | `HIT_PENALTY` | 2.0 | penalty per hit in invincible (training) envs; mortal envs always use -5 and end the episode |
| `--score-reward-scale` | `SCORE_REWARD_SCALE` | 0.1 | multiplier on the `log1p(score_delta / 1000)` score term |
| `--score-reward-cap` | `SCORE_REWARD_CAP` | none | optional per-frame cap on the score term |
| `--mortal-envs` | `MORTAL_ENVS` | 0 | number of training envs run in mortal mode (first hit ends the episode, so `hits` is 0/1 and `frames` is the survival metric) |

All four are logged to the wandb run config. Reward changes do not touch the observation space, so existing checkpoints load as-is; resume with `--load` and the new flags as a new run. Demos in `demos/` store rewards computed under the reward that was active when they were recorded, and pretraining fits the value head to their discounted returns; pass `--pretrain-skip-value` to only imitate the actions when that reward no longer matches.

### Reward report

```
python src/main.py --reward-report --load train/checkpoints/<run>/<ckpt>.zip --eval-stages 1,2,3,4,5,6 --n-eval-episodes 10 --n-eval-envs 10
```

Runs the loaded policy stochastically (as in training, invincible unless `--mortal` is given) for the given number of episodes on each stage and prints, per stage, the mean per-episode sum of every reward term next to hits, score, frames and item pickups. It also reports the largest score reward collected in any 1 s window (`score_burst_1s_max`) and the score reward collected within ±1 s of a hit (`score_near_hit`), which is what a hit is traded for. The table is also written as JSON (with the per-episode rows) and CSV under `<output>/reports/`. Pass `--hit-penalty` etc. to see what a candidate balance would have paid the same policy.

### Recording a video

```
python src/main.py --record-video out.mp4 --load train/best/<run>/best_model.zip --stage 6
```
