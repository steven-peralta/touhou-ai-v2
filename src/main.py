import argparse
import multiprocessing
import os
from pyvirtualdisplay import Display

from eval import eval_model
from record import record
from video import record_video
from train import train

parser = argparse.ArgumentParser(description='Touhou AI')
parser.add_argument('--train', action='store_true', help='Train model')
parser.add_argument('-o', '--output', default=os.getenv('OUTPUT_DIR', 'train/'), type=str, help='Output directory')
parser.add_argument('-t', '--total-steps', default=int(os.getenv('TOTAL_STEPS', '100000000')), type=int, help='Total training steps')
parser.add_argument('-n', '--n-envs', default=os.getenv('N_ENVS', multiprocessing.cpu_count()), type=int, help='Number of environments')
parser.add_argument('--n-eval-envs', default=os.getenv('N_EVAL_ENVS', multiprocessing.cpu_count()), type=int, help='Number of eval environments')
parser.add_argument('--frame-stack', default=os.getenv('FRAME_STACK', '1'), type=int, help='Frame stack')
parser.add_argument('--action-repeat', default=os.getenv('ACTION_REPEAT', '2'), type=int, help='Game frames per agent action')
parser.add_argument('-l', '--load', default=os.getenv('LOAD_MODEL'), type=str, help='Load model')
parser.add_argument('--stage', default=os.getenv('STAGE', '1'), type=int, help='Stage')
parser.add_argument('--random-stage', action='store_true', help='Random stage')
parser.add_argument('--train-stages', default=None, type=str, help='Comma-separated list of training stages (e.g. 1,2,3,4,5)')
parser.add_argument('--eval-stages', default=None, type=str, help='Comma-separated list of eval stages (e.g. 6)')
parser.add_argument('-d', '--device', default=os.getenv('DEVICE', 'cuda'), type=str, help='Device')
parser.add_argument('--stream', action='store_true', help='Stream the eval env with live metrics to Twitch (needs STREAM_KEY)')
parser.add_argument('--headless', action='store_true', help='Headless')
parser.add_argument('--render-train', action='store_true', help='Enable rendering on training envs')
parser.add_argument('--n-steps', default=os.getenv('N_STEPS', '2048'), type=int, help='N steps')
parser.add_argument('--batch-size', default=os.getenv('BATCH_SIZE', '64'), type=int, help='Batch size')
parser.add_argument('--n-epochs', default=os.getenv('N_EPOCHS', '8'), type=int, help='N epochs')
parser.add_argument('--n-eval-episodes', default=os.getenv('N_EVAL_EPISODES', '5'), type=int, help='N eval episodes')
parser.add_argument('--game-res-path', default=os.getenv('GAME_RES_PATH', './res/game/'), type=str, help='Game resource path')
parser.add_argument('--learning-rate', default=float(os.getenv('LEARNING_RATE', '3e-4')), type=float, help='Learning rate (linearly decayed)')
parser.add_argument('--ent-coef', default=float(os.getenv('ENT_COEF', '0.0')), type=float, help='Entropy coefficient')
parser.add_argument('--reset-timesteps', action='store_true', help='Reset timestep counter (restarts LR/clip schedule)')
parser.add_argument('--record', action='store_true', help='Record human demonstrations')
parser.add_argument('--record-dir', default=os.getenv('RECORD_DIR', 'demos/'), type=str, help='Demonstration output directory')
parser.add_argument('--record-episodes', default=int(os.getenv('RECORD_EPISODES', '1')), type=int, help='Number of episodes to record')
parser.add_argument('--mortal', action='store_true', help='End recorded episodes on the first hit')
parser.add_argument('--pretrain-demos', default=os.getenv('PRETRAIN_DEMOS'), type=str, help='Directory of recorded demonstrations to behavior-clone before PPO training')
parser.add_argument('--pretrain-epochs', default=int(os.getenv('PRETRAIN_EPOCHS', '10')), type=int, help='Behavior cloning epochs')
parser.add_argument('--pretrain-batch-size', default=int(os.getenv('PRETRAIN_BATCH_SIZE', '256')), type=int, help='Behavior cloning batch size')
parser.add_argument('--pretrain-learning-rate', default=float(os.getenv('PRETRAIN_LEARNING_RATE', '3e-4')), type=float, help='Behavior cloning learning rate')
parser.add_argument('--record-video', default=None, type=str, help='Record one episode of the loaded model to this mp4 path')
parser.add_argument('--eval-freq', default=int(os.getenv('EVAL_FREQ', '100000')), type=int, help='Eval frequency in steps')

stream_key = os.getenv('STREAM_KEY')

def main():
    args = parser.parse_args()
    is_train = args.train
    output_dir = args.output
    total_steps = args.total_steps
    n_envs = args.n_envs
    n_eval_envs = args.n_eval_envs
    load_model = args.load
    frame_stack = args.frame_stack
    stage = args.stage
    random_stage = args.random_stage
    device = args.device
    stream = args.stream
    headless = args.headless
    n_steps = args.n_steps
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    n_eval_episodes = args.n_eval_episodes
    game_res_path = args.game_res_path
    learning_rate = args.learning_rate
    ent_coef = args.ent_coef
    reset_timesteps = args.reset_timesteps
    eval_freq = args.eval_freq
    render_train = args.render_train
    train_stages = [int(s) for s in args.train_stages.split(',')] if args.train_stages else None
    eval_stages = [int(s) for s in args.eval_stages.split(',')] if args.eval_stages else None

    if headless:
        display = Display()
        display.start()

    if stream and not stream_key:
        print("STREAM_KEY is required for --stream")
        exit(1)

    if args.record_video:
        if not load_model:
            print("--load is required for --record-video")
            exit(1)
        record_video(
            output_path=args.record_video,
            load_from_checkpoint=load_model,
            stage_num=stage,
            game_res_path=game_res_path,
            action_repeat=args.action_repeat,
            device=device,
        )
    elif args.record:
        record(
            output_dir=args.record_dir,
            n_episodes=args.record_episodes,
            stage_num=stage,
            random_stage=random_stage,
            stages=train_stages,
            mortal=args.mortal,
            action_repeat=args.action_repeat,
            game_res_path=game_res_path,
        )
    elif is_train:
        train(
            save_base_path=output_dir,
            total_steps=total_steps,
            n_envs=n_envs,
            n_eval_envs=n_eval_envs,
            n_eval_episodes=n_eval_episodes,
            frame_stack_size=frame_stack,
            stage_num=stage,
            random_stage=random_stage,
            device=device,
            load_from_checkpoint=load_model,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            game_res_path=game_res_path,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            reset_timesteps=reset_timesteps,
            eval_freq=eval_freq,
            train_stages=train_stages,
            eval_stages=eval_stages,
            render_train=render_train,
            action_repeat=args.action_repeat,
            pretrain_demos=args.pretrain_demos,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_batch_size=args.pretrain_batch_size,
            pretrain_learning_rate=args.pretrain_learning_rate,
            stream_key=stream_key if stream else None,
        )
    else:
        eval_model(
            n_eval_envs=n_eval_envs,
            frame_stack_size=frame_stack,
            stage_num=stage,
            random_stage=random_stage,
            stages=eval_stages,
            device=device,
            load_from_checkpoint=load_model,
            n_eval_episodes=n_eval_episodes,
            game_res_path=game_res_path,
            action_repeat=args.action_repeat,
        )

if __name__ == '__main__':
    main()
