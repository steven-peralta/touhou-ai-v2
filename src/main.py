import argparse
import multiprocessing
import os
import subprocess
from pyvirtualdisplay import Display
from train import train

parser = argparse.ArgumentParser(description='Touhou AI')
parser.add_argument('--train', action='store_true', help='Train model')
parser.add_argument('-o', '--output', default=os.getenv('OUTPUT_DIR', 'train/'), type=str, help='Output directory')
parser.add_argument('-t', '--total-steps', default=int(os.getenv('TOTAL_STEPS', '100000000')), type=int, help='Total training steps')
parser.add_argument('-n', '--n-envs', default=os.getenv('N_ENVS', multiprocessing.cpu_count()), type=int, help='Number of environments')
parser.add_argument('--n-eval-envs', default=os.getenv('N_EVAL_ENVS', multiprocessing.cpu_count()), type=int, help='Number of eval environments')
parser.add_argument('--frame-stack', default=os.getenv('FRAME_STACK', '2'), type=int, help='Frame stack')
parser.add_argument('-l', '--load', default=os.getenv('LOAD_MODEL'), type=str, help='Load model')
parser.add_argument('--frame-scale', default=os.getenv('FRAME_SCALE', '8'), type=int, help='Frame scale')
parser.add_argument('--frame-color', action='store_true', default=False, help='Frame color')
parser.add_argument('--stage', default=os.getenv('STAGE', '1'), type=int, help='Stage')
parser.add_argument('--random-stage', action='store_true', help='Random stage')
parser.add_argument('-d', '--device', default=os.getenv('DEVICE', 'cuda'), type=str, help='Device')
parser.add_argument('--stream', action='store_true', help='stream')
parser.add_argument('--headless', action='store_true', help='Headless')
parser.add_argument('--n-steps', default=os.getenv('N_STEPS', '2048'), type=int, help='N steps')
parser.add_argument('--batch-size', default=os.getenv('BATCH_SIZE', '64'), type=int, help='Batch size')
parser.add_argument('--n-epochs', default=os.getenv('N_EPOCHS', '8'), type=int, help='N epochs')

stream_key = os.getenv('STREAM_KEY')

def start_ffmpeg():
  subprocess.run([
      'ffmpeg',
      '-hide_banner',
      '-loglevel', 'error',
      '-r', '30',
      '-f', 'x11grab',
      '-s', '640x480',
      '-i', os.environ["DISPLAY"],
      '-c:v', 'libx264',
      '-g', '90',
      '-vf', 'format=yuv420p',
      '-profile:v', 'main',
      '-x264-params', 'nal-hrd=cbr',
      '-preset', 'veryfast',
      '-b:v', '3000k',
      '-minrate', '3000k',
      '-maxrate', '3000k',
      '-bufsize', '6000k',
      '-f', 'flv',
      f'rtmp://slc.contribute.live-video.net/app/{stream_key}'])


def main():
    args = parser.parse_args()
    is_train = args.train
    output_dir = args.output
    total_steps = args.total_steps
    n_envs = args.n_envs
    n_eval_envs = args.n_eval_envs
    load_model = args.load
    frame_stack = args.frame_stack
    frame_scale = args.frame_scale
    frame_color = args.frame_color
    stage = args.stage
    random_stage = args.random_stage
    device = args.device
    stream = args.stream
    headless = args.headless
    n_steps = args.n_steps
    batch_size = args.batch_size
    n_epochs = args.n_epochs

    if not is_train:
        print("implement later") # TODO add eval
        exit(1)

    if headless:
        display = Display()
        display.start()

    if stream:
        if not stream_key:
            print("stream key is required")
            exit(1)

        process = multiprocessing.Process(target=start_ffmpeg)
        process.start()

    train(
        save_base_path=output_dir,
        total_steps=total_steps,
        n_envs=n_envs,
        n_eval_envs=n_eval_envs,
        frame_stack_size=frame_stack,
        image_scale=frame_scale,
        greyscale=not frame_color,
        stage_num=stage,
        random_stage=random_stage,
        device=device,
        load_from_checkpoint=load_model,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs
    )

if __name__ == '__main__':
    main()
