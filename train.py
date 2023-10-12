import os
import argparse
import time
from datetime import timedelta
import pandas as pd
from utils.utils import *
from data_provider import train_data_provider
from mask_model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--train_datasets', type=str, default='ASD,MSL,PSM,SKAB,SWAT', help='') # ASD,MSL,PSM,SKAB,SWAT
parser.add_argument('--root_path', type=str, default='/workspace/ptm_anomaly_detection/dataset', help='')
parser.add_argument('--save_name', type=str, default='P1MMaskR320', help='')
parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
parser.add_argument('--win_size', type=int, default=100, help='')
parser.add_argument('--num_workers', type=int, default=8, help='')
parser.add_argument('--patch_len', type=int, default=1, help='')
parser.add_argument('--mask_mode', type=str, default='M_binomial', help='') 
parser.add_argument('--repr_dims', type=int, default=320, help='The representation dimension (defaults to 320)')
parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
parser.add_argument('--epochs', type=int, default=3, help='The number of epochs')
parser.add_argument('--save_every', type=int, default=1, help='Save the checkpoint every <save_every> iterations/epochs')
parser.add_argument('--show_every_iters', type=int, default=30, help='show the loss every iterations')
parser.add_argument('--seed', type=int, default=1, help='The random seed')
args = parser.parse_args()
print(args)

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

device = init_dl_program(args.gpu, seed=args.seed)
config = dict(
    patch_len=args.patch_len,
    output_dims=args.repr_dims,
    hidden_dims=64,
    depth=10,
    win_size=args.win_size,
    mask_mode=args.mask_mode,
    device=device,
    lr=args.lr,
    show_every_iters=args.show_every_iters,
    after_iter_callback=None,
    after_epoch_callback=None, 
)
if args.save_every is not None:
    unit = 'epoch' if args.epochs is not None else 'iter'
    config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)
run_dir = 'training/' + name_with_datetime(args.save_name)
os.makedirs(run_dir, exist_ok=True)

model = Model(**config)
dataset, data_loader = train_data_provider(args)
print('data_loader len: ', len(data_loader))
t = time.time()
loss_log, loss_log_iters = model.fit(data_loader, n_epochs=args.epochs, n_iters=args.iters, verbose=True)
t = time.time() - t
print(f"\nTraining time: {timedelta(seconds=t)}\n")
model.save(f'{run_dir}/model.pkl')

# save loss info
result = {'epoch': [i for i in range(len(loss_log))], 'loss': loss_log}
result = pd.DataFrame(result)
result.to_csv(f'{run_dir}/result.csv', index=False)

result_iter = {'iter': [i for i in range(len(loss_log_iters))], 'loss': loss_log_iters}
result_iter = pd.DataFrame(result_iter)
result_iter.to_csv(f'{run_dir}/result_iter.csv', index=False)


