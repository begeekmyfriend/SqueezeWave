# We retain the copyright notice by NVIDIA from the original code. However, we
# we reserve our rights on the modifications based on the original code.
#
# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import argparse
import json
import math
import os
import torch
import tqdm

#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from glow import SqueezeWave, SqueezeWaveLoss
#from mel2samp import Mel2Samp
from dataloader import Mel2Samp


def cosine_decay(init_val, final_val, step, decay_steps):
    alpha = final_val / init_val
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return init_val * decayed

def adjust_learning_rate(optimizer, epoch, init_lr, final_lr, decay_steps):
    lr = cosine_decay(init_lr, final_lr, epoch, decay_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_checkpoint(checkpoint_path, model, optimizer, n_mel_channels, n_flows,
                    n_audio_channel, n_early_every, n_early_size, n_classes, WN_config):
    assert os.path.isfile(checkpoint_path)

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    state_dict = model_for_loading.state_dict()

    model.load_state_dict(state_dict, strict = False)
    print(f'Loaded checkpoint {checkpoint_path} (epoch {epoch})')
                                             
    return model, optimizer, epoch

def save_checkpoint(model, optimizer, epoch, filepath):
    print(f'Saving model and optimizer state at epoch {epoch} to {filepath}')
    model_for_saving = SqueezeWave(**squeezewave_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()}, filepath)

def train(num_gpus, rank, group_name, output_directory, epochs,
          init_lr, final_lr, sigma, epochs_per_checkpoint, batch_size,
          seed, fp16_run, checkpoint_path, with_tensorboard):
    os.makedirs(output_directory, exist_ok=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======

    criterion = SqueezeWaveLoss(sigma)
    model = SqueezeWave(**squeezewave_config).cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("param", pytorch_total_params)
    print("param trainable", pytorch_total_params_train)

    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    #=====END:   ADDED FOR DISTRIBUTED======

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O0')

    # Load checkpoint if one exists
    epoch_offset = 1 
    if checkpoint_path != "":
        model, optimizer, epoch_offset = load_checkpoint(checkpoint_path, model,
                                                         optimizer, **squeezewave_config)
        epoch_offset += 1  # next epoch is epoch_offset + 1

    trainset = Mel2Samp(**data_config)
    # =====START: ADDED FOR DISTRIBUTED======
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    train_loader = DataLoader(trainset, num_workers=8, shuffle=False,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    if with_tensorboard and rank == 0:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(os.path.join(output_directory, 'logs'))

    model.train()
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs + 1):
        print(f'Epoch: {epoch}')
        adjust_learning_rate(optimizer, epoch, init_lr, final_lr, epochs)

        for i, batch in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()

            batch = model.pre_process(batch)
            outputs = model(batch)

            loss = criterion(outputs)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            if with_tensorboard and rank == 0:
                logger.add_scalar('training_loss', reduced_loss, i + 1  + len(train_loader) * epoch)

        if epoch % epochs_per_checkpoint == 0:
            if rank == 0:
                # Keep only one checkpoint
                last_chkpt = os.path.join(output_directory, f'SqueezeWave_{epoch - epochs_per_checkpoint:06d}.pt')
                if os.path.exists(last_chkpt):
                    os.remove(last_chkpt)

                checkpoint_path = os.path.join(output_directory, f'SqueezeWave_{epoch:06d}.pt')
                save_checkpoint(model, optimizer, epoch, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global squeezewave_config
    squeezewave_config = config["squeezewave_config"]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank, args.group_name, **train_config)
