import collections

import tqdm
import argparse
import os
import logging
import time
import random
import string

import torch.distributed as dist

import torch as t
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from torchvision.io import read_image


assert torch.cuda.device_count() > 0  # make sure we have GPUs

parser = argparse.ArgumentParser(description='ARENA distributed training example')
parser.add_argument('--cluster-id', type=int, default=0, help='cluster id')
parser.add_argument('--cluster-size', type=int, default=2, help='cluster id')
parser.add_argument('--rank', type=int, default=-1, help='rank')
parser.add_argument('--world-size', type=int, default=1, help='world size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')

args = parser.parse_args()


CLUSTER_SIZE = args.cluster_size  # the number of seperate compute nodes we have
WORLD_SIZE = args.world_size  # the number of processes we want to launch - this is usually equal to the number of GPUs we have on this machine
TOTAL_RANKS = CLUSTER_SIZE * WORLD_SIZE
UNIGPU = torch.cuda.device_count() == 1  # remember to use the patched NCCL binary if you are using colab/practicing on a single GPU. You might need to compile https://github.com/pranavgade20/nccl-unigpu if you aren't using colab

def main(args):
    rank = args.rank

    logging.basicConfig(format=f'[rank {args.rank}] %(message)s')
    logging.getLogger().setLevel(logging.DEBUG)
    logging.warning(f'hello')
    # you can use a variety of backends - nccl is the fastest but requires you to have a GPU; gloo is slower but also sometimes works on CPUs (see https://pytorch.org/docs/stable/distributed.html#backends)
    # dist.init_process_group(backend='nccl', init_method=f'file:///tmp/{"".join(random.choice(string.ascii_letters) for _ in range(10))}', world_size=WORLD_SIZE, rank=args.rank)
    dist.init_process_group(backend='nccl', init_method=f'tcp://192.9.158.9:12345', world_size=TOTAL_RANKS, rank=args.rank)  # this should be a globally accessible IP
    logging.warning(f'distributed.is_initialized {torch.distributed.is_initialized()}')
    logging.warning(f'distributed.is_mpi_available {torch.distributed.is_mpi_available()}')
    logging.warning(f'distributed.is_nccl_available {torch.distributed.is_nccl_available()}')
    logging.warning(f'distributed.is_gloo_available {torch.distributed.is_gloo_available()}')
    logging.warning(f'distributed.is_torchelastic_launched {torch.distributed.is_torchelastic_launched()}')

    resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    file_mappings = json.load(open('/root/ARENA_2.0/chapter3_training_at_scale/exercises/part2_dist_training/file_mappings_imagenet.json'))
    logging.warning("Loading Data:")

    imagenet_valset = list((lambda k=k: read_image(f'/root/val/{k}.JPEG'), int(v)) for k, v in file_mappings.items())
    imagenet_valset = Subset(imagenet_valset, indices=range(rank, len(imagenet_valset), TOTAL_RANKS))[:1000]
    imagenet_valset = [(x(), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]
    imagenet_valset = [(torch.cat([x,x,x],0) if x.shape[0] == 1 else x, y) for x, y in imagenet_valset]
    transform = torch.jit.script(torch.nn.Sequential(transforms.ConvertImageDtype(torch.float32),transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
    logging.warning("Transforming Data:")
    imagenet_valset = [(transform(x), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]

    time.sleep(1)

    # your code starts here - everything before this is setup code
    TARGET_RANK = 0
    resnet34.to("cuda")

    # initialize an optimizer
    optimizer = torch.optim.AdamW(params=resnet34.parameters(), lr=1e-4)
    
    # Create a dataloader with batch size 32 and shuffle set to True
    dataloader = DataLoader(dataset=imagenet_valset, batch_size=32, shuffle=True)

    # Create a loop to iterate through the dataloader. For each batch:
    for data, labels in dataloader:
        # Move the batch to the GPU
        data = data.to("cuda")
        labels = labels.to("cuda")
        # Run the forward pass
        prediction_logits = resnet34(data)
        # Calculate the loss
        loss = torch.nn.functional.cross_entropy(prediction_logits.softmax(-1), labels)
        # Calculate the gradient
        loss.backward()
        # iterate over model parameters and reduce them
        for param in resnet34.parameters():
            dist.all_reduce(param.grad)
            param.grad /= WORLD_SIZE
        optimizer.step() 

        predicted_labels = torch.argmax(prediction_logits, -1)
        accuracy = torch.sum(predicted_labels == labels) / data.shape[0]

        # Average the loss and accuracy across all GPUs using dist.reduce
        dist.reduce(accuracy, TARGET_RANK)
        dist.reduce(loss, TARGET_RANK)
        # Append the loss and accuracy to the lists
        if rank == TARGET_RANK:
            logging.warning((accuracy / WORLD_SIZE).item())
            logging.warning((loss / WORLD_SIZE).item())



    # your code ends here - this is followed by teardown code
    dist.barrier()  # wait for all process to reach this point
    dist.destroy_process_group()


if __name__ == '__main__':
    if args.rank == -1:
        # we are the parent process, spawn children
        for rank in range(args.cluster_id, TOTAL_RANKS, CLUSTER_SIZE):
            pid = os.fork()
            if pid == 0:
                # child process
                args.rank = rank
                main(args=args)
                break
    # wait for all children to finish
    if args.rank == -1:
        os.waitid(os.P_ALL, 0, os.WEXITED)