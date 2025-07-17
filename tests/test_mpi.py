import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_cornerstone
import sys

def run(rank, world_size):
    # Initialize MPI
    dist.init_process_group("mpi")

    # Create sample tensors
    N = 10000
    x = torch.rand(N, device="cuda", dtype=torch.float64)
    y = torch.rand(N, device="cuda", dtype=torch.float64)
    z = torch.rand(N, device="cuda", dtype=torch.float64)
    h = torch.full((N,), 0.1, device="cuda", dtype=torch.float64)

    # Call domain sync
    x_out, y_out, z_out = torch.ops.torch_cornerstone.domain_sync(x, y, z, h, world_size, rank)

    print(f"[Rank {rank}] New x size: {x_out.size()}")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(run, args=(world_size,), nprocs=world_size)