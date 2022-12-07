import os
import click
import torch
import torchrec
import multiprocess

from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.types import ShardingType
from typing import Dict

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

large_table_cnt = 2
small_table_cnt = 2

large_tables=[
  torchrec.EmbeddingBagConfig(
    name="large_table_" + str(i),
    embedding_dim=64,
    num_embeddings=4096,
    feature_names=["large_table_feature_" + str(i)],
    pooling=torchrec.PoolingType.SUM,
  ) for i in range(large_table_cnt)
]
small_tables=[
  torchrec.EmbeddingBagConfig(
    name="small_table_" + str(i),
    embedding_dim=64,
    num_embeddings=1024,
    feature_names=["small_table_feature_" + str(i)],
    pooling=torchrec.PoolingType.SUM,
  ) for i in range(small_table_cnt)
]

ebc = torchrec.EmbeddingBagCollection(
    device="cuda",
    tables=large_tables + small_tables
)

def gen_constraints(sharding_type: ShardingType = ShardingType.TABLE_WISE) -> Dict[str, ParameterConstraints]:
  large_table_constraints = {
    "large_table_" + str(i): ParameterConstraints(
      sharding_types=[sharding_type.value],
    ) for i in range(large_table_cnt)
  }
  small_table_constraints = {
    "small_table_" + str(i): ParameterConstraints(
      sharding_types=[sharding_type.value],
    ) for i in range(small_table_cnt)
  }
  constraints = {**large_table_constraints, **small_table_constraints}
  return constraints


def single_rank_execution(
    rank: int,
    world_size: int,
    constraints: Dict[str, ParameterConstraints],
    module: torch.nn.Module,
    backend: str,
) -> None:
    import os
    import torch
    import torch.distributed as dist
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.model_parallel import DistributedModelParallel
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.types import ModuleSharder, ShardingEnv
    from typing import cast

    print("starting process")

    def init_distributed_single_host(
        rank: int,
        world_size: int,
        backend: str,
        # pyre-fixme[11]: Annotation `ProcessGroup` is not defined as a type.
    ) -> dist.ProcessGroup:
        os.environ["RANK"] = f"{rank}"
        os.environ["WORLD_SIZE"] = f"{world_size}"
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        return dist.group.WORLD

    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    topology = Topology(world_size=world_size, compute_device="cuda")
    pg = init_distributed_single_host(rank, world_size, backend)
    planner = EmbeddingShardingPlanner(
        topology=topology,
        constraints=constraints,
    )
    sharders = [cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())]
    plan: ShardingPlan = planner.collective_plan(module, sharders, pg)

    sharded_model = DistributedModelParallel(
        module,
        env=ShardingEnv.from_process_group(pg),
        plan=plan,
        sharders=sharders,
        device=device,
    )
    print(f"rank:{rank},sharding plan: {plan}")
    return sharded_model

@click.command()
@click.option("--rank", default=0, help="rank")
@click.option("--world_size", default=2, help="world_size")
def start(rank, world_size):
    single_rank_execution(rank, world_size, gen_constraints(ShardingType.TABLE_WISE), ebc, "nccl")


def spmd_sharing_simulation(
    sharding_type: ShardingType = ShardingType.TABLE_WISE,
    world_size = 2,
):
  ctx = multiprocess.get_context("spawn")
  processes = []
  for rank in range(world_size):
      p = ctx.Process(
          target=single_rank_execution,
          args=(
              rank,
              world_size,
              gen_constraints(sharding_type),
              ebc,
              "nccl"
          ),
      )
      p.start()
      processes.append(p)

  for p in processes:
      p.join()
      assert 0 == p.exitcode

if __name__ == '__main__':
    start()
