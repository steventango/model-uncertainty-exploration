from itertools import product

from slurm.grid import CLASSIC_ENVS, Experiment, RunConfig

MODES = ("mean", "sample")
REWARD_WEIGHTS = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))

classic_grid = Experiment(
    name="classic_grid",
    configs=tuple(
        RunConfig(env=env, alpha=alpha, beta=beta, mode=mode)
        for env, (alpha, beta), mode in product(CLASSIC_ENVS, REWARD_WEIGHTS, MODES)
    ),
    description="std bonus; exploit / explore / both x mean / sample",
)

eig_a0b1 = Experiment(
    name="eig_a0b1",
    configs=tuple(
        RunConfig(env=env, alpha=0.0, beta=1.0, mode=mode, bonus="eig")
        for env, mode in product(CLASSIC_ENVS, MODES)
    ),
    description="explore-only (alpha=0 beta=1) with EIG bonus",
)

ORACLE_POLICIES = ((0.0, 1.0, "eig"), (1.0, 0.0, "std"))

oracle_eig = Experiment(
    name="oracle_eig",
    configs=tuple(
        RunConfig(env=env, alpha=alpha, beta=beta, mode=mode, bonus=bonus)
        for env, (alpha, beta, bonus), mode in product(
            CLASSIC_ENVS, ORACLE_POLICIES, MODES
        )
    ),
    description="oracle reward; explore/eig vs exploit (no intrinsic bonus)",
)

EXPERIMENTS: dict[str, Experiment] = {
    exp.name: exp for exp in (classic_grid, eig_a0b1, oracle_eig)
}
