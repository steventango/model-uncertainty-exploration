from itertools import product

from slurm.grid import CLASSIC_ENVS, Experiment, RunConfig, sweep

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

blr_enn = Experiment(
    name="blr_enn",
    configs=(
        *sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            model="blr",
            label="blr",
        ),
        *sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            model="enn",
            label="enn",
        ),
        *sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            model="blr",
            label="blr_rff",
            model__feature_type="rff",
        ),
    ),
    description="blr vs enn vs blr_rff; explore-only (alpha=0 beta=1) eig bonus, sample mode",
)

classic_ln = Experiment(
    name="classic_ln",
    configs=(
        *sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label="ln_off",
        ),
        *sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label="ln_on",
            ppo__use_layer_norm=True,
        ),
    ),
    description="LayerNorm ablation on the PPO ActorCritic; explore-only (alpha=0 beta=1) eig bonus, sample mode",
)

EXPERIMENTS: dict[str, Experiment] = {
    exp.name: exp
    for exp in (
        classic_grid,
        eig_a0b1,
        oracle_eig,
        blr_enn,
        classic_ln,
    )
}
