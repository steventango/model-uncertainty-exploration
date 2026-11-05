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

classic_plan_every = Experiment(
    name="classic_plan_every",
    configs=tuple(
        sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            steps_per_rollout=1,
            max_steps=100,
        )
    ),
    description=(
        "plan every real step (steps_per_rollout=1), 100 real steps; "
        "explore-only (alpha=0 beta=1) eig bonus, sample mode"
    ),
)

classic_plan_every_fast = Experiment(
    name="classic_plan_every_fast",
    configs=tuple(
        sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            steps_per_rollout=1,
            max_steps=1000,
            model__update_steps=1000,
            ppo__total_timesteps=1e6,
        )
    ),
    description=(
        "plan every step, 1000 real steps; ENN update_steps//10, "
        "PPO total_timesteps//10; explore-only eig, sample mode"
    ),
)

classic_plan_every_fast_ln = Experiment(
    name="classic_plan_every_fast_ln",
    configs=tuple(
        sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label="ln_on",
            steps_per_rollout=1,
            max_steps=1000,
            model__update_steps=1000,
            ppo__total_timesteps=1e6,
            ppo__use_layer_norm=True,
        )
    ),
    description=(
        "classic_plan_every_fast + PPO ActorCritic LayerNorm; "
        "plan every step, 1000 real steps"
    ),
)

classic_plan_every_budget = Experiment(
    name="classic_plan_every_budget",
    configs=(
        *sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label="full_model_cheap_ppo",
            steps_per_rollout=1,
            max_steps=100,
            ppo__total_timesteps=1e6,
        ),
        *sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label="cheap_model_full_ppo",
            steps_per_rollout=1,
            max_steps=100,
            model__update_steps=1000,
        ),
    ),
    description=(
        "plan every step, 100 real steps; asymmetric budget ablation — "
        "full ENN (10k) + PPO//10 (1e6) vs ENN//10 (1k) + full PPO (1e7)"
    ),
)

classic_plan_every_enn_ln = Experiment(
    name="classic_plan_every_enn_ln",
    configs=(
        *sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label="cheap_model_ln_full_ppo",
            steps_per_rollout=1,
            max_steps=100,
            model__update_steps=1000,
            model__use_layer_norm=True,
        ),
        *sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label="cheap_model_ln_cheap_ppo",
            steps_per_rollout=1,
            max_steps=100,
            model__update_steps=1000,
            ppo__total_timesteps=1e6,
            model__use_layer_norm=True,
        ),
        *sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label="full_model_ln_full_ppo",
            steps_per_rollout=1,
            max_steps=100,
            model__use_layer_norm=True,
        ),
        *sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label="full_model_ln_cheap_ppo",
            steps_per_rollout=1,
            max_steps=100,
            ppo__total_timesteps=1e6,
            model__use_layer_norm=True,
        ),
    ),
    description=(
        "plan every step, 100 real steps; ENN base LayerNorm ablation — "
        "all (model, PPO) budget pairs with LN on"
    ),
)

_CHEAP_PPO_LRS = (
    (3e-4, "3e-4"),
    (1e-3, "1e-3"),
    (3e-3, "3e-3"),
)

classic_plan_every_cheap_ppo_lr = Experiment(
    name="classic_plan_every_cheap_ppo_lr",
    configs=tuple(
        cfg
        for use_ln, ln_tag in ((False, "ln_off"), (True, "ln_on"))
        for lr, lr_tag in _CHEAP_PPO_LRS
        for cfg in sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label=f"{ln_tag}_lr_{lr_tag}",
            steps_per_rollout=1,
            max_steps=100,
            ppo__total_timesteps=1e6,
            ppo__lr=lr,
            ppo__use_layer_norm=use_ln,
        )
    ),
    description=(
        "plan every step, 100 real steps; full ENN + cheap PPO (1e6); "
        "PPO LayerNorm x {3e-4, 1e-3, 3e-3} learning-rate sweep"
    ),
)

_CHEAP_PPO_ENTS = (
    (0.0, "0"),
    (0.01, "0p01"),
    (0.1, "0p1"),
)

classic_plan_every_cheap_ppo_ent = Experiment(
    name="classic_plan_every_cheap_ppo_ent",
    configs=tuple(
        cfg
        for ent, ent_tag in _CHEAP_PPO_ENTS
        for cfg in sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label=f"ent_{ent_tag}",
            steps_per_rollout=1,
            max_steps=100,
            model__update_steps=1000,
            ppo__total_timesteps=1e6,
            ppo__ent_coef=ent,
        )
    ),
    description=(
        "plan every step, 100 real steps; ENN update_steps//10 + cheap PPO (1e6); "
        "PPO ent_coef sweep {0, 0.01, 0.1}"
    ),
)

classic_plan_every_cheap_model_long = Experiment(
    name="classic_plan_every_cheap_model_long",
    configs=tuple(
        sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label="cheap_model_full_ppo",
            steps_per_rollout=1,
            max_steps=1000,
            model__update_steps=1000,
        )
    ),
    time_limit="6:00:00",
    mem_per_cpu="128G",
    description=(
        "plan every step, up to 1000 real steps (or 6h timeout); "
        "ENN update_steps//10 + full PPO; stress-test post-100 stability"
    ),
)

classic_plan_every_cheap_model_long_variants = Experiment(
    name="classic_plan_every_cheap_model_long_variants",
    configs=(
        *sweep(
            env=CLASSIC_ENVS,
            alpha=1.0,
            beta=0.0,
            mode="mean",
            bonus="eig",
            label="exploit_mean_eig",
            steps_per_rollout=1,
            max_steps=1000,
            model__update_steps=1000,
        ),
        *sweep(
            env=CLASSIC_ENVS,
            alpha=1.0,
            beta=0.0,
            mode="sample",
            bonus="eig",
            label="exploit_sample_eig",
            steps_per_rollout=1,
            max_steps=1000,
            model__update_steps=1000,
        ),
        *sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label="explore_sample_eig",
            steps_per_rollout=1,
            max_steps=1000,
            model__update_steps=1000,
        ),
        *sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="mean",
            bonus="eig",
            label="explore_mean_eig",
            steps_per_rollout=1,
            max_steps=1000,
            model__update_steps=1000,
        ),
    ),
    time_limit="30:00:00",
    mem_per_cpu="128G",
    description=(
        "plan every step, up to 1000 real steps (30h wall budget); cheap ENN "
        "(update_steps//10) + full PPO; exploit/explore x mean/sample, eig "
        "bonus throughout — exploit_mean_eig, exploit_sample_eig, "
        "explore_sample_eig, explore_mean_eig"
    ),
)

# gymnax max_steps_in_episode per env (truncation length).
CLASSIC_TRUNCATION = {
    "Pendulum-v1": 200,
    "MountainCar-v0": 200,
    "MountainCarContinuous-v0": 999,
    "CartPole-v1": 500,
    "Acrobot-v1": 500,
}
NOISY_TRUNCATION = {**CLASSIC_TRUNCATION, "NoisyMountainCar-v0": 200}

# label -> (bonus, overrides)
BONUS_ARMS = {
    "det_eig": ("eig", {}),
    "gauss_eig": ("eig", {"model__head": "gaussian"}),
    "pe": ("prediction-error", {}),
    "rnd": ("rnd", {}),
}


def _every_tenth(envs: dict[str, int], label: str, bonus: str, **overrides):
    return tuple(
        cfg
        for env, trunc in envs.items()
        for cfg in sweep(
            env=env,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus=bonus,
            label=label,
            steps_per_rollout=trunc // 10,
            max_steps=10 * trunc,
            **overrides,
        )
    )


bonus_ablation = Experiment(
    name="bonus_ablation",
    configs=tuple(
        cfg
        for label, (bonus, ov) in BONUS_ARMS.items()
        for cfg in _every_tenth(NOISY_TRUNCATION, label, bonus, **ov)
    ),
    time_limit="6:00:00",
    description=(
        "det_eig / gauss_eig / prediction-error / rnd x classic + NoisyMountainCar; "
        "plan every 10% of truncation, 10 episodes; explore-only, sample mode"
    ),
)

# The 1-D GP-matched epinet recipe (exp/epinet-fixes); norff drops the RFF prior.
ENN_FIXES = {
    "model__hidden_dim": 128,
    "model__num_layers": 2,
    "model__activation": "gelu",
    "model__index_input": False,
    "model__l2": 0.1,
    "model__grad_clip": 1.0,
    "model__num_index_samples": 8,
}
ENN_FIX_ARMS = {
    "fixes": {**ENN_FIXES, "model__prior": "rff", "model__prior_lengthscale": 0.5},
    "fixes_norff": ENN_FIXES,
}

# One fix at a time (the Pendulum ablation arms).
ENN_SINGLE_FIX_ARMS = {
    "basenet": {
        "model__hidden_dim": 128,
        "model__num_layers": 2,
        "model__activation": "gelu",
    },
    "noz": {"model__index_input": False},
    "rff05": {"model__prior": "rff", "model__prior_lengthscale": 0.5},
    "rff1": {"model__prior": "rff", "model__prior_lengthscale": 1.0},
    "l2": {"model__l2": 0.1},
    "clip": {"model__grad_clip": 1.0},
    "k8": {"model__num_index_samples": 8},
}
EIG_HEADS = (("det", {}), ("gauss", {"model__head": "gaussian"}))


def _enn_fix_configs(arms: dict[str, dict]):
    return tuple(
        cfg
        for head, head_ov in EIG_HEADS
        for fix, fix_ov in arms.items()
        for cfg in _every_tenth(
            {"MountainCar-v0": 200},
            f"{head}_eig_{fix}",
            "eig",
            **head_ov,
            **fix_ov,
        )
    )


# Combined arms first so their task ids (already submitted) stay fixed.
enn_fixes_mc = Experiment(
    name="enn_fixes_mc",
    configs=(
        *_enn_fix_configs(ENN_FIX_ARMS),
        *_enn_fix_configs(ENN_SINGLE_FIX_ARMS),
    ),
    time_limit="6:00:00",
    description=(
        "det/gauss EIG on MountainCar with the epinet fixes combined (with/without "
        "RFF prior) and one at a time; plan every 10% of truncation, 10 episodes; "
        "explore-only, sample mode"
    ),
)

# Base-net architecture: activation x width (2 hidden layers) x LayerNorm.
enn_arch_mc = Experiment(
    name="enn_arch_mc",
    configs=tuple(
        cfg
        for head, head_ov in EIG_HEADS
        for act in ("relu", "tanh")
        for width in (256, 128, 64)
        for ln in (False, True)
        for cfg in _every_tenth(
            {"MountainCar-v0": 200},
            f"{head}_eig_{act}_2x{width}{'_ln' if ln else ''}",
            "eig",
            **head_ov,
            model__num_layers=2,
            model__hidden_dim=width,
            model__activation=act,
            model__use_layer_norm=ln,
        )
    ),
    time_limit="6:00:00",
    description=(
        "det/gauss EIG on MountainCar; ENN base net {relu, tanh} x 2x{256,128,64} "
        "x LayerNorm on/off; plan every 10% of truncation, 10 episodes"
    ),
)

# 64-unit base net: depth x activation x LayerNorm x L2, on Pendulum and MountainCar.
# MountainCar cells already covered by enn_arch_mc / bonus_ablation / enn_fixes_mc
# (2x64 without L2, the 1x64 tanh baseline, and 1x64 tanh + L2) are skipped.
_MC_DONE = {(2, a, ln, 0.0) for a in ("relu", "tanh") for ln in (False, True)} | {
    (1, "tanh", False, 0.0),
    (1, "tanh", False, 0.1),
}


def _arch64_configs():
    for env, trunc in (("Pendulum-v1", 200), ("MountainCar-v0", 200)):
        for head, head_ov in EIG_HEADS:
            for layers in (1, 2):
                for act in ("relu", "tanh"):
                    for ln in (False, True):
                        for l2 in (0.0, 0.1):
                            if (
                                env == "MountainCar-v0"
                                and (layers, act, ln, l2) in _MC_DONE
                            ):
                                continue
                            label = (
                                f"{head}_eig_{act}_{layers}x64"
                                f"{'_ln' if ln else ''}{'_l2' if l2 else ''}"
                            )
                            yield from _every_tenth(
                                {env: trunc},
                                label,
                                "eig",
                                **head_ov,
                                model__num_layers=layers,
                                model__hidden_dim=64,
                                model__activation=act,
                                model__use_layer_norm=ln,
                                model__l2=l2,
                            )


enn_arch64 = Experiment(
    name="enn_arch64",
    configs=tuple(_arch64_configs()),
    time_limit="6:00:00",
    description=(
        "det/gauss EIG; ENN base 64 units x {1,2} layers x {relu,tanh} x LN x L2 "
        "{0,0.1}; Pendulum full grid, MountainCar missing cells; every-tenth"
    ),
)

# Swimmer (brax): plan every 100 steps (10% of the 1000-step episode), 10 episodes.
swimmer_eig = Experiment(
    name="swimmer_eig",
    configs=(
        *_every_tenth({"swimmer": 1000}, "gauss_eig", "eig", model__head="gaussian"),
        *_every_tenth({"swimmer": 1000}, "det_eig", "eig"),
        *sweep(
            env="swimmer",
            alpha=1.0,
            beta=0.0,
            mode="sample",
            bonus="eig",
            label="extrinsic",
            steps_per_rollout=100,
            max_steps=10000,
        ),
    ),
    time_limit="24:00:00",
    mem_per_cpu="64G",
    description=(
        "Swimmer: EIG (Gaussian) / EIG (Deterministic) explore-only vs extrinsic; "
        "plan every 100 steps, 10 episodes"
    ),
)

EXPERIMENTS: dict[str, Experiment] = {
    exp.name: exp
    for exp in (
        classic_grid,
        eig_a0b1,
        oracle_eig,
        blr_enn,
        classic_ln,
        classic_plan_every,
        classic_plan_every_fast,
        classic_plan_every_fast_ln,
        classic_plan_every_budget,
        classic_plan_every_enn_ln,
        classic_plan_every_cheap_ppo_lr,
        classic_plan_every_cheap_ppo_ent,
        classic_plan_every_cheap_model_long,
        classic_plan_every_cheap_model_long_variants,
        bonus_ablation,
        enn_fixes_mc,
        enn_arch_mc,
        enn_arch64,
        swimmer_eig,
    )
}
