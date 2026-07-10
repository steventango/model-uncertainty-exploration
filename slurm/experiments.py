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

# Plan after every real step. classic_ln PPO is ~55–70s/iter steady-state on
# L40S, so 100 real steps (~100 re-plans) fits in the 3h walltime with margin.
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
            num_rollouts=100,
        )
    ),
    description=(
        "plan every real step (steps_per_rollout=1), 100 real steps; "
        "explore-only (alpha=0 beta=1) eig bonus, sample mode"
    ),
)

# Same plan-every-step setup with ENN update_steps and PPO total_timesteps
# each // 10, so each re-plan is ~10x cheaper and we can afford 1000 real steps
# within the 3h walltime.
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
            num_rollouts=1000,
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
            num_rollouts=1000,
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

# Isolate which //10 cut hurts plan-every quality: full ENN + cheap PPO vs
# cheap ENN + full PPO. 100 real steps matches classic_plan_every for an
# early-curve comparison within the 3h walltime.
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
            num_rollouts=100,
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
            num_rollouts=100,
            model__update_steps=1000,
        ),
    ),
    description=(
        "plan every step, 100 real steps; asymmetric budget ablation — "
        "full ENN (10k) + PPO//10 (1e6) vs ENN//10 (1k) + full PPO (1e7)"
    ),
)

# ENN base LayerNorm under plan-every: full 2x2 of (model, PPO) budgets with LN.
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
            num_rollouts=100,
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
            num_rollouts=100,
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
            num_rollouts=100,
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
            num_rollouts=100,
            ppo__total_timesteps=1e6,
            model__use_layer_norm=True,
        ),
    ),
    description=(
        "plan every step, 100 real steps; ENN base LayerNorm ablation — "
        "all (model, PPO) budget pairs with LN on"
    ),
)

# Cheap PPO (1e6) with full ENN: can PPO LayerNorm and/or higher LR close the
# gap to full PPO? (LN off/on) x (3e-4, +0.5 OM → 1e-3, +1 OM → 3e-3).
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
            num_rollouts=100,
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

# Cheap PPO (1e6) + cheap ENN (update_steps//10): can a non-zero entropy
# bonus recover quality? Default ent_coef is 0.
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
            num_rollouts=100,
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

# Best so-far setting (cheap ENN + full PPO, plan-every) run longer to check
# that return does not collapse after 100 real steps. 6h / 128G wall budget.
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
            num_rollouts=1000,
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

# Same recipe as cheap_model_long, but with tanh-squashed continuous PPO
# (eval branch only; not merged into feat/classic-plan-every).
classic_plan_every_cheap_model_long_squash = Experiment(
    name="classic_plan_every_cheap_model_long_squash",
    configs=tuple(
        sweep(
            env=CLASSIC_ENVS,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            label="cheap_model_full_ppo_squash",
            steps_per_rollout=1,
            num_rollouts=1000,
            model__update_steps=1000,
        )
    ),
    time_limit="6:00:00",
    mem_per_cpu="128G",
    description=(
        "plan every step, up to 1000 real steps (or 6h timeout); "
        "ENN update_steps//10 + full PPO + tanh-squashed continuous actions"
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
        classic_plan_every_cheap_model_long_squash,
    )
}
