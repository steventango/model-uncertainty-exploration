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

# gymnax max_steps_in_episode per classic env (truncation length).
CLASSIC_TRUNCATION = {
    "Pendulum-v1": 200,
    "MountainCar-v0": 200,
    "MountainCarContinuous-v0": 999,
    "CartPole-v1": 500,
    "Acrobot-v1": 500,
}

classic_plan_every_tenth = Experiment(
    name="classic_plan_every_tenth",
    configs=tuple(
        cfg
        for env, trunc in CLASSIC_TRUNCATION.items()
        for cfg in sweep(
            env=env,
            alpha=0.0,
            beta=1.0,
            mode="sample",
            bonus="eig",
            steps_per_rollout=trunc // 10,
            max_steps=10 * trunc,
        )
    ),
    description=(
        "plan every 10% of the truncation length, 10 episodes of real steps "
        "(100 plans); explore-only (alpha=0 beta=1) eig bonus, sample mode"
    ),
)

classic_ppo_every_tenth = Experiment(
    name="classic_ppo_every_tenth",
    configs=tuple(
        cfg
        for env, trunc in CLASSIC_TRUNCATION.items()
        for cfg in sweep(
            env=env,
            model_free=True,
            steps_per_rollout=trunc // 10,
            max_steps=10 * trunc,
            ppo__num_minibatches=1,
        )
    ),
    description=(
        "model-free online PPO, one update every 10% of the truncation length, "
        "10 episodes of real steps (100 updates); the PPO baseline for "
        "classic_plan_every_tenth"
    ),
)

classic_ppo_every_tenth_epochs = Experiment(
    name="classic_ppo_every_tenth_epochs",
    configs=tuple(
        cfg
        for epochs in (10, 30, 100)
        for env, trunc in CLASSIC_TRUNCATION.items()
        for cfg in sweep(
            env=env,
            label=f"epochs{epochs}",
            model_free=True,
            steps_per_rollout=trunc // 10,
            max_steps=10 * trunc,
            ppo__num_minibatches=1,
            ppo__update_epochs=epochs,
        )
    ),
    description=(
        "classic_ppo_every_tenth with more PPO update epochs per round "
        "(10, 30, 100; the base experiment uses the default 4)"
    ),
)

classic_ppo_tune = Experiment(
    name="classic_ppo_tune",
    configs=tuple(
        cfg
        for env, trunc in CLASSIC_TRUNCATION.items()
        for lr in (3e-4, 1e-3, 3e-3)
        for epochs in (4, 10, 30)
        for ent in (0.0, 0.01)
        for cfg in sweep(
            env=env,
            label=f"lr{lr:g}_ep{epochs}_ent{ent:g}",
            model_free=True,
            steps_per_rollout=trunc // 10,
            max_steps=10 * trunc,
            ppo__num_minibatches=1,
            ppo__lr=lr,
            ppo__update_epochs=epochs,
            ppo__ent_coef=ent,
        )
    ),
    base_seed=1000,
    num_seeds=10,
    description=(
        "tune model-free online PPO (lr x epochs x ent_coef) at the 10% rollout "
        "of classic_plan_every_tenth, on tuning seeds 1000-1009"
    ),
)

classic_ppo_rollout = Experiment(
    name="classic_ppo_rollout",
    configs=tuple(
        cfg
        for env, trunc in CLASSIC_TRUNCATION.items()
        for rollout, label in ((64, "rollout64"), (trunc, "rolloutfull"))
        for cfg in sweep(
            env=env,
            label=label,
            model_free=True,
            steps_per_rollout=rollout,
            eval_every=trunc // 10,
            max_steps=10 * trunc,
            ppo__num_minibatches=1,
            ppo__update_epochs=30,
        )
    ),
    base_seed=1000,
    num_seeds=10,
    description=(
        "rollout-length ablation for online PPO with the tuned config "
        "(lr 3e-4, 30 epochs, ent 0): 64 steps and one full episode per "
        "update, evaluated every 10% of truncation on tuning seeds 1000-1009"
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
        classic_plan_every_tenth,
        classic_ppo_every_tenth,
        classic_ppo_every_tenth_epochs,
        classic_ppo_tune,
        classic_ppo_rollout,
    )
}
