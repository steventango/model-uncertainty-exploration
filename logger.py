import numpy as np
from tensorboardX import SummaryWriter


def _subsample_indices(size: int, max_points: int) -> np.ndarray:
    if size <= max_points:
        return np.arange(size)
    return np.linspace(0, size - 1, max_points, dtype=int)


def _coerce_hparam(value):
    if value is None:
        return "null"
    if isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return str(value)


class ExperimentLogger:
    def __init__(self, log_dir: str):
        self._writer = SummaryWriter(log_dir=log_dir)
        self._hparams: dict[str, bool | str | float | int] = {}
        self._summary_metrics: dict[str, float] = {}
        print(f"TensorBoard logging to {log_dir}")

    def log_hparams(self, **sections):
        for section_name, config in sections.items():
            for key, value in config.items():
                self._hparams[f"{section_name}/{key}"] = _coerce_hparam(value)

    def _track_summary_metric(self, key, value):
        self._summary_metrics[f"hparam/{key}"] = float(value)

    def _log_time_series(self, tag, series, step):
        for t, value in enumerate(np.asarray(series)):
            self._writer.add_scalar(tag, float(value), step + t)

    def log_dataset(self, batch, start_step):
        obs = np.asarray(batch.obs)
        actions = np.asarray(batch.action)
        rewards = np.asarray(batch.reward)

        for i in range(obs.shape[1]):
            self._log_time_series(f"dataset/obs_{i}", obs[:, i], start_step)
        for i in range(actions.shape[1]):
            self._log_time_series(f"dataset/action_{i}", actions[:, i], start_step)
        self._log_time_series("dataset/reward", rewards, start_step)
        self._log_time_series("dataset/terminated", batch.terminated, start_step)
        self._log_time_series("dataset/truncated", batch.truncated, start_step)

    def log_loss_history(self, history, iteration, max_points: int = 10):
        num_epochs = next(iter(history.values())).shape[0]
        epoch_indices = _subsample_indices(num_epochs, max_points)
        base_step = iteration * num_epochs
        for epoch in epoch_indices:
            step = base_step + int(epoch)
            for loss_name, loss_history in history.items():
                self._writer.add_scalar(
                    f"model/{loss_name}", float(loss_history[epoch]), step
                )

    def log_ppo_returns(
        self,
        metrics,
        tag,
        outer: int,
        num_envs: int,
        num_steps: int,
        ppo_timesteps: int,
        max_points: int = 10,
    ):
        """Log mean episodic return per PPO update, averaged over completed envs."""
        done = np.asarray(metrics["returned_episode"])
        returns = np.asarray(metrics["returned_episode_returns"], dtype=np.float64)
        update_steps = []
        mean_returns = []
        env_steps_per_update = num_steps * num_envs
        for update_idx in range(done.shape[0]):
            mask = done[update_idx]
            if not mask.any():
                continue
            update_steps.append(
                outer * ppo_timesteps + update_idx * env_steps_per_update
            )
            mean_returns.append(returns[update_idx][mask].mean())
        if not mean_returns:
            return
        update_steps = np.asarray(update_steps)
        mean_returns = np.asarray(mean_returns)
        if mean_returns.size > max_points:
            idx = _subsample_indices(mean_returns.size, max_points)
            update_steps = update_steps[idx]
            mean_returns = mean_returns[idx]
        for step, value in zip(update_steps, mean_returns, strict=True):
            self._writer.add_scalar(tag, float(value), int(step))

    def log_eval_return(self, dataset_size, mean_return):
        self._writer.add_scalar(
            "eval/mean_return", float(mean_return), int(dataset_size)
        )
        self._track_summary_metric("eval/mean_return", mean_return)

    def log_validation_metrics(self, dyn_mae, rew_mae, term_bce, term_f1, dataset_size):
        self._writer.add_scalar(
            "validation/dynamics_mae", float(dyn_mae), int(dataset_size)
        )
        self._writer.add_scalar(
            "validation/reward_mae", float(rew_mae), int(dataset_size)
        )
        self._writer.add_scalar(
            "validation/termination_bce", float(term_bce), int(dataset_size)
        )
        self._writer.add_scalar(
            "validation/termination_f1", float(term_f1), int(dataset_size)
        )
        self._track_summary_metric("validation/dynamics_mae", dyn_mae)
        self._track_summary_metric("validation/reward_mae", rew_mae)
        self._track_summary_metric("validation/termination_bce", term_bce)
        self._track_summary_metric("validation/termination_f1", term_f1)

    def close(self):
        if self._hparams:
            metrics = self._summary_metrics or {"hparam/complete": 1.0}
            self._writer.add_hparams(self._hparams, metrics)
        self._writer.close()


def log_validation(
    loggers,
    batched_val_metrics,
    models,
    val_obs,
    val_act,
    val_true_delta_obs,
    val_true_reward,
    val_true_terminated,
    dataset_ptr,
):
    dyn_mae, rew_mae, term_bce, term_f1 = batched_val_metrics(
        models,
        val_obs,
        val_act,
        val_true_delta_obs,
        val_true_reward,
        val_true_terminated,
    )
    for b, logger in enumerate(loggers):
        logger.log_validation_metrics(
            dyn_mae[b], rew_mae[b], term_bce[b], term_f1[b], dataset_ptr
        )


def log_eval(loggers, batched_eval, eval_policy_state, eval_keys, dataset_ptr):
    mean_returns = batched_eval(eval_policy_state, eval_keys)
    for b, logger in enumerate(loggers):
        logger.log_eval_return(dataset_ptr, mean_returns[b])
