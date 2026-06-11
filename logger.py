import numpy as np
from tensorboardX import SummaryWriter


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

    def log_loss_history(self, history, iteration):
        num_epochs = next(iter(history.values())).shape[0]
        for epoch in range(num_epochs):
            step = iteration * num_epochs + epoch
            for loss_name, loss_history in history.items():
                self._writer.add_scalar(
                    f"model/{loss_name}", float(loss_history[epoch]), step
                )

    def log_ppo_returns(self, timesteps, returns, tag):
        for step, value in zip(
            np.asarray(timesteps).flatten(),
            np.asarray(returns).flatten(),
            strict=True,
        ):
            self._writer.add_scalar(tag, float(value), int(step))

    def log_eval_return(self, dataset_size, mean_return):
        self._writer.add_scalar(
            "eval/mean_return", float(mean_return), int(dataset_size)
        )
        self._track_summary_metric("eval/mean_return", mean_return)

    def log_validation_metrics(
        self, dyn_mae, rew_mae, term_bce, term_f1, iteration
    ):
        self._writer.add_scalar("validation/dynamics_mae", float(dyn_mae), iteration)
        self._writer.add_scalar("validation/reward_mae", float(rew_mae), iteration)
        self._writer.add_scalar("validation/termination_bce", float(term_bce), iteration)
        self._writer.add_scalar("validation/termination_f1", float(term_f1), iteration)
        self._track_summary_metric("validation/dynamics_mae", dyn_mae)
        self._track_summary_metric("validation/reward_mae", rew_mae)
        self._track_summary_metric("validation/termination_bce", term_bce)
        self._track_summary_metric("validation/termination_f1", term_f1)

    def close(self):
        if self._hparams:
            metrics = self._summary_metrics or {"hparam/complete": 1.0}
            self._writer.add_hparams(self._hparams, metrics)
        self._writer.close()
