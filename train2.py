import os
import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo.ppo import *
from ray.rllib.agents.ppo.ppo_torch_policy import *
from ray.rllib.agents.trainer_template import build_trainer
import shutil

from algorithm.env import MinichessEnv
from mini_train import MCGardnerNNet
from game.board import GardnerChessBoard

ray.init(ignore_reinit_error=True)

# print("Dashboard URL: http://{}".format(ray.get_webui_url()))

CHECKPOINT_ROOT = "tmp/ppo/minichess"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
config["framework"] = "torch"


model = MCGardnerNNet(GardnerChessBoard(), {"num_channels": 512, "dropout": 0.3, "cuda": False})


trainer = build_trainer(
    name="PPO",
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=build_policy_class(
        name="PPOTorchPolicy",
        framework="torch",
        get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
        loss_fn=ppo_surrogate_loss,
        stats_fn=kl_and_loss_stats,
        extra_action_out_fn=vf_preds_fetches,
        postprocess_fn=compute_gae_for_sample_batch,
        extra_grad_process_fn=apply_grad_clipping,
        before_init=setup_config,
        before_loss_init=setup_mixins,
        mixins=[
            LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
            ValueNetworkMixin
        ],
        make_model=None,
    ),
    execution_plan=execution_plan,
)
agent = trainer(config, env=MinichessEnv)


N_ITER = 30
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

for n in range(N_ITER):
  result = agent.train()
  file_name = agent.save(CHECKPOINT_ROOT)

  print(s.format(
    n + 1,
    result["episode_reward_min"],
    result["episode_reward_mean"],
    result["episode_reward_max"],
    result["episode_len_mean"],
    file_name
   ))