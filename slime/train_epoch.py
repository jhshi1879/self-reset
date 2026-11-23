import ray
import os
import shutil
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
except ImportError:
    GPU_MEMORY_TYPE_CUDA_GRAPH = None

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.baseline_eval import reuse_initial_eval
from slime.utils.wandb_utils import init_wandb_primary
from slime.rollout.Guardforward import ContentEvaluator

def delete_old_checkpoints(checkpoint_dir):
    tracker_path = os.path.join(checkpoint_dir, "latest_checkpointed_iteration.txt")
    with open(tracker_path, encoding="utf-8") as tracker_file:
        latest_iteration_text = tracker_file.read().strip()
    if not latest_iteration_text.isdigit():
        raise ValueError(f"Invalid checkpoint iteration in {tracker_path}: {latest_iteration_text!r}")

    latest_iteration = int(latest_iteration_text)
    latest_checkpoint = os.path.join(checkpoint_dir, f"iter_{latest_iteration:07d}")
    if not os.path.isdir(latest_checkpoint):
        raise FileNotFoundError(f"Latest checkpoint directory not found: {latest_checkpoint}")

    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if (
                entry.is_dir(follow_symlinks=False)
                and entry.name.startswith("iter_")
                and entry.name[5:].isdigit()
                and int(entry.name[5:]) != latest_iteration
            ):
                shutil.rmtree(entry.path)
                print(f"Deleted old checkpoint: {entry.path}", flush=True)

def train(args):
    reused_initial_eval = reuse_initial_eval(args)
    requested_rollout_limit = args.num_rollout
    epoch_limit = None if requested_rollout_limit is not None else args.num_epoch

    # allocate the GPUs
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"], wandb_run_id)
    
    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager, wandb_run_id)

    if args.offload_rollout:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.offload_rollout:
        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    epoch_now = 0
    rollout_id = args.start_rollout_id
    rollouts_completed = 0
    while (epoch_limit is None or epoch_now < epoch_limit) and (
        requested_rollout_limit is None or rollouts_completed < requested_rollout_limit
    ):
        if args.eval_interval is not None and rollout_id == 0 and not args.evaluation_only and not reused_initial_eval:
            ray.get(rollout_manager.eval.remote(rollout_id))

        if args.evaluation_only: # evaluation only
            print(f"Evaluating at rollout_id {rollout_id}")
            ray.get(rollout_manager.eval.remote(rollout_id))
            break

        last_epoch_id = ray.get(rollout_manager.get_epoch_id.remote())

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        now_epoch_id = ray.get(rollout_manager.get_epoch_id.remote())
        epoch_now = ray.get(rollout_manager.get_epoch_now.remote())
        is_final_rollout = (
            requested_rollout_limit is not None
            and rollouts_completed + 1 >= requested_rollout_limit
        )

        if (
            (
                (args.save_interval is not None and (rollout_id + 1) % args.save_interval == 0)
                or last_epoch_id != now_epoch_id
                or rollout_id == 74
                or is_final_rollout
            )
            and not args.no_save
        ):
            # if os.path.exists(args.save):
            #     if os.path.isdir(args.save):
            #         print(f"Removing directory: {args.save}")
            #         shutil.rmtree(args.save)
            #     else:
            #         print(f"Removing file: {args.save}")
            #         os.remove(args.save)
                    
            if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
                actor_model.save_model(rollout_id)
                if args.keep_only_last_checkpoint:
                    delete_old_checkpoints(args.save)
            if args.use_critic:
                critic_model.save_model(rollout_id)
                if args.keep_only_last_checkpoint and args.critic_save is not None:
                    delete_old_checkpoints(args.critic_save)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))
            if last_epoch_id != now_epoch_id:
                print(f"saved epoch {now_epoch_id} at rollout_id {rollout_id}", flush=True)

        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()

        if args.offload_rollout:
            if not args.offload_train:
                actor_model.clear_memory()
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

        actor_model.update_weights()

        if args.offload_rollout:
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        if (
            (args.eval_interval is not None and (rollout_id + 1) % args.eval_interval == 0)
            or last_epoch_id != now_epoch_id
            or rollout_id == 74
            or is_final_rollout
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))
            if last_epoch_id != now_epoch_id:
                print(f"evaluation epoch {now_epoch_id} at rollout_id {rollout_id}", flush=True)
                
        rollout_id += 1
        rollouts_completed += 1
        print(f"last_epoch_id: {last_epoch_id}, now_epoch_id: {now_epoch_id}, epoch_now: {epoch_now}, rollout_id: {rollout_id}", flush=True)

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
