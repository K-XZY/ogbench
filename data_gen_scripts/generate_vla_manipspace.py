"""Generate the VLA dataset from OGBench's manipspace oracles.

Derived from `generate_manipspace.py`. Differences that matter:

- **Per-episode records, not one flat npz.** Each episode is assembled as the
  frozen `schema.EPISODE_FIELDS` record and handed to the writer.
- **T+1 observations against T actions.** Upstream drops the frame after the last
  action; replay determinism needs it, because it is what that action is checked
  against.
- **Multi-camera renders** at the configured resolution, with the goal-leaking
  `visualize_info` overlay off.
- **Target and segment metadata** recorded per frame, which the released npz
  loses entirely.

Run on the box (nothing here runs on the Mac):
    MUJOCO_GL=egl python data_gen_scripts/generate_vla_manipspace.py \
        --env_name=cube-triple-v0 --dataset_type=play --num_episodes=50 \
        --save_root=/mnt/data/Work/goal-conditioned-vla/data
"""

import contextlib
import importlib
import json
import pathlib
import sys
import time

import gymnasium
import mujoco
import numpy as np
from absl import app, flags
from tqdm import trange

import ogbench.manipspace  # noqa
from ogbench.manipspace.envs.manipspace_env import DEFAULT_RENDER_ZNEAR
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Base random seed; episode i uses seed + i.')
flags.DEFINE_string('env_name', 'cube-triple-v0', 'Environment name.')
flags.DEFINE_string('dataset_type', 'play', "Oracle flavour: 'play' or 'noisy'.")
flags.DEFINE_string('save_root', None, 'Root under which the timestamped run directory is created.')
flags.DEFINE_integer('num_episodes', 50, 'Number of episodes to generate.')
flags.DEFINE_integer('max_episode_steps', 1001, 'Cap on episode length.')
flags.DEFINE_integer('resolution', 256, 'Master render resolution.')
flags.DEFINE_float('noise', 0.1, 'Action noise level.')
flags.DEFINE_float('noise_smoothing', 0.5, 'Action noise smoothing for the plan oracle.')
flags.DEFINE_float('min_norm', 0.4, 'Minimum action norm for the Markov oracle.')
flags.DEFINE_float('p_random_action', 0.0, 'Probability of a random action (noisy only).')
flags.DEFINE_bool('lighting', True, 'Enable the extra render lighting.')
flags.DEFINE_bool('dry_run', False, 'Assemble and validate episodes without writing them.')

# Lane B owns the shard writer (plan section 5). Import it lazily so --dry_run works before it lands.
WRITER_MODULE = 'goal_conditioned_vla.data.writer'


def _schema():
    """Import the frozen schema from the parent repo.

    This script lives in the ogbench submodule but the schema is the parent's, and it is deliberately
    the one shared artefact between the two lanes.
    """
    here = pathlib.Path(__file__).resolve()
    repo_src = here.parents[3] / 'src'
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from goal_conditioned_vla.data import schema

    return schema


schema = _schema()


def collect_provenance(writer_mod, repo_root):
    """The three fields that identify the data: parent SHA, ogbench SHA, MuJoCo version.

    A dirty tree is fatal for a real run -- data attributed to a commit that was never committed is
    not attributable -- but tolerated for `--dry_run`, which writes nothing and exists to be run
    mid-edit. `allow_dirty` stamps a `-dirty` suffix, so a dry run's provenance can never be
    mistaken for a clean one's.
    """
    return writer_mod.Provenance.collect(str(repo_root), allow_dirty=FLAGS.dry_run)


def run_config():
    """The full config, frozen to config.yaml by the writer. Section 3b: every run, no exceptions.

    This module's own flags only -- absl's logging flags are not part of what identifies a run.
    """
    own = FLAGS.flags_by_module_dict().get(__file__, [])
    return {flag.name: flag.value for flag in own}


def make_env():
    """Build the data-collection env on the VLA render path."""
    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=FLAGS.max_episode_steps,
        # Must be False: it paints the target ghost and success recolour into the pixels. The render
        # path refuses to build otherwise, and schema.validate_episode asserts it too.
        visualize_info=False,
        width=FLAGS.resolution,
        height=FLAGS.resolution,
        render_camera_names=list(schema.CAMERAS),
        # Driven off the schema rather than hardcoded, so the camera set follows the frozen contract
        # instead of needing a second edit whenever it changes.
        wrist_camera='wrist' in schema.CAMERAS,
        overhead_camera='overhead' in schema.CAMERAS,
        visual_znear=DEFAULT_RENDER_ZNEAR,
        render_lighting=True if FLAGS.lighting else None,
        pixel_recolor_arm=False,
        pixel_transparent_arm=False,
    )

    # `action_raw` is defined by the schema as `action_norm * ACTION_SCALE`, which is only true
    # because OGBench's action range is symmetric. Check rather than assume: a future upstream change
    # here would silently corrupt every unnormalized action in the dataset.
    unwrapped = env.unwrapped
    if not np.allclose(unwrapped.action_high, schema.ACTION_SCALE) or not np.allclose(
        unwrapped.action_low, -schema.ACTION_SCALE
    ):
        raise ValueError(
            f'env action range ({unwrapped.action_low}, {unwrapped.action_high}) no longer matches '
            f'schema.ACTION_SCALE {schema.ACTION_SCALE}; action_raw would be wrong'
        )
    return env


def make_agents(env):
    """Oracles keyed by target task, matching upstream's dispatch."""
    if 'cube' not in FLAGS.env_name:
        raise ValueError(f'{FLAGS.env_name} is not a cube environment; only cube envs are in scope.')

    if FLAGS.dataset_type == 'noisy':
        return {'cube': CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm)}
    return {'cube': CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing)}


def p_stack_for_env():
    """Cube stacking probability, matching upstream's per-env ranges."""
    name = FLAGS.env_name
    if 'single' in name:
        return 0.0
    if 'double' in name:
        return np.random.uniform(0.0, 0.25)
    if 'triple' in name:
        return np.random.uniform(0.05, 0.35)
    if 'quadruple' in name:
        return np.random.uniform(0.1, 0.5)
    if 'octuple' in name:
        return np.random.uniform(0.0, 0.35)
    return 0.5


def read_frame(env, num_cubes):
    """One FRAME row: everything observed at the current sim state.

    Called after `reset` and after every `step`, and crucially *after* any `set_new_target`, so the
    target and segment fields describe the subtask this frame actually belongs to.
    """
    unwrapped = env.unwrapped
    info = unwrapped.compute_ob_info()
    images = unwrapped.render_cameras()

    cube_quat = np.stack([info[f'privileged/block_{i}_quat'] for i in range(num_cubes)]).astype(np.float32)
    target_quat = np.asarray(info['privileged/target_block_quat'], dtype=np.float64)
    effector_pos = np.asarray(info['proprio/effector_pos'], dtype=np.float64)
    effector_yaw = float(info['proprio/effector_yaw'][0])
    target_pos = np.asarray(info['privileged/target_block_pos'], dtype=np.float64)

    row = {f'image_{cam}': images[cam] for cam in schema.CAMERAS}
    row.update(
        joint_pos=np.asarray(info['proprio/joint_pos'], dtype=np.float32),
        joint_vel=np.asarray(info['proprio/joint_vel'], dtype=np.float32),
        # The normalized `gripper_opening` is also recorded; this is the raw driver joint behind it.
        gripper_joint_pos=np.float32(unwrapped._data.qpos[unwrapped._gripper_opening_joint_id]),
        gripper_joint_vel=np.float32(info['proprio/gripper_vel'][0]),
        effector_pos=effector_pos.astype(np.float32),
        effector_yaw=np.float32(effector_yaw),
        gripper_opening=np.float32(info['proprio/gripper_opening'][0]),
        gripper_contact=np.float32(info['proprio/gripper_contact'][0]),
        # float64 verbatim: downcasting these would break the exact-replay guarantee they exist for.
        qpos=np.asarray(info['qpos'], dtype=np.float64),
        qvel=np.asarray(info['qvel'], dtype=np.float64),
        cube_pos=np.stack([info[f'privileged/block_{i}_pos'] for i in range(num_cubes)]).astype(np.float32),
        cube_quat=cube_quat,
        # Derived from the float32 quaternion that is actually stored, not from OGBench's float64
        # `block_i_yaw`. Same formula, but yaw is genuinely undefined when a cube tips onto an edge:
        # both arctan2 arguments collapse to ~1e-11, which survives in float64 and rounds to exactly
        # zero in float32, so the two disagree by radians. Deriving it here makes `cube_yaw` and
        # `cube_quat` consistent by construction. `cube_quat` remains the trustworthy field for a
        # tipped cube -- see the note in claude-notes on why yaw alone cannot be.
        cube_yaw=schema.quat_to_yaw(cube_quat).astype(np.float32),
        target_cube_idx=np.int32(info['privileged/target_block']),
        target_pos=target_pos.astype(np.float32),
        target_quat=target_quat.astype(np.float32),
        # World axes, not rotated into the EE frame. See the schema module docstring.
        target_pos_rel_ee=(target_pos - effector_pos).astype(np.float32),
        target_yaw_rel_ee=np.float32(
            schema.wrap_to_pi(schema.quat_to_yaw(target_quat) - effector_yaw)
        ),
        segment_idx=np.int32(info['privileged/segment_index']),
    )
    return row


def collect_episode(env, agents, seed):
    """Roll one episode out and assemble the schema record.

    Returns (arrays, meta). The frame/action alignment is the delicate part: frame t is the state
    *before* action t, so an episode of T actions has T+1 frames.
    """
    unwrapped = env.unwrapped
    num_cubes = unwrapped._num_cubes
    p_stack = p_stack_for_env()

    ob, info = env.reset(seed=seed)
    agent = agents[info['privileged/target_task']]
    agent.reset(ob, info)

    xi = np.random.uniform(0, FLAGS.noise) if FLAGS.dataset_type == 'noisy' else 0.0

    frames = [read_frame(env, num_cubes)]
    actions = []
    done = False

    while not done and len(actions) < FLAGS.max_episode_steps:
        if FLAGS.dataset_type == 'noisy' and np.random.rand() < FLAGS.p_random_action:
            action = env.action_space.sample()
        else:
            action = np.array(agent.select_action(ob, info))
            if FLAGS.dataset_type == 'noisy':
                action = action + np.random.normal(0, [xi, xi, xi, xi * 3, xi * 10], action.shape)
        action = np.clip(action, -1, 1)

        ob, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        actions.append(action)

        # Order matters. The oracle finishing its subtask starts a new segment, and the frame we are
        # about to record is the first frame of that new segment -- so re-target *before* reading it.
        if agent.done and not done:
            agent_ob, agent_info = unwrapped.set_new_target(p_stack=p_stack)
            agent = agents[agent_info['privileged/target_task']]
            agent.reset(agent_ob, agent_info)

        frames.append(read_frame(env, num_cubes))

    num_steps = len(actions)
    arrays = {}
    for spec in schema.EPISODE_FIELDS:
        # `segment_start` is a frame field but is derived from `segment_idx` below rather than read
        # per frame, so it is not present in the rows.
        if spec.axis != schema.FRAME or spec.name == 'segment_start':
            continue
        arrays[spec.name] = np.stack([f[spec.name] for f in frames]).astype(spec.dtype)

    action_norm = np.asarray(actions, dtype=np.float64).reshape(num_steps, schema.ACTION_DIM)
    arrays['action_norm'] = action_norm.astype(np.float32)
    arrays['action_raw'] = (action_norm * schema.ACTION_SCALE).astype(np.float32)

    # segment_start is derived from segment_idx rather than read from the env, so it cannot disagree
    # with it -- which is exactly the invariant the validator checks.
    seg = arrays['segment_idx'].astype(np.int64)
    segment_start = np.zeros(len(seg), dtype=bool)
    segment_start[0] = True
    segment_start[1:] = np.diff(seg) != 0
    arrays['segment_start'] = segment_start

    seg_ids = np.unique(seg)
    if not np.array_equal(seg_ids, np.arange(len(seg_ids))):
        raise ValueError(f'segment indices are not contiguous from 0: {seg_ids}')

    s_cube, s_start, s_end, s_success = [], [], [], []
    for i in seg_ids:
        run = np.nonzero(seg == i)[0]
        end = int(run[-1])
        cube = int(arrays['target_cube_idx'][end])
        distance = np.linalg.norm(
            arrays['cube_pos'][end, cube].astype(np.float64) - arrays['target_pos'][end].astype(np.float64)
        )
        s_cube.append(cube)
        s_start.append(int(run[0]))
        s_end.append(end)
        s_success.append(bool(distance <= schema.SUCCESS_RADIUS_M))

    arrays['segment_target_cube'] = np.asarray(s_cube, dtype=np.int32)
    arrays['segment_start_step'] = np.asarray(s_start, dtype=np.int32)
    arrays['segment_end_step'] = np.asarray(s_end, dtype=np.int32)
    arrays['segment_success'] = np.asarray(s_success, dtype=bool)

    meta = schema.EpisodeMeta(
        env_id=FLAGS.env_name,
        oracle_type=FLAGS.dataset_type,
        seed=seed,
        num_steps=num_steps,
        num_segments=len(seg_ids),
        num_cubes=num_cubes,
        model_nq=int(unwrapped._model.nq),
        model_nv=int(unwrapped._model.nv),
        cameras=list(schema.CAMERAS),
        image_height=FLAGS.resolution,
        image_width=FLAGS.resolution,
        visualize_info=False,
        cube_colors=list(schema.CUBE_COLOR_NAMES[:num_cubes]),
        # Resolved, read back from the compiled model. The ogbench SHA pins the pose *defaults*, but
        # any of them can be overridden per env instance and an override leaves no trace in the SHA,
        # so without this a run made before a camera moved is indistinguishable from one made after.
        render_config=unwrapped.render_config,
        mujoco_version=mujoco.__version__,
        git_sha_parent=GIT_SHA_PARENT,
        git_sha_ogbench=GIT_SHA_OGBENCH,
        control_timestep=float(unwrapped._control_timestep),
        physics_timestep=float(unwrapped._physics_timestep),
    )
    return arrays, meta


GIT_SHA_PARENT = ''
GIT_SHA_OGBENCH = ''


def main(_):
    global GIT_SHA_PARENT, GIT_SHA_OGBENCH

    if FLAGS.dataset_type not in ('play', 'noisy'):
        raise ValueError(f'dataset_type must be play or noisy, got {FLAGS.dataset_type!r}')
    if not FLAGS.dry_run and FLAGS.save_root is None:
        raise ValueError('--save_root is required unless --dry_run.')

    ogbench_root = pathlib.Path(__file__).resolve().parents[1]
    repo_root = ogbench_root.parents[1]
    writer_mod = importlib.import_module(WRITER_MODULE)
    provenance = collect_provenance(writer_mod, repo_root)
    GIT_SHA_PARENT = provenance.git_sha_parent
    GIT_SHA_OGBENCH = provenance.git_sha_ogbench
    mujoco_version = provenance.mujoco_version

    np.random.seed(FLAGS.seed)
    env = make_env()
    agents = make_agents(env)

    total_steps = 0
    total_segments = 0
    total_successes = 0
    started = time.time()

    with contextlib.ExitStack() as stack:
        run_writer = None
        if not FLAGS.dry_run:
            run_writer = stack.enter_context(
                writer_mod.RunWriter(
                    FLAGS.save_root,
                    env_id=FLAGS.env_name,
                    oracle_type=FLAGS.dataset_type,
                    config=run_config(),
                    provenance=provenance,
                )
            )
            print(f'Run directory: {run_writer.run_dir}', flush=True)

        for ep_idx in trange(FLAGS.num_episodes):
            arrays, meta = collect_episode(env, agents, seed=FLAGS.seed + ep_idx)

            if run_writer is None:
                # Writing validates and raises; only the dry run has to check for itself.
                problems = schema.validate_episode(arrays, meta)
                if problems:
                    raise ValueError(
                        f'episode {ep_idx} does not satisfy the schema:\n  ' + '\n  '.join(problems)
                    )
            else:
                run_writer.write_episode(arrays, meta, index=ep_idx)
                if ep_idx == 0:
                    # One reference still per camera, for eyeballing the run later.
                    for cam in schema.CAMERAS:
                        run_writer.write_camera_reference(cam, arrays[f'image_{cam}'][0])

            total_steps += meta.num_steps
            total_segments += meta.num_segments
            total_successes += int(arrays['segment_success'].sum())

        summary = dict(
            env_id=FLAGS.env_name,
            oracle_type=FLAGS.dataset_type,
            num_episodes=FLAGS.num_episodes,
            total_steps=total_steps,
            total_segments=total_segments,
            segment_success_rate=(total_successes / total_segments) if total_segments else 0.0,
            mean_steps_per_episode=total_steps / max(FLAGS.num_episodes, 1),
            wall_time_s=time.time() - started,
            steps_per_second=total_steps / max(time.time() - started, 1e-9),
            schema_version=schema.SCHEMA_VERSION,
            git_sha_parent=GIT_SHA_PARENT,
            git_sha_ogbench=GIT_SHA_OGBENCH,
            mujoco_version=mujoco_version,
            dry_run=FLAGS.dry_run,
        )
        print(json.dumps(summary, indent=2), flush=True)

        if run_writer is not None:
            # The writer owns summary.json; these are the generation-side aggregates it cannot know.
            run_writer.close(extra_summary=summary)


if __name__ == '__main__':
    app.run(main)
