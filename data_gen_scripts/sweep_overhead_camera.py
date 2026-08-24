"""Sweep the overhead camera pose and pick the one that stays visible during a grasp.

A camera pointing straight down at the workspace centre shares its sight line with the arm's
approach corridor, so the UR5e links occlude the target cube during exactly the phase the dataset
exists to capture. Lane B's audit measured 60.6% mean target visibility with an unbroken 34-frame
blackout spanning approach->grasp->lift.

The metric here is deliberately the same one `scripts/camera_audit.py` reports, so numbers are
comparable: a segmentation pass, target cube visible when it covers >= `min_pixels`.

Camera height and fovy are held fixed. Raising a camera does not move a shared axis; only tilting
off vertical does. Candidates sit on a sphere of constant radius about the look-at point, so scale
is preserved and only the viewing angle changes.

Run on the box:
    MUJOCO_GL=egl python data_gen_scripts/sweep_overhead_camera.py --num_seeds=7
"""

import json

import gymnasium
import mujoco
import numpy as np
from absl import app, flags

import ogbench.manipspace  # noqa
from ogbench.manipspace.envs.manipspace_env import DEFAULT_OVERHEAD_CAMERA, DEFAULT_RENDER_ZNEAR
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'cube-triple-v0', 'Environment name.')
flags.DEFINE_integer('num_seeds', 7, 'Seeds per candidate; matches the audit that found the defect.')
flags.DEFINE_integer('max_steps', 160, 'Cap per rollout. Long enough to cover approach, grasp and lift.')
flags.DEFINE_integer('resolution', 256, 'Render resolution.')
flags.DEFINE_integer('min_pixels', 10, 'Pixels of the target cube needed to call it visible.')
flags.DEFINE_string('out', None, 'Optional path to write the full report as JSON.')

# Workspace centre, a little above the table: what every candidate aims at.
LOOK_AT = np.array([0.425, 0.0, 0.05])
# Baseline camera sits 0.7 up over a 0.02 table, so this preserves the framing scale.
DISTANCE = 0.68


def pose_from_angles(tilt_deg, azimuth_deg, distance=DISTANCE, look_at=LOOK_AT):
    """Camera config for a camera on a sphere about `look_at`, aimed at it.

    `tilt_deg` is the angle off vertical and `azimuth_deg` the direction of that tilt in the xy
    plane: 0 tilts toward +x (away from the robot base, toward where `front` already sits) and 90
    tilts toward +y (across the arm's approach corridor). At tilt 0 this reproduces the current
    straight-down pose exactly.
    """
    tilt, azimuth = np.deg2rad(tilt_deg), np.deg2rad(azimuth_deg)
    offset = np.array([
        np.sin(tilt) * np.cos(azimuth),
        np.sin(tilt) * np.sin(azimuth),
        np.cos(tilt),
    ])
    pos = look_at + distance * offset

    forward = look_at - pos
    forward /= np.linalg.norm(forward)
    # Image-up points toward the robot base, so the view stays oriented the way `front` is.
    up_ref = np.array([-1.0, 0.0, 0.0])
    right = np.cross(forward, up_ref)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    return dict(
        pos=tuple(float(v) for v in pos),
        xyaxes=tuple(float(v) for v in np.concatenate([right, up])),
        fovy=DEFAULT_OVERHEAD_CAMERA['fovy'],
    )


def candidates():
    """Baseline first, then lateral tilts, then tilts toward `front` for contrast."""
    out = [('baseline_straight_down', dict(DEFAULT_OVERHEAD_CAMERA))]
    for tilt in (15, 25, 35, 45):
        out.append((f'tilt{tilt}_y+', pose_from_angles(tilt, 90)))
    for tilt in (25, 35):
        out.append((f'tilt{tilt}_y-', pose_from_angles(tilt, -90)))
    # Toward +x is where `front` already lives; measured so the tradeoff is visible, not assumed.
    for tilt in (25, 35):
        out.append((f'tilt{tilt}_x+', pose_from_angles(tilt, 0)))
    return out


def measure(overhead_camera, seeds):
    """Roll the oracle out per seed and measure target-cube visibility from the overhead camera."""
    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=FLAGS.max_steps + 1,
        visualize_info=False,
        width=FLAGS.resolution,
        height=FLAGS.resolution,
        render_camera_names=['overhead'],
        overhead_camera=overhead_camera,
        visual_znear=DEFAULT_RENDER_ZNEAR,
        render_lighting=True,
        pixel_recolor_arm=False,
        pixel_transparent_arm=False,
    )
    inner = env.unwrapped
    model, data = inner._model, inner._data
    cam = inner.resolve_camera_name('overhead')
    cube_geoms = [set(ids) for ids in inner._cube_geom_ids_list]

    per_seed = []
    for seed in seeds:
        ob, info = env.reset(seed=seed)
        agent = CubePlanOracle(env=env, noise=0.0, noise_smoothing=0.5)
        agent.reset(ob, info)

        visible, grasping = [], []
        for _ in range(FLAGS.max_steps):
            target = int(info['privileged/target_block'])
            seg = np.asarray(inner.render(camera=cam, segmentation=True))
            geom_pix = np.where(seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[:, :, 0], -1)
            count = int(np.isin(geom_pix, list(cube_geoms[target])).sum())
            visible.append(count >= FLAGS.min_pixels)
            grasping.append(float(info['proprio/gripper_contact'][0]) > 0.5)

            action = np.clip(np.array(agent.select_action(ob, info)), -1, 1)
            ob, _, terminated, truncated, info = env.step(action)
            if terminated or truncated or agent.done:
                break

        visible = np.asarray(visible)
        grasping = np.asarray(grasping)

        # Longest unbroken blackout, and whether one of them swallows the grasp.
        hidden = ~visible
        longest, run = 0, 0
        blackout_over_grasp = False
        start = 0
        for i, h in enumerate(hidden):
            if h:
                if run == 0:
                    start = i
                run += 1
                longest = max(longest, run)
            else:
                if run and grasping[start:i].any():
                    blackout_over_grasp = True
                run = 0
        if run and grasping[start:len(hidden)].any():
            blackout_over_grasp = True

        per_seed.append(dict(
            seed=int(seed),
            frames=int(len(visible)),
            visible_frac=float(visible.mean()),
            visible_frac_at_grasp=float(visible[grasping].mean()) if grasping.any() else None,
            longest_hidden_run=int(longest),
            grasp_frames=int(grasping.sum()),
            blackout_spanning_grasp=bool(blackout_over_grasp),
        ))

    env.close()

    at_grasp = [s['visible_frac_at_grasp'] for s in per_seed if s['visible_frac_at_grasp'] is not None]
    return dict(
        per_seed=per_seed,
        mean_visible_frac=float(np.mean([s['visible_frac'] for s in per_seed])),
        worst_visible_frac=float(np.min([s['visible_frac'] for s in per_seed])),
        mean_visible_at_grasp=float(np.mean(at_grasp)) if at_grasp else None,
        worst_visible_at_grasp=float(np.min(at_grasp)) if at_grasp else None,
        max_hidden_run=int(np.max([s['longest_hidden_run'] for s in per_seed])),
        seeds_with_blackout_over_grasp=int(sum(s['blackout_spanning_grasp'] for s in per_seed)),
    )


def main(_):
    seeds = list(range(FLAGS.num_seeds))
    report = {}

    header = (f'{"candidate":<24} {"vis":>7} {"worst":>7} {"@grasp":>8} {"worst@g":>8} '
              f'{"maxrun":>7} {"blackouts":>10}')
    print(header)
    print('-' * len(header))

    for name, config in candidates():
        stats = measure(config, seeds)
        report[name] = dict(config={k: list(v) if isinstance(v, tuple) else v
                                    for k, v in config.items()}, **stats)
        mg = stats['mean_visible_at_grasp']
        wg = stats['worst_visible_at_grasp']
        print(f'{name:<24} {stats["mean_visible_frac"]:>6.1%} {stats["worst_visible_frac"]:>6.1%} '
              f'{(f"{mg:.1%}" if mg is not None else "n/a"):>8} '
              f'{(f"{wg:.1%}" if wg is not None else "n/a"):>8} '
              f'{stats["max_hidden_run"]:>7} '
              f'{stats["seeds_with_blackout_over_grasp"]:>4}/{len(seeds)}')

    print('\n"blackouts" = seeds where an unbroken hidden run covers a grasp frame. Target: 0.')

    if FLAGS.out:
        with open(FLAGS.out, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'wrote {FLAGS.out}')


if __name__ == '__main__':
    app.run(main)
