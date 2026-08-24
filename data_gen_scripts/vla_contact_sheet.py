"""Render a camera-tuning contact sheet for the VLA dataset render path.

Renders every camera at several moments of a real oracle episode, including both candidate wrist
mounting sides, and reports framing and brightness statistics over a number of resets. This exists
to tune the camera poses in `manipspace_env` from images rather than from MJCF arithmetic.

Run on the box:
    MUJOCO_GL=egl python data_gen_scripts/vla_contact_sheet.py --save_dir /mnt/data/.../contact_sheet
"""

import json
import pathlib

import gymnasium
import numpy as np
from absl import app, flags
from PIL import Image, ImageDraw

import ogbench.manipspace  # noqa
from ogbench.manipspace.envs.manipspace_env import DEFAULT_RENDER_ZNEAR, DEFAULT_WRIST_CAMERA, ManipSpaceEnv
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-v0', 'Environment name.')
flags.DEFINE_string('save_dir', None, 'Directory to write the contact sheet and stats into.')
flags.DEFINE_integer('resolution', 256, 'Render resolution; the dataset master size.')
flags.DEFINE_integer('num_stat_resets', 20, 'Number of resets to measure framing and brightness over.')
flags.DEFINE_integer('max_episode_steps', 300, 'Cap on the demo episode used for the contact sheet.')

# The wrist camera is mounted off one side of the gripper `base` frame. Which side is clean and which
# is behind a linkage is not answerable from the MJCF, so render both and look.
WRIST_VARIANTS = {
    # Default: camera on the -x side, looking back at the pinch site.
    'wrist_negx': dict(DEFAULT_WRIST_CAMERA),
    # Mirrored through the x=0 plane. `xyaxes` is re-derived, not sign-flipped: image-right stays the
    # finger-opening axis and image-up keeps a -z component in the base frame, so the view is not
    # upside down.
    'wrist_posx': dict(
        pos=(0.06, 0.0, 0.03),
        xyaxes=(0.0, -1.0, 0.0, -0.8866, 0.0, -0.4626),
        fovy=DEFAULT_WRIST_CAMERA['fovy'],
    ),
}


def make_env(wrist_camera, render_camera_names):
    """Build a data-collection env on the VLA render path."""
    return gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=FLAGS.max_episode_steps,
        # The render path refuses to build with visualize_info=True, which would paint the goal in.
        visualize_info=False,
        width=FLAGS.resolution,
        height=FLAGS.resolution,
        render_camera_names=render_camera_names,
        wrist_camera=wrist_camera,
        overhead_camera=True,
        visual_znear=DEFAULT_RENDER_ZNEAR,
        pixel_recolor_arm=False,
        pixel_transparent_arm=False,
    )


def cube_pixel_coverage(env, frames):
    """Fraction of each frame occupied by the workspace, approximated by the cube target region.

    Uses the projected positions of the cubes rather than a segmentation pass: for each camera we
    report how many of the cubes fall inside the image, which is the thing under dispute.
    """
    unwrapped = env.unwrapped
    num_cubes = unwrapped._num_cubes
    model, data = unwrapped._model, unwrapped._data

    visible = {}
    for name, frame in frames.items():
        cam_id = model.camera(ManipSpaceEnv.resolve_camera_name(name)).id
        # World -> camera frame.
        cam_pos = data.cam_xpos[cam_id]
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
        fovy = np.deg2rad(model.cam_fovy[cam_id])
        height, width = frame.shape[:2]
        focal = 0.5 * height / np.tan(0.5 * fovy)

        count = 0
        for i in range(num_cubes):
            xyz = data.joint(f'object_joint_{i}').qpos[:3]
            rel = cam_mat.T @ (xyz - cam_pos)
            # MuJoCo cameras look down -z of their own frame.
            depth = -rel[2]
            if depth <= 0:
                continue
            px = 0.5 * width + focal * rel[0] / depth
            py = 0.5 * height - focal * rel[1] / depth
            if 0 <= px < width and 0 <= py < height:
                count += 1
        visible[name] = count
    return visible


def collect_stats(env, camera_names, num_resets):
    """Measure cube visibility and brightness across resets."""
    per_camera = {name: dict(visible_counts=[], mean_full=[], mean_center=[]) for name in camera_names}

    for i in range(num_resets):
        env.reset(seed=FLAGS.seed + 10_000 + i)
        frames = env.unwrapped.render_cameras()
        visible = cube_pixel_coverage(env, frames)
        for name, frame in frames.items():
            height, width = frame.shape[:2]
            quarter_h, quarter_w = height // 4, width // 4
            center = frame[quarter_h : height - quarter_h, quarter_w : width - quarter_w]
            per_camera[name]['visible_counts'].append(visible[name])
            per_camera[name]['mean_full'].append(float(frame.mean()))
            # A whole-frame mean is dominated by dark floor and skybox when the workspace is small in
            # frame, so report the centre crop too before concluding anything about lighting.
            per_camera[name]['mean_center'].append(float(center.mean()))

    num_cubes = env.unwrapped._num_cubes
    summary = {}
    for name, rec in per_camera.items():
        counts = np.array(rec['visible_counts'])
        summary[name] = dict(
            num_cubes=num_cubes,
            mean_cubes_in_frame=float(counts.mean()),
            frac_resets_all_cubes_visible=float((counts == num_cubes).mean()),
            min_cubes_in_frame=int(counts.min()),
            mean_brightness_full=float(np.mean(rec['mean_full'])),
            mean_brightness_center=float(np.mean(rec['mean_center'])),
        )
    return summary


def run_demo_episode(env, camera_names):
    """Roll the plan oracle out and keep frames at reset, first grasp, mid-reach and episode end."""
    unwrapped = env.unwrapped
    ob, info = env.reset(seed=FLAGS.seed)
    agent = CubePlanOracle(env=env, noise=0.0, noise_smoothing=0.5)
    agent.reset(ob, info)

    keyframes = {'reset': env.unwrapped.render_cameras()}
    grasp_captured = False
    step = 0
    done = False

    while not done and step < FLAGS.max_episode_steps:
        action = np.clip(np.array(agent.select_action(ob, info)), -1, 1)
        ob, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        # First frame where the gripper is actually holding something: the wrist view's whole reason
        # for existing, so it is the frame worth judging its pose on.
        if not grasp_captured and info['proprio/gripper_contact'][0] > 0.5:
            keyframes['grasp'] = unwrapped.render_cameras()
            grasp_captured = True

        if step == FLAGS.max_episode_steps // 4:
            keyframes['quarter'] = unwrapped.render_cameras()

        if agent.done:
            break

    keyframes['end'] = unwrapped.render_cameras()
    if not grasp_captured:
        print('WARNING: no grasp detected; the wrist frames show no held cube.', flush=True)
    return keyframes


def build_contact_sheet(rows, camera_order, path, pad=6, label_h=22):
    """Tile frames into a labelled grid: one row per moment, one column per camera."""
    tile = FLAGS.resolution
    row_names = list(rows.keys())
    width = pad + len(camera_order) * (tile + pad)
    height = label_h + pad + len(row_names) * (tile + pad + label_h)

    sheet = Image.new('RGB', (width, height), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)

    for col, cam in enumerate(camera_order):
        x = pad + col * (tile + pad)
        draw.text((x + 4, 6), cam, fill=(235, 235, 235))

    y = label_h + pad
    for row_name in row_names:
        frames = rows[row_name]
        draw.text((pad + 4, y - 2), row_name, fill=(160, 200, 255))
        for col, cam in enumerate(camera_order):
            x = pad + col * (tile + pad)
            frame = frames.get(cam)
            if frame is None:
                continue
            sheet.paste(Image.fromarray(frame), (x, y + label_h - 4))
        y += tile + pad + label_h

    sheet.save(path)


def main(_):
    assert FLAGS.save_dir is not None, '--save_dir is required.'
    save_dir = pathlib.Path(FLAGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_cameras = ['front', 'overhead']
    all_rows = {}
    stats = {}

    for variant, wrist_camera in WRIST_VARIANTS.items():
        # The camera pose is baked into the compiled model, so each wrist side needs its own env.
        env = make_env(wrist_camera, base_cameras + ['wrist'])

        variant_stats = collect_stats(env, base_cameras + ['wrist'], FLAGS.num_stat_resets)
        keyframes = run_demo_episode(env, base_cameras + ['wrist'])

        for moment, frames in keyframes.items():
            row = all_rows.setdefault(moment, {})
            # Front and overhead do not depend on the wrist side; keep the first variant's copy.
            for cam in base_cameras:
                row.setdefault(cam, frames[cam])
            row[variant] = frames['wrist']

        stats[variant] = variant_stats['wrist']
        for cam in base_cameras:
            stats.setdefault(cam, variant_stats[cam])

        for moment, frames in keyframes.items():
            for cam, frame in frames.items():
                name = variant if cam == 'wrist' else cam
                Image.fromarray(frame).save(save_dir / f'{moment}__{name}.png')

        env.close()

    camera_order = base_cameras + list(WRIST_VARIANTS.keys())
    moment_order = [m for m in ['reset', 'quarter', 'grasp', 'end'] if m in all_rows]
    build_contact_sheet({m: all_rows[m] for m in moment_order}, camera_order, save_dir / 'contact_sheet.png')

    stats_out = dict(
        env_name=FLAGS.env_name,
        resolution=FLAGS.resolution,
        num_stat_resets=FLAGS.num_stat_resets,
        seed=FLAGS.seed,
        cameras=stats,
    )
    (save_dir / 'camera_stats.json').write_text(json.dumps(stats_out, indent=2))
    print(json.dumps(stats_out, indent=2), flush=True)
    print(f'Wrote {save_dir / "contact_sheet.png"}', flush=True)


if __name__ == '__main__':
    app.run(main)
