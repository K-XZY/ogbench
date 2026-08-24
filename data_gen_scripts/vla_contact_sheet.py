"""Render a camera-tuning contact sheet for the VLA dataset render path.

Renders every camera at several moments of a real oracle episode, including both candidate wrist
mounting sides, and reports framing and brightness statistics over a number of resets. This exists
to tune the camera poses in `manipspace_env` from images rather than from MJCF arithmetic.

Run on the box:
    MUJOCO_GL=egl python data_gen_scripts/vla_contact_sheet.py --save_dir /mnt/data/.../contact_sheet
"""

import json
import pathlib
from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from PIL import Image, ImageDraw

import ogbench.manipspace  # noqa
from ogbench.manipspace.envs.manipspace_env import (
    DEFAULT_RENDER_LIGHTING,
    DEFAULT_RENDER_ZNEAR,
    DEFAULT_WRIST_CAMERA,
    ManipSpaceEnv,
)
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-v0', 'Environment name.')
flags.DEFINE_string('save_dir', None, 'Directory to write the contact sheet and stats into.')
flags.DEFINE_integer('resolution', 256, 'Render resolution; the dataset master size.')
flags.DEFINE_integer('num_stat_resets', 20, 'Number of resets to measure framing and brightness over.')
flags.DEFINE_integer('max_episode_steps', 300, 'Cap on the demo episode used for the contact sheet.')
flags.DEFINE_bool('lighting', True, 'Whether to enable the extra render lighting.')
flags.DEFINE_string('rollout_path', None, 'If set, save the demo oracle rollout as an npz for Lane B.')
flags.DEFINE_float('headlight_ambient', None, 'Override the headlight ambient level.')
flags.DEFINE_float('headlight_diffuse', None, 'Override the headlight diffuse level.')
flags.DEFINE_float('workspace_diffuse', None, 'Override the workspace fill light diffuse level.')

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


def lighting_config():
    """Assemble the render lighting from flags, so levels can be swept without editing code."""
    if not FLAGS.lighting:
        return None

    config = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULT_RENDER_LIGHTING.items()}
    if FLAGS.headlight_ambient is not None:
        config['headlight_ambient'] = FLAGS.headlight_ambient
    if FLAGS.headlight_diffuse is not None:
        config['headlight_diffuse'] = FLAGS.headlight_diffuse
    if FLAGS.workspace_diffuse is not None:
        config['workspace_light']['diffuse'] = (FLAGS.workspace_diffuse,) * 3
    return config


def cube_color_fidelity(env, frames):
    """Measure whether cube colours survive the exposure.

    Cube colour is the object identity signal -- the dataset records a colour name per cube index and
    any later labelling leans on it -- so brightness that clips saturated colours toward white costs
    more than it buys. Reported per frame: the fraction of pixels clipping, and the mean HSV
    saturation of pixels whose hue matches a known cube colour.
    """
    unwrapped = env.unwrapped
    cube_rgb = (unwrapped._cube_colors[: unwrapped._num_cubes, :3] * 255.0).astype(np.float32)
    cube_unit = cube_rgb / np.linalg.norm(cube_rgb, axis=1, keepdims=True)

    out = {}
    for name, frame in frames.items():
        arr = frame.astype(np.float32)
        hi = arr.max(axis=-1)
        lo = arr.min(axis=-1)

        flat = arr.reshape(-1, 3)
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        unit = flat / np.clip(norms, 1e-6, None)
        # Match on hue direction only, so a washed-out cube still counts as a cube pixel and drags
        # the saturation number down instead of quietly dropping out of the population.
        is_cube = ((unit @ cube_unit.T).max(axis=1) > 0.995) & (norms[:, 0] > 40)

        sat = np.where(hi > 0, (hi - lo) / np.clip(hi, 1e-6, None), 0.0).reshape(-1)
        out[name] = dict(
            frac_clipped=float((hi >= 250).mean()),
            cube_pixel_frac=float(is_cube.mean()),
            mean_cube_saturation=float(sat[is_cube].mean()) if is_cube.any() else 0.0,
        )
    return out


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
        render_lighting=lighting_config(),
        pixel_recolor_arm=False,
        pixel_transparent_arm=False,
    )


def project(model, data, cam_id, points, height, width):
    """Project world points into pixel coordinates for one camera."""
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    focal = 0.5 * height / np.tan(0.5 * np.deg2rad(model.cam_fovy[cam_id]))

    out = []
    for xyz in points:
        rel = cam_mat.T @ (np.asarray(xyz) - cam_pos)
        depth = -rel[2]  # MuJoCo cameras look down -z of their own frame.
        if depth <= 0:
            continue
        out.append((0.5 * width + focal * rel[0] / depth, 0.5 * height - focal * rel[1] / depth))
    return out


def workspace_frame_fraction(env, frames):
    """Fraction of each frame spanned by the object sampling volume.

    The axis-aligned pixel bounding box of the eight corners of the sampling volume, clipped to the
    frame, over total image area. This is the number behind "the workspace fills a third of the
    image", and it is what the overhead pose should be tuned against.
    """
    unwrapped = env.unwrapped
    model, data = unwrapped._model, unwrapped._data
    (x_lo, y_lo), (x_hi, y_hi) = unwrapped._object_sampling_bounds
    corners = [(x, y, z) for x in (x_lo, x_hi) for y in (y_lo, y_hi) for z in (0.02, 0.10)]

    fractions = {}
    for name, frame in frames.items():
        height, width = frame.shape[:2]
        cam_id = model.camera(ManipSpaceEnv.resolve_camera_name(name)).id
        pts = project(model, data, cam_id, corners, height, width)
        if not pts:
            fractions[name] = 0.0
            continue
        xs, ys = zip(*pts)
        box_w = max(0.0, min(max(xs), width) - max(min(xs), 0.0))
        box_h = max(0.0, min(max(ys), height) - max(min(ys), 0.0))
        fractions[name] = float(box_w * box_h / (width * height))
    return fractions


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
        height, width = frame.shape[:2]
        cubes = [data.joint(f'object_joint_{i}').qpos[:3] for i in range(num_cubes)]
        pts = project(model, data, cam_id, cubes, height, width)
        # An in-frame test, not a visibility test: a cube hidden behind the arm still counts. It
        # bounds the framing question only; occlusion during a grasp is measured separately.
        visible[name] = sum(1 for px, py in pts if 0 <= px < width and 0 <= py < height)
    return visible


def collect_stats(env, camera_names, num_resets):
    """Measure cube visibility and brightness across resets."""
    per_camera = {
        name: dict(visible_counts=[], mean_full=[], mean_center=[], workspace_frac=[], clipped=[], cube_sat=[])
        for name in camera_names
    }

    for i in range(num_resets):
        env.reset(seed=FLAGS.seed + 10_000 + i)
        frames = env.unwrapped.render_cameras()
        visible = cube_pixel_coverage(env, frames)
        workspace = workspace_frame_fraction(env, frames)
        fidelity = cube_color_fidelity(env, frames)
        for name, frame in frames.items():
            height, width = frame.shape[:2]
            quarter_h, quarter_w = height // 4, width // 4
            center = frame[quarter_h : height - quarter_h, quarter_w : width - quarter_w]
            per_camera[name]['visible_counts'].append(visible[name])
            per_camera[name]['mean_full'].append(float(frame.mean()))
            # A whole-frame mean is dominated by dark floor and skybox when the workspace is small in
            # frame, so report the centre crop too before concluding anything about lighting.
            per_camera[name]['mean_center'].append(float(center.mean()))
            per_camera[name]['workspace_frac'].append(workspace[name])
            per_camera[name]['clipped'].append(fidelity[name]['frac_clipped'])
            per_camera[name]['cube_sat'].append(fidelity[name]['mean_cube_saturation'])

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
            std_brightness_full=float(np.std(rec['mean_full'])),
            mean_workspace_frame_fraction=float(np.mean(rec['workspace_frac'])),
            frac_pixels_clipped=float(np.mean(rec['clipped'])),
            mean_cube_saturation=float(np.mean(rec['cube_sat'])),
        )
    return summary


def run_demo_episode(env, camera_names, record_trajectory=False):
    """Roll the plan oracle out, keeping key frames and optionally the whole trajectory.

    The recorded trajectory is what Lane B measures grasp-time occlusion against: occlusion matters
    when the arm is over the cube it is manipulating, which never happens at reset.
    """
    unwrapped = env.unwrapped
    ob, info = env.reset(seed=FLAGS.seed)
    agent = CubePlanOracle(env=env, noise=0.0, noise_smoothing=0.5)
    agent.reset(ob, info)

    frames = unwrapped.render_cameras()
    keyframes = {'reset': frames}
    traj = None
    if record_trajectory:
        traj = defaultdict(list)
        traj['qpos'].append(info['qpos'])
        traj['qvel'].append(info['qvel'])
        for cam, frame in frames.items():
            traj[f'frames/{cam}'].append(frame)

    grasp_captured = False
    grasp_step = -1
    step = 0
    done = False

    while not done and step < FLAGS.max_episode_steps:
        action = np.clip(np.array(agent.select_action(ob, info)), -1, 1)
        ob, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        frames = None
        if record_trajectory:
            frames = unwrapped.render_cameras()
            traj['actions'].append(action)
            traj['qpos'].append(info['qpos'])
            traj['qvel'].append(info['qvel'])
            traj['gripper_contact'].append(info['proprio/gripper_contact'])
            traj['target_block'].append(info['privileged/target_block'])
            traj['segment_index'].append(info['privileged/segment_index'])
            traj['segment_start'].append(info['privileged/segment_start'])
            for cam, frame in frames.items():
                traj[f'frames/{cam}'].append(frame)

        # First frame where the gripper is actually holding something: the wrist view's whole reason
        # for existing, so it is the frame worth judging its pose on.
        if not grasp_captured and info['proprio/gripper_contact'][0] > 0.5:
            keyframes['grasp'] = frames if frames is not None else unwrapped.render_cameras()
            grasp_captured = True
            grasp_step = step

        if step == FLAGS.max_episode_steps // 4:
            keyframes['quarter'] = frames if frames is not None else unwrapped.render_cameras()

        if agent.done:
            break

    keyframes['end'] = frames if frames is not None else unwrapped.render_cameras()
    if not grasp_captured:
        print('WARNING: no grasp detected; the wrist frames show no held cube.', flush=True)
    return keyframes, traj, dict(num_steps=step, grasp_step=grasp_step)


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
    lighting_used = None

    for variant, wrist_camera in WRIST_VARIANTS.items():
        # The camera pose is baked into the compiled model, so each wrist side needs its own env.
        env = make_env(wrist_camera, base_cameras + ['wrist'])

        variant_stats = collect_stats(env, base_cameras + ['wrist'], FLAGS.num_stat_resets)
        # Only the default wrist side is worth saving a full rollout for.
        want_traj = FLAGS.rollout_path is not None and variant == 'wrist_negx'
        keyframes, traj, traj_meta = run_demo_episode(env, base_cameras + ['wrist'], record_trajectory=want_traj)

        if want_traj:
            rollout_path = pathlib.Path(FLAGS.rollout_path)
            rollout_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                rollout_path,
                **{k: np.asarray(v) for k, v in traj.items()},
                env_name=FLAGS.env_name,
                seed=FLAGS.seed,
                resolution=FLAGS.resolution,
                lighting=json.dumps(env.unwrapped.render_lighting),
                grasp_step=traj_meta['grasp_step'],
            )
            print(f'Wrote rollout {rollout_path} ({traj_meta})', flush=True)

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

        lighting_used = env.unwrapped.render_lighting
        env.close()

    camera_order = base_cameras + list(WRIST_VARIANTS.keys())
    moment_order = [m for m in ['reset', 'quarter', 'grasp', 'end'] if m in all_rows]
    build_contact_sheet({m: all_rows[m] for m in moment_order}, camera_order, save_dir / 'contact_sheet.png')

    stats_out = dict(
        env_name=FLAGS.env_name,
        resolution=FLAGS.resolution,
        num_stat_resets=FLAGS.num_stat_resets,
        seed=FLAGS.seed,
        lighting_enabled=FLAGS.lighting,
        lighting=lighting_used,
        cameras=stats,
    )
    (save_dir / 'camera_stats.json').write_text(json.dumps(stats_out, indent=2))
    print(json.dumps(stats_out, indent=2), flush=True)
    print(f'Wrote {save_dir / "contact_sheet.png"}', flush=True)


if __name__ == '__main__':
    app.run(main)
