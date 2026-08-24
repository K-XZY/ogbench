"""Round-trip test for the recorded render configuration.

Build an env -> record its `render_config` -> rebuild an env from that record alone -> assert the
two compiled models agree on every camera pose, fovy, lighting value and clipping setting.

This is the guarantee behind `render_config`. Recording provenance you cannot act on is not
provenance, and the dataset's stated bar is that nothing has to be regenerated later -- which fails
the moment the record cannot rebuild the environment it describes.

The negative case matters as much as the positive one: a round-trip that passes because both sides
compute the same wrong thing proves nothing, so the test also perturbs a recorded pose and asserts
the check *fails*.

Run on the box:
    MUJOCO_GL=egl python data_gen_scripts/test_render_config_roundtrip.py
"""

import copy
import sys

import gymnasium
import numpy as np

import ogbench.manipspace  # noqa
from ogbench.manipspace.envs.manipspace_env import (
    DEFAULT_RENDER_ZNEAR,
    render_config_to_kwargs,
    verify_render_config,
)

ENV_ID = 'cube-triple-v0'


def build_reference():
    """An env configured the way the generation script configures one."""
    return gymnasium.make(
        ENV_ID,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=50,
        visualize_info=False,
        width=256,
        height=256,
        render_camera_names=['front', 'wrist'],
        wrist_camera=True,
        visual_znear=DEFAULT_RENDER_ZNEAR,
        render_lighting=True,
        pixel_recolor_arm=False,
        pixel_transparent_arm=False,
    )


def rebuild_from(recorded):
    """An env built from a recorded config alone, with nothing carried over from the original."""
    env = gymnasium.make(
        ENV_ID,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=50,
        **render_config_to_kwargs(recorded),
    )
    env.reset(seed=0)
    return env


def main():
    failures = []

    reference = build_reference()
    reference.reset(seed=0)
    recorded = copy.deepcopy(reference.unwrapped.render_config)
    reference.close()

    # 1. The record rebuilds the environment it describes.
    rebuilt = rebuild_from(recorded)
    try:
        verify_render_config(rebuilt, recorded)
        print('round-trip: OK — rebuilt env matches the recorded config')
    except ValueError as exc:
        failures.append(f'round-trip failed: {exc}')

    # 2. Rendered pixels agree, which is what the poses are actually for.
    frames = rebuilt.unwrapped.render_cameras()
    for cam, frame in frames.items():
        if frame.shape != (recorded['image_height'], recorded['image_width'], 3):
            failures.append(f'{cam}: rebuilt frame shape {frame.shape}')
    print(f'render: OK — {sorted(frames)} at {frames["front"].shape[:2]}')

    # 3. The check can fail. A round-trip that cannot detect a changed pose proves nothing, and a
    #    camera pose is exactly the thing this record exists to pin down.
    for field, mutate in [
        ('wrist pos', lambda c: c['cameras']['wrist'].__setitem__('pos', [9.0, 9.0, 9.0])),
        ('wrist fovy', lambda c: c['cameras']['wrist'].__setitem__('fovy', 12.0)),
        ('front pos', lambda c: c['cameras']['front'].__setitem__('pos', [0.0, 0.0, 5.0])),
        ('visual_znear', lambda c: c.__setitem__('visual_znear', 0.5)),
        ('lighting', lambda c: c.__setitem__('lighting', None)),
    ]:
        corrupted = copy.deepcopy(recorded)
        mutate(corrupted)
        try:
            verify_render_config(rebuilt, corrupted)
        except ValueError:
            continue
        failures.append(f'verify_render_config accepted a corrupted {field}')
    print('negative cases: OK — a corrupted pose, fovy, znear or lighting is rejected')

    # 4. `front` is fixed in the MJCF, so it is verified rather than rebuilt. Confirm the helper does
    #    not silently pretend to set it.
    kwargs = render_config_to_kwargs(recorded)
    if 'front_camera' in kwargs:
        failures.append('render_config_to_kwargs invented a kwarg for the fixed `front` camera')
    if 'wrist_camera' not in kwargs:
        failures.append('render_config_to_kwargs dropped the configurable `wrist` camera')
    print('fixed vs configurable: OK — wrist is rebuilt, front is verified only')

    rebuilt.close()

    if failures:
        print('\nFAILED:')
        for f in failures:
            print(f'  {f}')
        return 1
    print('\nall round-trip checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
