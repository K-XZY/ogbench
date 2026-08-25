"""Does a mid-episode restore from qpos/qvel reproduce the control exactly?

`set_control` reads `site_xpos`, which `mj_step` leaves stale relative to `qpos`: the forward pass
runs at the current state and integration follows, and with 25 physics substeps per control step the
lag is one substep -- a configuration that appears in no recorded frame. So with default dynamics,
`qpos`/`qvel` are *not* sufficient to reproduce a mid-episode control, and `consistent_kinematics`
is what makes them sufficient.

This test asserts both halves, because a test that only checked the fixed case would not show what
the flag is for -- and a check that cannot come out the other way is not a check.

Run on the box:
    MUJOCO_GL=egl python data_gen_scripts/test_state_restore.py
"""

import sys

import gymnasium
import numpy as np

import ogbench.manipspace  # noqa
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle

CAPTURE_AT = (40, 60, 80)
STEPS = 100


def make(consistent):
    return gymnasium.make(
        'cube-triple-v0', terminate_at_goal=False, mode='data_collection',
        max_episode_steps=STEPS + 1, visualize_info=False, width=64, height=64,
        consistent_kinematics=consistent,
    )


def rollout(consistent, seed=3):
    """Record (qpos, qvel, action, resulting ctrl) at a few mid-episode steps."""
    env = make(consistent)
    inner = env.unwrapped
    ob, info = env.reset(seed=seed)
    agent = CubePlanOracle(env=env, noise=0.1, noise_smoothing=0.5)
    agent.reset(ob, info)

    captured = {}
    for t in range(STEPS):
        action = np.clip(np.array(agent.select_action(ob, info)), -1, 1)
        if t in CAPTURE_AT:
            qpos = inner._data.qpos.copy()
            qvel = inner._data.qvel.copy()
            inner.set_control(action)
            captured[t] = (qpos, qvel, action.copy(), inner._data.ctrl.copy())
        ob, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    return captured


def restore_error(consistent, captured, seed=3):
    worst, exact = 0.0, 0
    for qpos, qvel, action, ctrl in captured.values():
        env = make(consistent)
        inner = env.unwrapped
        env.reset(seed=seed)
        inner.set_state(qpos, qvel)
        inner.set_control(action)
        worst = max(worst, float(np.abs(inner._data.ctrl - ctrl).max()))
        exact += int(np.array_equal(inner._data.ctrl, ctrl))
        env.close()
    return worst, exact


def main():
    failures = []

    worst_on, exact_on = restore_error(True, rollout(True))
    print('consistent_kinematics=True :  max|dctrl| = %.3e   bit-identical %d/%d'
          % (worst_on, exact_on, len(CAPTURE_AT)))
    if exact_on != len(CAPTURE_AT):
        failures.append(
            'restore is not bit-identical with consistent_kinematics=True: there is further hidden '
            'state beyond the stale kinematics, and it must be found before scaling'
        )

    # The negative half. If this ever passes, the flag has stopped doing anything and the test above
    # would be green for the wrong reason.
    worst_off, exact_off = restore_error(False, rollout(False))
    print('consistent_kinematics=False:  max|dctrl| = %.3e   bit-identical %d/%d'
          % (worst_off, exact_off, len(CAPTURE_AT)))
    if exact_off == len(CAPTURE_AT):
        failures.append(
            'restore is bit-identical with the flag OFF, so the test cannot distinguish the fix from '
            'its absence'
        )

    if failures:
        print('\nFAILED:')
        for f in failures:
            print('  ' + f)
        return 1
    print('\nstate restore behaves as documented in both directions')
    return 0


if __name__ == '__main__':
    sys.exit(main())
