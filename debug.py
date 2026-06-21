"""
Add this logging block INSIDE your main while-loop, right after
mujoco.mj_step(mj_model, data) and viewer.sync(), to see exactly what's
happening when the arm appears frozen. Print every 200 steps.

Paste this snippet into your real script temporarily:
"""

DEBUG_EVERY = 200

# put this right after step_counter += 1 at the bottom of the loop
if step_counter % DEBUG_EVERY == 0:
    print(f"\n[DEBUG step={step_counter}] state={state}")
    print(f"  ctrl[0:8]  = {np.round(data.ctrl[:8], 3)}")
    print(f"  qpos[0:7]  = {np.round(data.qpos[:7], 3)}  (arm joints)")
    print(f"  qvel[0:7]  = {np.round(data.qvel[:7], 3)}")
    print(f"  hand xpos  = {np.round(data.xpos[hand_id], 3)}")
    if carrying:
        print(f"  tool qpos[{qs}:{qs+3}] = {np.round(data.qpos[qs:qs+3], 3)}  <- being forced by _carry")
    # Check if anything looks pinned: if qvel stays ~0 every print while
    # ctrl is clearly nonzero and far from qpos, something external is
    # holding the joints (contact, weld, or a qpos overwrite elsewhere).