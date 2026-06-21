"""
Run this BEFORE the main sim to find out why the arm isn't moving.
It prints actuator info, applies a test ctrl, and shows whether qpos
actually changes after stepping.
"""

import numpy as np
import mujoco

MODEL_PATH = (
    r"C:\Users\srush\Downloads\mujoco_menagerie-main"
    r"\mujoco_menagerie-main\franka_emika_panda\manufacturing_scene.xml"
)

mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(mj_model)

print(f"nu (actuators) = {mj_model.nu}")
print(f"nq (qpos size) = {mj_model.nq}")
print(f"nv (qvel size) = {mj_model.nv}\n")

print("=== ACTUATOR LIST ===")
for i in range(mj_model.nu):
    name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    trntype = mj_model.actuator_trntype[i]
    gain = mj_model.actuator_gainprm[i][:3]
    bias = mj_model.actuator_biasprm[i][:3]
    ctrlrange = mj_model.actuator_ctrlrange[i]
    ctrllimited = mj_model.actuator_ctrllimited[i]
    joint_id = mj_model.actuator_trnid[i][0]
    jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    print(f"  [{i}] name={name!r:25s} -> joint={jname!r:20s} "
          f"trntype={trntype} ctrlrange={ctrlrange} ctrllimited={ctrllimited}")
    print(f"        gainprm={gain}  biasprm={bias}")

print("\n=== JOINT LIST (first 12) ===")
for i in range(min(mj_model.njnt, 12)):
    jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jtype = mj_model.jnt_type[i]
    qadr = mj_model.jnt_qposadr[i]
    jrange = mj_model.jnt_range[i]
    print(f"  [{i}] name={jname!r:20s} type={jtype} qposadr={qadr} range={jrange}")

# ---- TEST: apply a big ctrl signal and see if joints actually move ----
print("\n=== STEP TEST ===")
mujoco.mj_resetData(mj_model, data)
mujoco.mj_forward(mj_model, data)

before = data.qpos.copy()

# Try to set ctrl for the first 7 actuators (assumed arm joints) to a nonzero pose
test_pose = [0.5, 0.5, 0.3, -1.8, 0.2, 1.8, 0.5]
for i, v in enumerate(test_pose):
    if i < mj_model.nu:
        data.ctrl[i] = v

print(f"ctrl set to: {data.ctrl[:8]}")

for step in range(2000):
    mujoco.mj_step(mj_model, data)

after = data.qpos.copy()

print(f"\nqpos BEFORE (first 7): {before[:7]}")
print(f"qpos AFTER  (first 7): {after[:7]}")
diff = after[:7] - before[:7]
print(f"DIFF                 : {diff}")

if np.allclose(diff, 0, atol=1e-4):
    print("\n[RESULT] Joints did NOT move after 2000 steps with ctrl set.")
    print("  Likely causes:")
    print("   1. ctrlrange clips your values to 0 (check ctrllimited/ctrlrange above)")
    print("   2. Actuator is velocity/torque type, not position — ctrl=0.5 means")
    print("      'small velocity/torque', not 'go to angle 0.5'")
    print("   3. Wrong actuator index -> joint mapping (check actuator list above")
    print("      to confirm data.ctrl[0] really drives shoulder_pan, etc.)")
    print("   4. A keyframe / equality constraint / weld is holding the arm fixed")
    print("   5. gainprm/biasprm indicate it's NOT a standard position servo")
else:
    print("\n[RESULT] Joints DID move. The actuator mapping is correct.")
    print("  -> The bug is in your main script's logic, not the model.")
    print("  -> Check: is data.ctrl being overwritten elsewhere right before")
    print("     mj_step? Is launch_passive's viewer thread resetting ctrl?")

# ---- Check for equality constraints / welds that could lock the arm ----
print("\n=== EQUALITY CONSTRAINTS ===")
if mj_model.neq == 0:
    print("  None found.")
else:
    for i in range(mj_model.neq):
        ename = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
        etype = mj_model.eq_type[i]
        active = data.eq_active[i] if hasattr(data, "eq_active") else "?"
        print(f"  [{i}] name={ename!r} type={etype} active={active}")

# ---- Check keyframes (a 'home' keyframe could be resetting pose) ----
print("\n=== KEYFRAMES ===")
if mj_model.nkey == 0:
    print("  None found.")
else:
    for i in range(mj_model.nkey):
        kname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_KEY, i)
        print(f"  [{i}] name={kname!r}")