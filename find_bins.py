"""
Prints exact world XYZ of every bin body AND
tests 20 different joint1 values to map arm reach.
Run this, paste full output, I fix everything.
"""
import numpy as np
import mujoco
import mujoco.viewer
import time
import threading

MODEL_PATH = (
    r"C:\Users\srush\Downloads\mujoco_menagerie-main"
    r"\mujoco_menagerie-main\franka_emika_panda\manufacturing_scene.xml"
)

def main():
    mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data     = mujoco.MjData(mj_model)

    # Step once to get body positions
    mujoco.mj_forward(mj_model, data)

    print("\n" + "="*60)
    print("BIN WORLD POSITIONS")
    print("="*60)
    for name in ["bin_red", "bin_green", "bin_blue"]:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            pos = data.xpos[bid]
            print(f"  {name:12s} world XYZ = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        else:
            print(f"  {name:12s} NOT FOUND in scene")

    hand_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    print(f"\n  hand body_id = {hand_id}")

    print("\n" + "="*60)
    print("ARM REACH SWEEP — joint1 scan")
    print("Fixing joints 2-7, sweeping joint1 from -2.0 to +2.0")
    print("="*60)

    # Fixed pose for joints 2-7 (same as BIN_POSES)
    fixed = [0.0,  0.50,  0.00, -1.80,  0.00,  2.00,  0.80]

    sweep_values = np.linspace(-2.0, 2.0, 17)

    print(f"\n  {'joint1':>8}  {'hand_x':>8}  {'hand_y':>8}  {'hand_z':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for j1 in sweep_values:
        pose = [j1] + fixed[1:]
        for i, v in enumerate(pose):
            data.ctrl[i] = v
        # Step a few times to let arm settle
        for _ in range(200):
            mujoco.mj_step(mj_model, data)
        hp = data.xpos[hand_id]
        print(f"  {j1:>8.3f}  {hp[0]:>8.4f}  {hp[1]:>8.4f}  {hp[2]:>8.4f}")

    print("\n" + "="*60)
    print("DONE — paste this entire output")
    print("="*60)

if __name__ == "__main__":
    main()