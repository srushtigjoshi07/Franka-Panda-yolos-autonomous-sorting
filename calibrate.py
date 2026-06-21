"""
Run this script to find the correct BIN_POSES and BIN_DROP values.
The arm will move to each pose and print the exact hand XYZ position.
Press ENTER in terminal to cycle through poses.
"""
import numpy as np
import mujoco
import mujoco.viewer
import threading

MODEL_PATH = (
    r"C:\Users\srush\Downloads\mujoco_menagerie-main"
    r"\mujoco_menagerie-main\franka_emika_panda\manufacturing_scene.xml"
)

# Poses to test — tweak joint1 (index 0) to rotate arm left/right
# Positive joint1 = rotate LEFT  (toward positive Y = BIN_RED at y=+0.50)
# Negative joint1 = rotate RIGHT (toward negative Y = BIN_BLUE at y=-0.50)
# Zero joint1     = forward       (toward BIN_GREEN at y=0.00)

TEST_POSES = {
    "PICKUP":    [0.00,  0.50,  0.00, -1.80,  0.00,  2.00,  0.80],
    "LIFT":      [0.00, -0.20,  0.00, -1.50,  0.00,  1.60,  0.80],
    "BIN_RED":   [1.20,  0.50,  0.00, -1.80,  0.00,  2.00,  0.80],
    "BIN_GREEN": [0.40,  0.50,  0.00, -1.80,  0.00,  2.00,  0.80],
    "BIN_BLUE":  [-0.40, 0.50,  0.00, -1.80,  0.00,  2.00,  0.80],
}

pose_names = list(TEST_POSES.keys())
pose_idx   = [0]   # mutable for thread access


def input_thread(mj_model, data):
    while True:
        input("\n>>> Press ENTER to go to next pose ...\n")
        pose_idx[0] = (pose_idx[0] + 1) % len(pose_names)
        name = pose_names[pose_idx[0]]
        pose = TEST_POSES[name]
        for i, v in enumerate(pose[:7]):
            data.ctrl[i] = v

        # Let physics settle then print hand position
        import time
        time.sleep(1.5)

        hand_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        hand_pos = data.xpos[hand_id].copy()
        print(f"\n{'='*50}")
        print(f"  POSE: {name}")
        print(f"  ctrl: {pose}")
        print(f"  hand XYZ = [{hand_pos[0]:.4f}, {hand_pos[1]:.4f}, {hand_pos[2]:.4f}]")
        print(f"{'='*50}")
        print(f"\n  Bin targets:")
        print(f"    BIN_RED   should be near [0.15,  0.50, 0.05]")
        print(f"    BIN_GREEN should be near [0.15,  0.00, 0.05]")
        print(f"    BIN_BLUE  should be near [0.15, -0.50, 0.05]")


def main():
    mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data     = mujoco.MjData(mj_model)

    # Start at PICKUP pose
    first_pose = TEST_POSES["PICKUP"]
    for i, v in enumerate(first_pose[:7]):
        data.ctrl[i] = v

    print("\n" + "="*50)
    print("  CALIBRATION MODE")
    print("="*50)
    print("  Starting at PICKUP pose.")
    print("  Press ENTER to cycle through bin poses.")
    print("  Read the hand XYZ and compare to bin positions.")
    print("="*50)
    print(f"\n  Bin positions from XML:")
    print(f"    bin_red   = [0.15,  0.50, 0.0]")
    print(f"    bin_green = [0.15,  0.00, 0.0]")
    print(f"    bin_blue  = [0.15, -0.50, 0.0]")

    # Start input thread
    t = threading.Thread(
        target=input_thread, args=(mj_model, data), daemon=True
    )
    t.start()

    with mujoco.viewer.launch_passive(mj_model, data) as viewer:
        viewer.cam.distance  = 2.2
        viewer.cam.azimuth   = 130
        viewer.cam.elevation = -18
        while viewer.is_running():
            mujoco.mj_step(mj_model, data)
            viewer.sync()


if __name__ == "__main__":
    main()