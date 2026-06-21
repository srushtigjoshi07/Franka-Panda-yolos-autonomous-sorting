import time
import threading
import random

import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# =============================================================================
# PATHS
# =============================================================================

MODEL_PATH = (
    r"C:\Users\srush\Downloads\mujoco_menagerie-main"
    r"\mujoco_menagerie-main\franka_emika_panda\manufacturing_scene.xml"
)

FINETUNED_PATH = r"C:\Users\srush\Desktop\yolos_finetuned\best"

# =============================================================================
# CONVEYOR & TOOL CONFIG
# =============================================================================

CONV_SURFACE_Z = 0.24

TOOL_REST_Z = {
    "spanner": CONV_SURFACE_Z + 0.065,
    "bolt":    CONV_SURFACE_Z + 0.037,
    "nut":     CONV_SURFACE_Z + 0.008,
}

# World-X belt geometry (belt body is at world x=0.35)
# REVERSED: tools now enter near the arm's far side and travel the opposite
# way along the belt before reaching the pickup point.
SPAWN_X  = 0.20    # near end — new tools enter here (widened track)
PICKUP_X = 0.55    # arm grips tools here (belt moves +X toward pickup)
PICKUP_Y = 0.00

# Continuous belt speed: world units (meters) per real second.
BELT_SPEED_MPS = 0.06        # tune for visual pacing
BELT_DIRECTION = +1          # +1 = moving toward PICKUP_X from SPAWN_X

# Fixed spacing between consecutive tools on the belt (world units).
# Track length is (PICKUP_X - SPAWN_X) = 0.35 m here, so spacing must be
# comfortably smaller than that to let 3 items coexist with room to spare.
TOOL_SPACING = 0.10


# Maximum number of tools allowed on the belt at once (queue depth visible
# on the belt simultaneously). Limited by how many physical tool bodies
# exist (3) minus any currently being carried/sorted.
MAX_ON_BELT = 3

PARK = np.array([0.0, 0.0, -50.0])
GRASP_OFFSET_LOCAL = np.array([0.0, 0.0, 0.105])

TOOL_DEFS = [
    {"name": "spanner", "bin": "BIN_RED"},
    {"name": "bolt",    "bin": "BIN_BLUE"},
    {"name": "nut",     "bin": "BIN_BLUE"},
]

BIN_STACK = {"BIN_RED": 0, "BIN_BLUE": 0}

# =============================================================================
# ARM POSES
# =============================================================================

HOME_POSE   = [ 0.00,  0.00,  0.00, -1.57,  0.00,  1.57,  0.00]
PICKUP_POSE = [ 0.00,  0.50,  0.00, -1.80,  0.00,  2.00,  0.80]
LIFT_POSE   = [ 0.00, -0.20,  0.00, -1.50,  0.00,  1.60,  0.80]

BIN_POSES = {
    "BIN_RED":  [ 1.20,  0.50,  0.00, -1.80,  0.00,  2.00,  0.80],
    "BIN_BLUE": [-0.40,  0.50,  0.00, -1.80,  0.00,  2.00,  0.80],
}

GRIPPER_OPEN   = 200
GRIPPER_CLOSED = 0

# Arm-cycle timing (seconds) — time-based, never distance-gated, so the
# state machine can never silently stall.
T_ARM_TO_PICKUP = 2.5
T_GRIP_CLOSE    = 1.0
T_LIFT          = 1.5
T_SWING_TO_BIN  = 3.0
T_RETURN_HOME   = 2.0

# =============================================================================
# YOLOS
# =============================================================================

DETECT_CONF  = 0.30
DETECT_EVERY = 60

print("[INFO] Loading fine-tuned YOLOS ...")
try:
    processor   = AutoImageProcessor.from_pretrained(FINETUNED_PATH)
    yolos_model = AutoModelForObjectDetection.from_pretrained(FINETUNED_PATH)
    yolos_model.eval()
    print(f"[INFO] YOLOS ready — {list(yolos_model.config.id2label.values())}")
    YOLOS_ON = True
except Exception as e:
    print(f"[WARN] YOLOS not loaded ({e})")
    YOLOS_ON = False

RENDER_W, RENDER_H = 640, 480
_det_lock = threading.Lock()
_det_labels: list = []
_det_scores: list = []


def _run_detection(frame_rgb: np.ndarray) -> None:
    if not YOLOS_ON:
        return
    try:
        img    = Image.fromarray(frame_rgb)
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = yolos_model(**inputs)
        sizes   = torch.tensor([[img.size[1], img.size[0]]])
        results = processor.post_process_object_detection(
            outputs, threshold=DETECT_CONF, target_sizes=sizes
        )[0]
        labels = [yolos_model.config.id2label[int(c)] for c in results["labels"]]
        scores = [round(float(s), 2) for s in results["scores"]]
        with _det_lock:
            _det_labels.clear();  _det_labels.extend(labels)
            _det_scores.clear();  _det_scores.extend(scores)
        if labels:
            print(f"[YOLOS] {[f'{l}({s})' for l, s in zip(labels, scores)]}")
    except Exception as e:
        print(f"[YOLOS error] {e}")


# =============================================================================
# TOOL -> BIN MAPPING
# =============================================================================

_BIN_MAP = {"spanner": "BIN_RED", "bolt": "BIN_BLUE", "nut": "BIN_BLUE"}


def _identify_tool(name: str) -> str:
    result = _BIN_MAP[name]
    print(f"[TOOL]  {name:8s}  ->  {result}")
    return result


# =============================================================================
# KINEMATIC HELPERS — all must be called AFTER mj_step
# =============================================================================

def _force_free_body(mj_model, data: mujoco.MjData, qs: int,
                     pos: np.ndarray,
                     quat: np.ndarray = np.array([1., 0., 0., 0.])) -> None:
    data.qpos[qs:qs+3]   = pos
    data.qpos[qs+3]      = quat[0]
    data.qpos[qs+4:qs+7] = quat[1:]
    data.qvel[qs:qs+6]   = 0.0
    mujoco.mj_kinematics(mj_model, data)


def _lock_on_belt(mj_model, data, qs: int, x: float, tool_name: str) -> None:
    pos = np.array([x, PICKUP_Y, TOOL_REST_Z[tool_name]])
    _force_free_body(mj_model, data, qs, pos)


def _glue_to_hand(mj_model, data, qs: int, hand_id: int) -> None:
    hand_pos  = data.xpos[hand_id].copy()
    hand_xmat = data.xmat[hand_id].reshape(3, 3)
    hand_quat = data.xquat[hand_id].copy()
    tool_pos  = hand_pos + hand_xmat @ GRASP_OFFSET_LOCAL
    _force_free_body(mj_model, data, qs, tool_pos, hand_quat)


def _park(mj_model, data, qs: int) -> None:
    _force_free_body(mj_model, data, qs, PARK)


# =============================================================================
# ARM / GRIPPER CONTROL
# =============================================================================

def _set_arm(data: mujoco.MjData, pose: list) -> None:
    for i, v in enumerate(pose[:7]):
        data.ctrl[i] = v


def _set_gripper(data: mujoco.MjData, open_: bool) -> None:
    data.ctrl[7] = GRIPPER_OPEN if open_ else GRIPPER_CLOSED


# =============================================================================
# DROP
# =============================================================================

def _drop_in_bin(mj_model, data, qs: int, hand_id: int, bin_name: str) -> None:
    hand_pos = data.xpos[hand_id].copy()
    count    = BIN_STACK[bin_name]
    sx = [ 0.00,  0.03, -0.03,  0.02, -0.02,  0.00]
    sy = [ 0.00,  0.02, -0.02,  0.03, -0.03,  0.02]
    sz = [0.05 + i * 0.04 for i in range(10)]
    drop = np.array([
        hand_pos[0] + sx[count % len(sx)],
        hand_pos[1] + sy[count % len(sy)],
        sz[count % len(sz)],
    ])
    _force_free_body(mj_model, data, qs, drop)
    mujoco.mj_forward(mj_model, data)
    BIN_STACK[bin_name] += 1
    print(f"  [DROP] {bin_name}  drop=[{drop[0]:.3f},{drop[1]:.3f},{drop[2]:.3f}]")


# =============================================================================
# CALIBRATION HELPER
# =============================================================================

def _print_hand_positions(mj_model, data, hand_id: int) -> None:
    print("\n[CALIBRATION] Hand XYZ at each pose (note: mj_forward alone does")
    print("  not converge position actuators — these numbers are informational")
    print("  only, real motion happens via mj_step in the live loop):")
    for label, pose in [
        ("HOME     ", HOME_POSE),
        ("PICKUP   ", PICKUP_POSE),
        ("LIFT     ", LIFT_POSE),
        ("BIN_RED  ", BIN_POSES["BIN_RED"]),
        ("BIN_BLUE ", BIN_POSES["BIN_BLUE"]),
    ]:
        _set_arm(data, pose)
        mujoco.mj_forward(mj_model, data)
        hp = data.xpos[hand_id]
        print(f"  {label} -> [{hp[0]:.3f}, {hp[1]:.3f}, {hp[2]:.3f}]")
    _set_arm(data, HOME_POSE)
    mujoco.mj_forward(mj_model, data)
    print()


# =============================================================================
# BELT ITEM — one tool currently riding the belt or being processed by the arm
# =============================================================================

class BeltItem:
    """Tracks one physical tool body's journey: belt -> pickup -> arm -> bin."""

    __slots__ = ("name", "qs", "bin_name", "x", "on_belt", "arm_state",
                 "state_start")

    def __init__(self, name: str, qs: int, bin_name: str, x: float):
        self.name        = name
        self.qs          = qs
        self.bin_name    = bin_name
        self.x            = x        # current world-X while on belt
        self.on_belt      = True      # False once handed off to the arm
        self.arm_state    = None      # set when this item becomes the arm's target
        self.state_start  = 0.0


ARM_STATE_NAMES = {
    "TO_PICKUP":   "ARM_TO_PICKUP",
    "GRIP_CLOSE":  "GRIP_CLOSE",
    "LIFT":        "LIFT",
    "SWING_TO_BIN":"SWING_TO_BIN",
    "DROP":        "DROP",
    "RETURN_HOME": "RETURN_HOME",
}


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data     = mujoco.MjData(mj_model)

    for i in range(mj_model.neq):
        data.eq_active[i] = 1  # confirmed benign (link6<->link7 internal weld)

    belt_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "belt_slide")
    if belt_joint_id < 0:
        raise RuntimeError("Joint 'belt_slide' not found in XML.")
    belt_qposadr = mj_model.jnt_qposadr[belt_joint_id]

    print("\n[DEBUG] Tool joints & bodies:")
    qpos_by_name = {}
    for i in range(mj_model.njnt):
        jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        adr   = mj_model.jnt_qposadr[i]
        for td in TOOL_DEFS:
            if jname == f"{td['name']}_joint":
                qpos_by_name[td["name"]] = adr
                print(f"  {td['name']:8s} -> qpos_start={adr}")

    for td in TOOL_DEFS:
        if td["name"] not in qpos_by_name:
            raise RuntimeError(f"Joint '{td['name']}_joint' not found in XML.")

    hand_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if hand_id < 0:
        raise RuntimeError("Body 'hand' not found in model.")

    mujoco.mj_forward(mj_model, data)
    print("\n[INFO] Bin world positions:")
    for bname in ["bin_red", "bin_blue"]:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid >= 0:
            pos = data.xpos[bid]
            print(f"  {bname:12s} = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    _print_hand_positions(mj_model, data, hand_id)

    _set_arm(data, HOME_POSE)
    _set_gripper(data, open_=True)
    for td in TOOL_DEFS:
        _park(mj_model, data, qpos_by_name[td["name"]])
    mujoco.mj_forward(mj_model, data)

    print("[INFO] Continuous sorting plan:")
    print(f"  Belt speed = {BELT_SPEED_MPS} m/s, spacing = {TOOL_SPACING} m")
    print(f"  Spawn X={SPAWN_X}, pickup X={PICKUP_X}")
    print(f"  Tool rest Z: spanner={TOOL_REST_Z['spanner']:.3f} "
          f"bolt={TOOL_REST_Z['bolt']:.3f} nut={TOOL_REST_Z['nut']:.3f}\n")

    renderer = mujoco.Renderer(mj_model, height=RENDER_H, width=RENDER_W)

    # ------------------------------------------------------------------
    # Belt item bookkeeping
    # ------------------------------------------------------------------
    belt_items: list[BeltItem] = []     # items currently riding the belt, sorted far->near
    arm_item:   BeltItem | None = None  # item currently being handled by the arm
    free_tool_names: list = [td["name"] for td in TOOL_DEFS]  # bodies not in use

    sorted_counts = {"BIN_RED": 0, "BIN_BLUE": 0}
    step_counter  = 0
    last_time     = time.time()

    def _next_random_tool_name() -> str:
        return random.choice([td["name"] for td in TOOL_DEFS])

    def _try_spawn_new_item() -> None:
        """Spawn a new tool at SPAWN_X if there's room and a free body."""
        if len(belt_items) >= MAX_ON_BELT or not free_tool_names:
            return
        # Respect spacing: don't spawn on top of the item nearest the spawn end.
        # "Nearest to spawn" means smallest |x - SPAWN_X| in the direction of travel.
        if belt_items:
            if BELT_DIRECTION > 0:
                nearest_to_spawn_x = min(b.x for b in belt_items)
                gap = nearest_to_spawn_x - SPAWN_X
            else:
                nearest_to_spawn_x = max(b.x for b in belt_items)
                gap = SPAWN_X - nearest_to_spawn_x
            if gap < TOOL_SPACING:
                return
        name = free_tool_names.pop(0)
        qs   = qpos_by_name[name]
        bin_name = _identify_tool(name)
        item = BeltItem(name=name, qs=qs, bin_name=bin_name, x=SPAWN_X)
        _lock_on_belt(mj_model, data, qs, item.x, name)
        belt_items.append(item)
        print(f"[SPAWN] {name} entering belt at X={SPAWN_X:.2f} -> {bin_name}  "
              f"(belt_items now: {[b.name for b in belt_items]}, free: {free_tool_names})")

    def _enter_arm_state(item: BeltItem, new_state: str) -> None:
        item.arm_state   = new_state
        item.state_start = time.time()
        print(f"  -> {ARM_STATE_NAMES[new_state]}  ({item.name})")

    with mujoco.viewer.launch_passive(mj_model, data) as viewer:
        viewer.cam.distance  = 2.8
        viewer.cam.azimuth   = 130
        viewer.cam.elevation = -22
        viewer.cam.lookat[:] = [0.55, 0.0, 0.4]

        while viewer.is_running():
            now = time.time()
            dt  = now - last_time
            last_time = now

            # ==============================================================
            # 1. SET ARM/GRIPPER CONTROLS for this frame, based on arm_item
            # ==============================================================
            if arm_item is None:
                _set_arm(data, HOME_POSE)
                _set_gripper(data, open_=True)
            else:
                bin_pose = BIN_POSES[arm_item.bin_name]
                st = arm_item.arm_state
                if st == "TO_PICKUP":
                    _set_arm(data, PICKUP_POSE); _set_gripper(data, open_=True)
                elif st == "GRIP_CLOSE":
                    _set_arm(data, PICKUP_POSE); _set_gripper(data, open_=False)
                elif st == "LIFT":
                    _set_arm(data, LIFT_POSE);   _set_gripper(data, open_=False)
                elif st == "SWING_TO_BIN":
                    _set_arm(data, bin_pose);    _set_gripper(data, open_=False)
                elif st == "DROP":
                    _set_arm(data, bin_pose);    _set_gripper(data, open_=True)
                elif st == "RETURN_HOME":
                    _set_arm(data, HOME_POSE);   _set_gripper(data, open_=True)

            # Belt motor actuator left at zero — belt is driven kinematically.
            belt_actuator_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "belt_motor"
            )
            if belt_actuator_id >= 0:
                data.ctrl[belt_actuator_id] = 0.0

            # ==============================================================
            # 2. ADVANCE PHYSICS
            # ==============================================================
            mujoco.mj_step(mj_model, data)
            step_counter += 1

            # ==============================================================
            # 3. ADVANCE BELT ITEMS CONTINUOUSLY (kinematic, post-step)
            #    Every item not yet handed to the arm moves toward PICKUP_X
            #    at a constant speed, maintaining spacing naturally because
            #    they all move at the same rate and started spaced apart.
            # ==============================================================
            for item in belt_items:
                if item.on_belt:
                    item.x += BELT_DIRECTION * BELT_SPEED_MPS * dt
                    if BELT_DIRECTION > 0:
                        item.x = min(PICKUP_X, item.x)
                    else:
                        item.x = max(PICKUP_X, item.x)
                    _lock_on_belt(mj_model, data, item.qs, item.x, item.name)

            # ==============================================================
            # 4. HAND OFF THE FRONT ITEM TO THE ARM once it reaches pickup,
            #    but only if the arm is currently idle (arm_item is None).
            # ==============================================================
            if arm_item is None:
                for item in belt_items:
                    reached = (item.x >= PICKUP_X - 1e-6) if BELT_DIRECTION > 0 \
                              else (item.x <= PICKUP_X + 1e-6)
                    if item.on_belt and reached:
                        item.on_belt = False
                        arm_item = item
                        belt_items.remove(item)
                        _enter_arm_state(arm_item, "TO_PICKUP")
                        print(f"[HANDOFF] '{arm_item.name}' reached pickup, arm engaging")
                        break

            # ==============================================================
            # 5. KEEP THE ARM'S CURRENT ITEM LOCKED/GLUED appropriately
            # ==============================================================
            if arm_item is not None:
                if arm_item.arm_state in ("TO_PICKUP",):
                    _lock_on_belt(mj_model, data, arm_item.qs, PICKUP_X, arm_item.name)
                elif arm_item.arm_state in ("GRIP_CLOSE", "LIFT", "SWING_TO_BIN"):
                    _glue_to_hand(mj_model, data, arm_item.qs, hand_id)
                # DROP and RETURN_HOME handle their own positioning explicitly

            # ==============================================================
            # 6. SPAWN NEW ITEMS to keep the belt continuously fed
            # ==============================================================
            _try_spawn_new_item()

            # ==============================================================
            # 7. YOLOS DETECTION
            # ==============================================================
            if step_counter % DETECT_EVERY == 0:
                renderer.update_scene(data)
                frame = renderer.render().copy()
                threading.Thread(
                    target=_run_detection, args=(frame,), daemon=True
                ).start()

            if step_counter % 200 == 0:
                belt_str = ", ".join(f"{b.name}@{b.x:.3f}" for b in belt_items)
                arm_str  = f"{arm_item.name}:{arm_item.arm_state}" if arm_item else "idle"
                print(f"  [STATUS] belt=[{belt_str}]  arm={arm_str}")

            # ==============================================================
            # 8. ARM STATE MACHINE TRANSITIONS (time-based)
            # ==============================================================
            if arm_item is not None:
                elapsed = time.time() - arm_item.state_start
                st = arm_item.arm_state

                if st == "TO_PICKUP" and elapsed > T_ARM_TO_PICKUP:
                    _enter_arm_state(arm_item, "GRIP_CLOSE")

                elif st == "GRIP_CLOSE" and elapsed > T_GRIP_CLOSE:
                    print(f"[STATE] Gripped '{arm_item.name}'")
                    _enter_arm_state(arm_item, "LIFT")

                elif st == "LIFT" and elapsed > T_LIFT:
                    print(f"[STATE] Lifted '{arm_item.name}' — swinging to {arm_item.bin_name}")
                    _enter_arm_state(arm_item, "SWING_TO_BIN")

                elif st == "SWING_TO_BIN" and elapsed > T_SWING_TO_BIN:
                    hp = data.xpos[hand_id]
                    print(f"[STATE] At bin — hand=[{hp[0]:.3f},{hp[1]:.3f},{hp[2]:.3f}]")
                    _drop_in_bin(mj_model, data, arm_item.qs, hand_id, arm_item.bin_name)
                    sorted_counts[arm_item.bin_name] += 1
                    print(f"\n  Sorted: {arm_item.name:8s}  ->  {arm_item.bin_name}")
                    print(f"  +-------------------------------------------+")
                    print(f"  |  BIN_RED  (spanner):     {sorted_counts['BIN_RED']:2d} tool(s)      |")
                    print(f"  |  BIN_BLUE (bolt + nut):  {sorted_counts['BIN_BLUE']:2d} tool(s)      |")
                    print(f"  +-------------------------------------------+\n")
                    _enter_arm_state(arm_item, "RETURN_HOME")

                elif st == "RETURN_HOME" and elapsed > T_RETURN_HOME:
                    # Free this tool body for re-spawning, arm goes idle
                    free_tool_names.append(arm_item.name)
                    arm_item = None

            # ==============================================================
            # 9. SYNC VIEWER
            # ==============================================================
            viewer.sync()


if __name__ == "__main__":
    main()