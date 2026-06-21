import mujoco
model = mujoco.MjModel.from_xml_path(r"C:\Users\srush\Downloads\mujoco_menagerie-main\mujoco_menagerie-main\franka_emika_panda\scene.xml")
for i in range(model.nbody):
    print(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))