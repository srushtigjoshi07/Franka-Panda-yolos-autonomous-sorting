import mujoco

MODEL_PATH = (
    r"C:\Users\srush\Downloads\mujoco_menagerie-main"
    r"\mujoco_menagerie-main\franka_emika_panda\manufacturing_scene.xml"
)

mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(mj_model)
mujoco.mj_forward(mj_model, data)

print(f"neq = {mj_model.neq}\n")

for i in range(mj_model.neq):
    etype   = mj_model.eq_type[i]
    obj1id  = mj_model.eq_obj1id[i]
    obj2id  = mj_model.eq_obj2id[i]
    active0 = mj_model.eq_active0[i]
    data_   = mj_model.eq_data[i]

    # eq_obj1id/obj2id refer to body ids when eq_type == mjEQ_WELD (2) or mjEQ_CONNECT (1)
    name1 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, obj1id)
    name2 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, obj2id)

    print(f"[{i}] type={etype} (2=WELD, 1=CONNECT, 0=CONSTRAINT/JOINT)")
    print(f"      obj1id={obj1id} ({name1!r})   obj2id={obj2id} ({name2!r})")
    print(f"      eq_active0(default active)={active0}")
    print(f"      eq_data={data_}")
    print(f"      currently active in data.eq_active: {data.eq_active[i] if hasattr(data,'eq_active') else 'n/a'}")
    print()

print("If this weld connects 'hand' (or a finger body) to ANY tool or fixed")
print("body, it will fight your ctrl torques and can freeze the arm once")
print("contact/proximity triggers it, or if it's active from t=0.")