import os
import time
import numpy as np
import mujoco
import mujoco.viewer


ROOT_DIR = r"C:\Users\rad\itmo\new_roms"


# XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_block_touch_sensors.xml")
XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_pen_touch_sensors.xml")
# XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_egg_touch_sensors.xml")
# XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "hand_manipulate_clock.xml")

NUM_TRIALS  = 5
OPEN_STEPS  = 40
CLOSE_STEPS = 120
HOLD_STEPS  = 200


TH_TOUCH       = 0.01   # per-finger touch "active" threshold
TAU_TIP_SUM    = 0.05   # minimal total fingertip signal for "real contact"
MIN_TIP_ACTIVE = 2      # at least N fingers active

CGF_STREAK_MIN = 10     # CGF streak needed for "stable" flag
GD_SUCCESS_MIN = 50     # max CGF streak needed for "success"

# Viewer sync sleep
SLEEP_OPEN  = 0.002
SLEEP_CLOSE = 0.002
SLEEP_HOLD  = 0.004


BASE_SEED = 12345
XY_SIGMA = 0.010            
YAW_RANGE = np.deg2rad(25)  
Z_OFFSET_RANGE = (0.0, 0.0) 


FINGER_ACTUATORS = list(range(2, 20))

FINGERTIP_TOUCH_CANON = {
    "FF": "robot0:ST_Tch_fftip",
    "MF": "robot0:ST_Tch_mftip",
    "RF": "robot0:ST_Tch_rftip",
    "LF": "robot0:ST_Tch_lftip",
    "TH": "robot0:ST_Tch_thtip",
}
FINGERS = ["FF", "MF", "RF", "LF", "TH"]


HAND_BODY_KEYWORDS = [
    "robot0", "hand", "palm", "finger", "thumb",
    "ff", "mf", "rf", "lf", "th",
    "wrist", "forearm"
]

print("XML exists:", os.path.exists(XML_PATH))
print("XML:", XML_PATH)



model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

DT = float(model.opt.timestep)
print("\nScene loaded successfully")
print(f"Bodies={model.nbody} Joints={model.njnt} Actuators={model.nu} Sensors={model.nsensor} timestep={DT}")



# SENSOR SLICES 
sensor_slices = {}
off = 0
for i in range(model.nsensor):
    s = model.sensor(i)
    dim = int(np.asarray(s.dim).item())
    stype = int(np.asarray(s.type).item())
    sensor_slices[s.name] = (off, dim, stype)
    off += dim

def sensor_sum_by_name(name: str) -> float:
    if name not in sensor_slices:
        return 0.0
    st, dim, _ = sensor_slices[name]
    return float(np.sum(data.sensordata[st:st+dim]))

def is_touch_sensor(i: int) -> bool:
    return int(model.sensor(i).type) == int(mujoco.mjtSensor.mjSENS_TOUCH)

touch_sensor_names = [model.sensor(i).name for i in range(model.nsensor) if is_touch_sensor(i)]
print("Touch sensors count:", len(touch_sensor_names))

# Heuristic palm sensors 
PALM_SENSOR_NAMES = [
    n for n in sensor_slices.keys()
    if ("ts_palm" in (n or "").lower()) or ("palm" in (n or "").lower())
]



# OBJECT BODY FINDING (robust)

def find_object_body_id(model) -> int:
    # Prefer exact "object"
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if bid >= 0:
        return int(bid)

    # Otherwise try common substrings
    keys = ["object", "clock", "block", "pen", "egg", "target", "obj"]
    for b in range(model.nbody):
        name = model.body(b).name or ""
        low = name.lower()
        if any(k in low for k in keys):
            return int(b)

    return -1

OBJECT_BODY_ID = find_object_body_id(model)
if OBJECT_BODY_ID < 0:
    print("[WARN] Object body not found. Contacts/drift/speed will be 0/NaN.")
else:
    print("Object body:", model.body(OBJECT_BODY_ID).name, "| id=", OBJECT_BODY_ID)



# Find object's FREE joint qpos address 

def find_free_joint_qposadr_for_body(model, body_id: int):
    
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) != int(body_id):
            continue
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            adr = int(model.jnt_qposadr[j])
            return j, adr
    return None, None

OBJ_FREE_JID, OBJ_FREE_QPOSADR = (None, None)
if OBJECT_BODY_ID >= 0:
    OBJ_FREE_JID, OBJ_FREE_QPOSADR = find_free_joint_qposadr_for_body(model, OBJECT_BODY_ID)
    if OBJ_FREE_QPOSADR is not None:
        jname = model.joint(OBJ_FREE_JID).name
        print("Object FREE joint:", jname, "| qposadr=", OBJ_FREE_QPOSADR)
    else:
        print("[WARN] Object FREE joint not found -> trials may still be identical (if object pose fixed).")


def randomize_object_pose(seed: int):
   
    if OBJ_FREE_QPOSADR is None:
        return
    rng = np.random.default_rng(seed)

    qpos = data.qpos
    pos = qpos[OBJ_FREE_QPOSADR:OBJ_FREE_QPOSADR+3].copy()
    quat = qpos[OBJ_FREE_QPOSADR+3:OBJ_FREE_QPOSADR+7].copy()

    dx, dy = rng.normal(0.0, XY_SIGMA, size=2)
    pos[0] += float(dx)
    pos[1] += float(dy)

    if Z_OFFSET_RANGE is not None and (Z_OFFSET_RANGE[0] != 0.0 or Z_OFFSET_RANGE[1] != 0.0):
        pos[2] += float(rng.uniform(Z_OFFSET_RANGE[0], Z_OFFSET_RANGE[1]))

    yaw = float(rng.uniform(-YAW_RANGE, YAW_RANGE))
    q_yaw = np.zeros(4, dtype=float)
    mujoco.mju_axisAngle2Quat(q_yaw, np.array([0.0, 0.0, 1.0], dtype=float), yaw)

    q_new = np.zeros(4, dtype=float)
    mujoco.mju_mulQuat(q_new, q_yaw, quat)

    qpos[OBJ_FREE_QPOSADR:OBJ_FREE_QPOSADR+3] = pos
    qpos[OBJ_FREE_QPOSADR+3:OBJ_FREE_QPOSADR+7] = q_new

    mujoco.mj_forward(model, data)



# Fingertip touch reading

CANON_PRESENT = {f: (FINGERTIP_TOUCH_CANON[f] in sensor_slices) for f in FINGERS}
use_canon = all(CANON_PRESENT.values())
print("Canonical fingertip sensors present:", CANON_PRESENT, "| use_canon =", use_canon)


finger_touch_sensor_ids = {f: [] for f in FINGERS}
if not use_canon:
    # Map finger -> substrings in sensor name
    finger_sub = {
        "FF": ["ff", "fftip", "ff_tip"],
        "MF": ["mf", "mftip", "mf_tip"],
        "RF": ["rf", "rftip", "rf_tip"],
        "LF": ["lf", "lftip", "lf_tip"],
        "TH": ["th", "thtip", "th_tip", "thumb"],
    }
    for si in range(model.nsensor):
        if not is_touch_sensor(si):
            continue
        sname = (model.sensor(si).name or "").lower()
        for f in FINGERS:
            if any(sub in sname for sub in finger_sub[f]):
                finger_touch_sensor_ids[f].append(si)

    print("Fallback touch ids per finger:", {f: len(finger_touch_sensor_ids[f]) for f in FINGERS})

def read_touch_by_finger() -> dict:
    out = {f: 0.0 for f in FINGERS}
    if use_canon:
        for f in FINGERS:
            out[f] = sensor_sum_by_name(FINGERTIP_TOUCH_CANON[f])
        return out

    
    for f in FINGERS:
        s = 0.0
        for sid in finger_touch_sensor_ids[f]:
            
            adr = int(np.asarray(model.sensor(sid).adr).item())
            dim = int(np.asarray(model.sensor(sid).dim).item())
            s += float(np.sum(data.sensordata[adr:adr+dim]))
        out[f] = s
    return out



# Contact helpers 

def body_name(body_id: int) -> str:
    return (model.body(body_id).name or "")

def is_hand_body(body_name_str: str) -> bool:
    bn = (body_name_str or "").lower()
    return any(k in bn for k in HAND_BODY_KEYWORDS)

def get_object_contacts_split_by_body(obj_body_id: int):
    """
    Count contacts where one geom belongs to the object body.
    Split into hand vs env by other geom's body name.
    """
    if obj_body_id < 0:
        return 0, 0, 0

    con_total = con_hand = con_env = 0
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])

        # object involved?
        if not ((b1 == obj_body_id) or (b2 == obj_body_id)):
            continue

        con_total += 1
        other_body_id = b2 if (b1 == obj_body_id) else b1
        other_body_name = body_name(other_body_id)

        if is_hand_body(other_body_name):
            con_hand += 1
        else:
            con_env += 1

    return con_total, con_hand, con_env


def object_pos_world() -> np.ndarray:
    if OBJECT_BODY_ID < 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return data.xpos[OBJECT_BODY_ID].copy()

def object_speed_world() -> float:
    if OBJECT_BODY_ID < 0:
        return float("nan")
    v = data.cvel[OBJECT_BODY_ID][3:6]
    return float(np.linalg.norm(v))


def contact_entropy(touch_by_finger: dict, eps: float = 1e-12) -> float:
    vec = np.array([max(0.0, float(touch_by_finger[f])) for f in FINGERS], dtype=float)
    s = float(np.sum(vec))
    if s <= eps:
        return 0.0
    p = vec / s
    return float(-np.sum(p * np.log(p + eps)))


def reset_simulation(seed: int, viewer):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    randomize_object_pose(seed)  # makes trials different
    viewer.sync()
    time.sleep(0.15)


viewer = mujoco.viewer.launch_passive(model, data)


print(" BASE EVALUATION (robust metrics + randomized trials)")


for trial in range(NUM_TRIALS):
    trial_seed = BASE_SEED + 1000 * trial

   
    print(f" TRIAL {trial + 1}/{NUM_TRIALS} (seed={trial_seed})")
  

    reset_simulation(trial_seed, viewer)

    # OPEN HAND
    for _ in range(OPEN_STEPS):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(SLEEP_OPEN)

    # CLOSE FINGERS
    for _ in range(CLOSE_STEPS):
        data.ctrl[:] = 0.0
        for idx in FINGER_ACTUATORS:
            data.ctrl[idx] = 1.0
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(SLEEP_CLOSE)

    # Reference pose at start of HOLD
    p0 = object_pos_world()

    # HOLD accumulators
    cgf_streak = 0
    gd_max = 0

    grasp_drifts = []
    grasp_speeds = []
    grasp_tip_sums = []
    grasp_entropies = []

    hold_tip_sums = []
    hold_entropies = []

    hold_con_total = 0
    hold_con_hand  = 0
    hold_con_env   = 0

    # HOLD loop
    for _ in range(HOLD_STEPS):
        for idx in FINGER_ACTUATORS:
            data.ctrl[idx] = 1.0

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(SLEEP_HOLD)

        con_total, con_hand, con_env = get_object_contacts_split_by_body(OBJECT_BODY_ID)
        hold_con_total += con_total
        hold_con_hand  += con_hand
        hold_con_env   += con_env

        touch_f = read_touch_by_finger()
        tip_sum = float(sum(touch_f.values()))
        tip_active = int(sum(1 for f in FINGERS if touch_f[f] > TH_TOUCH))
        ce = contact_entropy(touch_f)

        hold_tip_sums.append(tip_sum)
        hold_entropies.append(ce)

        cgf_now = (tip_active >= MIN_TIP_ACTIVE) and (tip_sum >= TAU_TIP_SUM) and (con_hand > 0) and (con_env == 0)

        cgf_streak = (cgf_streak + 1) if cgf_now else 0
        gd_max = max(gd_max, cgf_streak)

        if cgf_now and np.all(np.isfinite(p0)):
            p = object_pos_world()
            drift = float(np.linalg.norm(p - p0)) if np.all(np.isfinite(p)) else float("nan")
            spd = object_speed_world()

            if np.isfinite(drift):
                grasp_drifts.append(drift)
            if np.isfinite(spd):
                grasp_speeds.append(spd)
            grasp_tip_sums.append(tip_sum)
            grasp_entropies.append(ce)

    # Summary 
    edr = (hold_con_env / hold_con_total) if hold_con_total > 0 else 0.0
    cgf_stable = int(gd_max >= CGF_STREAK_MIN)
    success = int(gd_max >= GD_SUCCESS_MIN)

    if len(grasp_drifts) > 0:
        drift_p95 = float(np.percentile(grasp_drifts, 95))
        speed_p95 = float(np.percentile(grasp_speeds, 95)) if len(grasp_speeds) > 0 else 0.0
        tip_sum_avg_grasp = float(np.mean(grasp_tip_sums)) if len(grasp_tip_sums) > 0 else 0.0
        ce_avg_grasp = float(np.mean(grasp_entropies)) if len(grasp_entropies) > 0 else 0.0
    else:
        drift_p95 = 0.0
        speed_p95 = 0.0
        tip_sum_avg_grasp = 0.0
        ce_avg_grasp = 0.0

    tip_sum_avg_hold = float(np.mean(hold_tip_sums)) if hold_tip_sums else 0.0
    ce_avg_hold = float(np.mean(hold_entropies)) if hold_entropies else 0.0

 
    contacts_total_per_step = (hold_con_total / HOLD_STEPS) if HOLD_STEPS > 0 else 0.0
    contacts_hand_per_step  = (hold_con_hand  / HOLD_STEPS) if HOLD_STEPS > 0 else 0.0
    contacts_env_per_step   = (hold_con_env   / HOLD_STEPS) if HOLD_STEPS > 0 else 0.0

    print(
        f"SUCCESS={success} CGF_stable={cgf_stable} GD={gd_max} "
        f"EDR={edr:.3f} "
        f"DriftP95@Grasp={drift_p95:.4f} SpeedP95@Grasp={speed_p95:.4f} "
        f"AvgTipSum@Grasp={tip_sum_avg_grasp:.4f} CE@Grasp={ce_avg_grasp:.4f} "
        f"| contacts_hand={contacts_hand_per_step:.2f} contacts_env={contacts_env_per_step:.2f} contacts_total={contacts_total_per_step:.2f} "
        f"| AvgTipSum@Hold={tip_sum_avg_hold:.4f} CE@Hold={ce_avg_hold:.4f}"
    )

print("\nAll trials finished.")
print("MuJoCo window will stay open.")
print("Close the viewer window manually to exit.")

while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.02)
