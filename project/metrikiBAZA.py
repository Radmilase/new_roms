# ============================================================
# BASE: Shadow Dexterous Hand
# + UPDATED METRICS (same as V3)
# - contacts_hand / contacts_env
# - drift_p95 / speed_p95 after SETTLE_STEPS
# - bad_drift_ratio / bad_speed_ratio
# - tip_sum_hold_avg / tip_active_hold_avg
# - tip_streak_max / thumb_opposition
# - success = stabilized & hand_only & real_fingertip_grasp
# ============================================================

import os
import time
import numpy as np
import mujoco
import mujoco.viewer


# -------------------------------
# CONFIG
# -------------------------------
ROOT_DIR = r"C:\Users\rad\itmo\new_roms"

# Поставь нужный XML (clock / block / pen / egg) как в твоих экспериментах
XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "hand_manipulate_clock.xml")
# XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_block_touch_sensors.xml")
# XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_pen_touch_sensors.xml")
# XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_egg_touch_sensors.xml")

NUM_TRIALS  = 5
OPEN_STEPS  = 40
CLOSE_STEPS = 120
HOLD_STEPS  = 200

# time slowdown for render
STEP_SLEEP_OPEN  = 0.005
STEP_SLEEP_CLOSE = 0.005
STEP_SLEEP_HOLD  = 0.010

# -------------------------------
# SUCCESS / METRICS THRESHOLDS (same idea as V3)
# -------------------------------
Z_MIN      = 0.05
V_MAX      = 2.0
DRIFT_MAX  = 0.25
SUCCESS_HOLD_RATIO = 0.90

SETTLE_STEPS = 40  # ignore first steps in HOLD for robust p95

MIN_TIP_ACTIVE_HOLD  = 2
MIN_TIP_SUM_HOLD_AVG = 0.25
TIP_CONTACT_THR      = 0.05
MIN_TIP_STREAK       = 15

MIN_CONTACTS_HAND = 3
MAX_CONTACTS_ENV  = 0  # if you want to allow tiny env contacts: set to 1


# -------------------------------
# LOAD MUJOCO MODEL
# -------------------------------
print("XML exists:", os.path.exists(XML_PATH))
print("XML:", XML_PATH)
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

print("\nScene loaded successfully")
print(f"Bodies={model.nbody} Joints={model.njnt} Actuators={model.nu} Sensors={model.nsensor} timestep={model.opt.timestep}")

# -------------------------------
# SENSOR SLICES (for touch sums)
# -------------------------------
sensor_slices = {}
off = 0
for i in range(model.nsensor):
    s = model.sensor(i)
    dim = int(np.asarray(s.dim).item())
    sensor_slices[s.name] = (off, dim, int(np.asarray(s.type).item()))
    off += dim

def sensor_sum(name: str) -> float:
    if name not in sensor_slices:
        return 0.0
    st, dim, _ = sensor_slices[name]
    return float(np.sum(data.sensordata[st:st + dim]))

FINGERS = ["FF", "MF", "RF", "LF", "TH"]
FINGERTIP_TOUCH = {
    "FF": "robot0:ST_Tch_fftip",
    "MF": "robot0:ST_Tch_mftip",
    "RF": "robot0:ST_Tch_rftip",
    "LF": "robot0:ST_Tch_lftip",
    "TH": "robot0:ST_Tch_thtip",
}
touch_present = {f: (FINGERTIP_TOUCH[f] in sensor_slices) for f in FINGERS}
print("Fingertip touch present:", touch_present)

PALM_SENSOR_NAMES = [n for n in sensor_slices.keys() if ("ts_palm" in n.lower()) or ("palm" in n.lower())]

# -------------------------------
# ACTUATORS (base control)
# -------------------------------
# В твоём base-скрипте пальцы = 2..19 (0–1 — запястье)
FINGER_ACTUATORS = list(range(2, 20))


# -------------------------------
# FIND OBJECT BODY (robust)
# -------------------------------
def find_object_body_id():
    # try common names first
    candidates = ["object", "clock", "block", "pen", "egg"]
    for name in candidates:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            return bid, name

    # fallback: find a non-hand body that has a free joint
    # heuristic: body name not containing robot0, and has a joint of type FREE
    for bid in range(model.nbody):
        bname = model.body(bid).name or ""
        if ("robot0" in bname) or ("hand" in bname.lower()) or ("forearm" in bname.lower()) or ("wrist" in bname.lower()):
            continue
        # check if this body has any joint and if that joint is FREE
        jadr = int(model.body_jntadr[bid])
        jnum = int(model.body_jntnum[bid])
        for j in range(jadr, jadr + jnum):
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return bid, bname

    raise RuntimeError("Could not infer object body id (tried object/clock/block/pen/egg + free-joint fallback).")

obj_bid, obj_body_name = find_object_body_id()
print("Object body inferred:", obj_body_name, "| id:", obj_bid)

# -------------------------------
# CONTACT SPLIT: hand vs env
# -------------------------------
def is_hand_body(bid: int) -> bool:
    name = model.body(bid).name or ""
    if ("robot0" in name) or name.startswith("robot0:"):
        return True
    if ("forearm" in name.lower()) or ("hand" in name.lower()) or ("wrist" in name.lower()):
        return True
    return False

def count_object_contacts_split():
    hand = 0
    env = 0
    total = 0
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[int(c.geom1)])
        b2 = int(model.geom_bodyid[int(c.geom2)])

        if b1 == obj_bid and b2 != obj_bid:
            other = b2
        elif b2 == obj_bid and b1 != obj_bid:
            other = b1
        else:
            continue

        total += 1
        if is_hand_body(other):
            hand += 1
        else:
            env += 1
    return hand, env, total


# -------------------------------
# RESET
# -------------------------------
def reset_simulation(viewer):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(1.0)


# -------------------------------
# ONE TRIAL (base control + NEW metrics)
# -------------------------------
def run_trial_base(viewer, do_render=True):
    # reset
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    latched_dummy = None  # not used in base
    energy = 0.0
    smooth = 0.0
    prev_u = np.zeros(model.nu, dtype=float)

    # OPEN
    for _ in range(OPEN_STEPS):
        u = np.zeros(model.nu, dtype=float)
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        if do_render:
            viewer.sync()
            time.sleep(STEP_SLEEP_OPEN)

    # CLOSE (fast)
    last_touch = {f: 0.0 for f in FINGERS}
    for _ in range(CLOSE_STEPS):
        u = np.zeros(model.nu, dtype=float)
        for idx in FINGER_ACTUATORS:
            u[idx] = 1.0
        data.ctrl[:] = u

        # sample touch (end of close)
        for f in FINGERS:
            last_touch[f] = sensor_sum(FINGERTIP_TOUCH[f]) if touch_present[f] else 0.0

        energy += float(np.sum(u * u))
        smooth += float(np.sum((u - prev_u) ** 2))
        prev_u = u

        mujoco.mj_step(model, data)
        if do_render:
            viewer.sync()
            time.sleep(STEP_SLEEP_CLOSE)

    contacts_hand, contacts_env, contacts_total = count_object_contacts_split()
    obj_pos = data.xpos[obj_bid].copy()
    z = float(obj_pos[2])

    tip_vals = np.array([last_touch[f] for f in FINGERS], dtype=float)
    tip_sum = float(np.sum(tip_vals))
    tip_active = int(np.sum(tip_vals > 0.0))
    palm_sum = float(np.sum([sensor_sum(n) for n in PALM_SENSOR_NAMES])) if len(PALM_SENSOR_NAMES) else 0.0

    # HOLD (new robust metrics)
    pos_ref = obj_pos.copy()
    stable_steps = 0

    speeds = []
    drifts = []
    bad_speed_steps = 0
    bad_drift_steps = 0

    tip_sum_hold = []
    tip_active_hold = []
    streak = 0
    tip_streak_max = 0
    opposition_seen = False

    for _ in range(HOLD_STEPS):
        u = np.zeros(model.nu, dtype=float)
        for idx in FINGER_ACTUATORS:
            u[idx] = 1.0
        data.ctrl[:] = u

        energy += float(np.sum(u * u))
        smooth += float(np.sum((u - prev_u) ** 2))
        prev_u = u

        mujoco.mj_step(model, data)

        p_now = data.xpos[obj_bid]
        v_now = data.cvel[obj_bid][3:6]
        speed = float(np.linalg.norm(v_now))
        drift = float(np.linalg.norm(p_now - pos_ref))

        speeds.append(speed)
        drifts.append(drift)

        is_stable = (p_now[2] > Z_MIN) and (speed < V_MAX) and (drift < DRIFT_MAX)
        if is_stable:
            stable_steps += 1

        if speed >= V_MAX:
            bad_speed_steps += 1
        if drift >= DRIFT_MAX:
            bad_drift_steps += 1

        # fingertip touch in HOLD
        touch_now = {f: (sensor_sum(FINGERTIP_TOUCH[f]) if touch_present[f] else 0.0) for f in FINGERS}
        tv = np.array([touch_now[f] for f in FINGERS], dtype=float)

        ts = float(np.sum(tv))
        ta = int(np.sum(tv > 0.0))
        tip_sum_hold.append(ts)
        tip_active_hold.append(ta)

        if ts > TIP_CONTACT_THR:
            streak += 1
            tip_streak_max = max(tip_streak_max, streak)
        else:
            streak = 0

        if (touch_now["TH"] > 0.0) and (
            (touch_now["FF"] > 0.0) or (touch_now["MF"] > 0.0) or
            (touch_now["RF"] > 0.0) or (touch_now["LF"] > 0.0)
        ):
            opposition_seen = True

        if do_render:
            viewer.sync()
            time.sleep(STEP_SLEEP_HOLD)

    hold_ratio = stable_steps / float(HOLD_STEPS)

    speeds = np.array(speeds, dtype=float)
    drifts = np.array(drifts, dtype=float)

    start = min(SETTLE_STEPS, max(0, HOLD_STEPS - 1))
    speeds2 = speeds[start:] if len(speeds) else np.array([0.0], dtype=float)
    drifts2 = drifts[start:] if len(drifts) else np.array([0.0], dtype=float)

    speed_p95 = float(np.percentile(speeds2, 95))
    drift_p95 = float(np.percentile(drifts2, 95))

    max_speed = float(np.max(speeds)) if len(speeds) else 0.0
    max_drift = float(np.max(drifts)) if len(drifts) else 0.0

    bad_speed_ratio = bad_speed_steps / float(HOLD_STEPS)
    bad_drift_ratio = bad_drift_steps / float(HOLD_STEPS)

    tip_sum_hold_avg = float(np.mean(tip_sum_hold)) if len(tip_sum_hold) else 0.0
    tip_active_hold_avg = float(np.mean(tip_active_hold)) if len(tip_active_hold) else 0.0
    thumb_opposition = bool(opposition_seen)

    # NEW success definition (same logic as V3)
    hand_only = (contacts_env <= MAX_CONTACTS_ENV) and (contacts_hand >= MIN_CONTACTS_HAND)
    real_fingertip_grasp = (
        (tip_sum_hold_avg >= MIN_TIP_SUM_HOLD_AVG) and
        (tip_active_hold_avg >= MIN_TIP_ACTIVE_HOLD) and
        (tip_streak_max >= MIN_TIP_STREAK) and
        thumb_opposition
    )
    stabilized = (hold_ratio >= SUCCESS_HOLD_RATIO)
    success = 1 if (stabilized and hand_only and real_fingertip_grasp) else 0

    return {
        # legacy
        "success": int(success),
        "hold_ratio": float(hold_ratio),
        "contacts": int(contacts_total),
        "contacts_hand": int(contacts_hand),
        "contacts_env": int(contacts_env),
        "z": float(z),
        "max_drift": float(max_drift),
        "max_speed": float(max_speed),
        "tip_active": int(tip_active),
        "tip_sum": float(tip_sum),
        "palm_sum": float(palm_sum),
        "energy": float(energy),
        "smooth": float(smooth),
        "touch_per_finger": {f: float(last_touch[f]) for f in FINGERS},

        # new
        "drift_p95": float(drift_p95),
        "speed_p95": float(speed_p95),
        "bad_drift_ratio": float(bad_drift_ratio),
        "bad_speed_ratio": float(bad_speed_ratio),
        "tip_sum_hold_avg": float(tip_sum_hold_avg),
        "tip_active_hold_avg": float(tip_active_hold_avg),
        "tip_streak_max": int(tip_streak_max),
        "thumb_opposition": int(thumb_opposition),
    }


# -------------------------------
# RUN
# -------------------------------
viewer = mujoco.viewer.launch_passive(model, data)

all_metrics = []

for trial in range(NUM_TRIALS):
    print("\n==============================")
    print(f" TRIAL {trial + 1}/{NUM_TRIALS}")
    print("==============================")

    reset_simulation(viewer)

    m = run_trial_base(viewer, do_render=True)
    all_metrics.append(m)

    tp = m["touch_per_finger"]
    print(
        f"SUCCESS={m['success']} hold={m['hold_ratio']:.2f} "
        f"con_hand={m['contacts_hand']} con_env={m['contacts_env']} con={m['contacts']} "
        f"z={m['z']:.3f} drift={m['max_drift']:.3f} speed={m['max_speed']:.3f} "
        f"driftP95={m['drift_p95']:.3f} speedP95={m['speed_p95']:.3f} "
        f"bad_drift={m['bad_drift_ratio']:.2f} bad_speed={m['bad_speed_ratio']:.2f} "
        f"tip_active={m['tip_active']} tip_sum={m['tip_sum']:.3f} palm_sum={m['palm_sum']:.3f} "
        f"hold_tip_avg={m['tip_sum_hold_avg']:.3f} streak={m['tip_streak_max']} opp={m['thumb_opposition']} "
        f"energy={m['energy']:.1f} smooth={m['smooth']:.1f} "
        f"FF={tp['FF']:.3f} MF={tp['MF']:.3f} RF={tp['RF']:.3f} LF={tp['LF']:.3f} TH={tp['TH']:.3f}"
    )

print("\n==============================")
print("SUMMARY (all trials)")
print("==============================")

succ_rate = float(np.mean([m["success"] for m in all_metrics]))
avg_contacts = float(np.mean([m["contacts"] for m in all_metrics]))
avg_contacts_hand = float(np.mean([m["contacts_hand"] for m in all_metrics]))
avg_contacts_env = float(np.mean([m["contacts_env"] for m in all_metrics]))
avg_hold = float(np.mean([m["hold_ratio"] for m in all_metrics]))
avg_max_drift = float(np.mean([m["max_drift"] for m in all_metrics]))
avg_drift_p95 = float(np.mean([m["drift_p95"] for m in all_metrics]))
avg_speed_p95 = float(np.mean([m["speed_p95"] for m in all_metrics]))
avg_tip_hold = float(np.mean([m["tip_sum_hold_avg"] for m in all_metrics]))
avg_streak = float(np.mean([m["tip_streak_max"] for m in all_metrics]))
opp_rate = float(np.mean([m["thumb_opposition"] for m in all_metrics]))

print(f"Success rate: {succ_rate:.2f}")
print(f"Avg contacts (total): {avg_contacts:.2f}")
print(f"Avg contacts_hand: {avg_contacts_hand:.2f} | Avg contacts_env: {avg_contacts_env:.2f}")
print(f"Avg hold_ratio: {avg_hold:.2f}")
print(f"Avg max_drift: {avg_max_drift:.3f}")
print(f"Avg driftP95: {avg_drift_p95:.3f} | Avg speedP95: {avg_speed_p95:.3f}")
print(f"Avg hold tip_sum: {avg_tip_hold:.3f} | Avg tip_streak_max: {avg_streak:.1f} | Opposition rate: {opp_rate:.2f}")

print("\nAll trials finished.")
print("MuJoCo window will stay open.")
print("Close the viewer window manually to exit.")

while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.02)
