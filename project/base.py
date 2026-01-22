# # ============================================================
# # Shadow Dexterous Hand + Crosley Alarm Clock
# # 5 grasp trials, fixed camera, no debug prints
# # ============================================================

# import os
# import time
# import mujoco
# import mujoco.viewer
# import gymnasium_robotics
# import numpy as np

# # -------------------------------
# # CONFIG
# # -------------------------------


# ROOT_DIR = r"C:\Users\rad\itmo\new_roms"
# XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "hand_manipulate_clock.xml")


# # ASSETS_DIR = os.path.join(
# #     os.path.dirname(gymnasium_robotics.__file__),
# #     "envs", "assets", "hand"
# # )

# # XML_PATH = os.path.join(
# #     ASSETS_DIR,
# #     "hand_manipulate_clock.xml"
# # )

# NUM_TRIALS  = 5
# OPEN_STEPS  = 40
# CLOSE_STEPS = 120
# HOLD_STEPS  = 200

# # print("Assets dir:", ASSETS_DIR)
# # print("Clock scene XML exists:", os.path.exists(XML_PATH))

# print("XML exists:", os.path.exists(XML_PATH))
# print("XML:", XML_PATH)

# # -------------------------------
# # LOAD MUJOCO MODEL
# # -------------------------------

# model = mujoco.MjModel.from_xml_path(XML_PATH)
# data  = mujoco.MjData(model)

# print("\nScene loaded successfully")
# print("Bodies:", model.nbody)
# print("Joints:", model.njnt)
# print("Actuators:", model.nu)

# # -------------------------------
# #ТУТ СЕНСОРЫ ПРОВЕРЯЮТСЯ НА НАЛИЧИЕ
# touch_sensors = [
#     model.sensor(i).name
#     for i in range(model.nsensor)
#     if model.sensor(i).type == mujoco.mjtSensor.mjSENS_TOUCH
# ]

# print("Touch sensors:", touch_sensors)
# # -------------------------------

# print("\n=== ACTUATORS ===")
# for i in range(model.nu):
#     print(i, model.actuator(i).name)

# # пальцы: 2..19 (0–1 — запястье)
# FINGER_ACTUATORS = list(range(2, 20))

# # -------------------------------
# # HELPERS
# # -------------------------------

# def get_contacts(model, data, object_keyword="clock"):
#     contacts = []
#     for i in range(data.ncon):
#         c = data.contact[i]
#         g1 = model.geom(c.geom1).name
#         g2 = model.geom(c.geom2).name
#         if object_keyword in g1 or object_keyword in g2:
#             contacts.append((g1, g2))
#     return contacts


# def is_grasping(model, data, min_contacts=2):
#     return len(get_contacts(model, data)) >= min_contacts


# def reset_simulation(model, data, viewer):
#     mujoco.mj_resetData(model, data)
#     mujoco.mj_forward(model, data)
#     viewer.sync()
#     time.sleep(2)

# # -------------------------------
# # LAUNCH VIEWER
# # -------------------------------

# viewer = mujoco.viewer.launch_passive(model, data)

# # -------------------------------
# # GRASP TRIALS
# # -------------------------------

# for trial in range(NUM_TRIALS):
#     print(f"\n==============================")
#     print(f" TRIAL {trial + 1}/{NUM_TRIALS}")
#     print(f"==============================")

#     # --- RESET ---
#     reset_simulation(model, data, viewer)

#     # --- OPEN HAND ---
#     for _ in range(OPEN_STEPS):
#         data.ctrl[:] = 0.0
#         mujoco.mj_step(model, data)
#         viewer.sync()
#         time.sleep(0.005)

#     # --- CLOSE FINGERS (FAST) ---
#     for _ in range(CLOSE_STEPS):
#         data.ctrl[:] = 0.0
#         for idx in FINGER_ACTUATORS:
#             data.ctrl[idx] = 1.0
#         mujoco.mj_step(model, data)
#         viewer.sync()
#         time.sleep(0.005)

#     print("Contacts after close:", get_contacts(model, data))
#     print("Is grasping:", is_grasping(model, data))

#     # --- HOLD ---
#     for _ in range(HOLD_STEPS):
#         for idx in FINGER_ACTUATORS:
#             data.ctrl[idx] = 1.0
#         mujoco.mj_step(model, data)
#         viewer.sync()
#         time.sleep(0.01)

#     print("Final grasp check:", is_grasping(model, data))

# # -------------------------------
# # KEEP WINDOW OPEN
# # -------------------------------

# print("\nAll trials finished.")
# print("MuJoCo window will stay open.")
# print("Close the viewer window manually to exit.")

# while viewer.is_running():
#     mujoco.mj_step(model, data)
#     viewer.sync()
#     time.sleep(0.02)
###########################################################################################
 # ============================================================
# Shadow Dexterous Hand + Crosley Alarm Clock
# 5 grasp trials + metrics (contacts, touch, hold_ratio, drift, energy, smooth, success)
# Fixed camera, minimal prints, viewer stays open
# ============================================================

import os
import time
import mujoco
import mujoco.viewer
import numpy as np

# -------------------------------
# CONFIG
# -------------------------------
ROOT_DIR = r"C:\Users\rad\itmo\new_roms"
# XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "hand_manipulate_clock.xml")

# XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_block_touch_sensors.xml")
# XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_pen_touch_sensors.xml")
XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_egg_touch_sensors.xml")


NUM_TRIALS  = 5
OPEN_STEPS  = 40
CLOSE_STEPS = 120
HOLD_STEPS  = 200

# success/stability thresholds (как в v3_4)
Z_MIN      = 0.05
V_MAX      = 2.0
DRIFT_MAX  = 0.25
SUCCESS_HOLD_RATIO = 0.90

# дополнительные ограничения на "реальный" захват
MIN_CONTACTS_AFTER_CLOSE = 3
MIN_TIP_ACTIVE_AFTER_CLOSE = 1
MIN_TIP_SUM_AFTER_CLOSE    = 0.25

# Optional: slowdown visualization
SLEEP_OPEN  = 0.005
SLEEP_CLOSE = 0.005
SLEEP_HOLD  = 0.01

print("XML exists:", os.path.exists(XML_PATH))
print("XML:", XML_PATH)

# -------------------------------
# LOAD MUJOCO MODEL
# -------------------------------
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

DT = float(model.opt.timestep)

print("\nScene loaded successfully")
print(f"Bodies={model.nbody} Joints={model.njnt} Actuators={model.nu} Sensors={model.nsensor} timestep={DT}")

# -------------------------------
# BUILD SENSOR SLICES (универсально для всех типов сенсоров)
# -------------------------------
sensor_slices = {}
off = 0
for i in range(model.nsensor):
    s = model.sensor(i)
    dim = int(np.asarray(s.dim).item())
    sensor_slices[s.name] = (off, dim, int(np.asarray(s.type).item()))
    off += dim

def sensor_sum(name: str) -> float:
    """Sum over sensor dimensions (если сенсора нет — 0)."""
    if name not in sensor_slices:
        return 0.0
    st, dim, _ = sensor_slices[name]
    return float(np.sum(data.sensordata[st:st+dim]))

# fingertip touch sensors (как в v3_4)
FINGERTIP_TOUCH = {
    "FF": "robot0:ST_Tch_fftip",
    "MF": "robot0:ST_Tch_mftip",
    "RF": "robot0:ST_Tch_rftip",
    "LF": "robot0:ST_Tch_lftip",
    "TH": "robot0:ST_Tch_thtip",
}
FINGERS = ["FF", "MF", "RF", "LF", "TH"]
touch_present = {f: (FINGERTIP_TOUCH[f] in sensor_slices) for f in FINGERS}
print("Fingertip touch present:", touch_present)

# palm sensors (если есть)
PALM_SENSOR_NAMES = [
    n for n in sensor_slices.keys()
    if ("ts_palm" in n.lower()) or ("palm" in n.lower())
]

# -------------------------------
# OBJECT ID (стараемся по "object", иначе fallback по ключевому слову)
# -------------------------------
OBJ_BODY_NAME = "object"
obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJ_BODY_NAME)

# fallback keyword for geoms in contacts, if body "object" not found
OBJECT_KEYWORD_FALLBACK = "clock"


# -------------------------------
# пальцы: 2..19 (0–1 — запястье)
# -------------------------------
FINGER_ACTUATORS = list(range(2, 20))

# -------------------------------
# HELPERS
# -------------------------------
def count_object_contacts_by_body(obj_body_id: int) -> int:
    """Количество контактов 'объект—не объект' по bodyid геомов."""
    if obj_body_id < 0:
        return 0
    cnt = 0
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[int(c.geom1)])
        b2 = int(model.geom_bodyid[int(c.geom2)])
        if (b1 == obj_body_id and b2 != obj_body_id) or (b2 == obj_body_id and b1 != obj_body_id):
            cnt += 1
    return cnt

def get_contacts_by_keyword(object_keyword=OBJECT_KEYWORD_FALLBACK):
    """Список пар имен геомов, где встречается ключевое слово (fallback)."""
    contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = model.geom(c.geom1).name
        g2 = model.geom(c.geom2).name
        if object_keyword in g1 or object_keyword in g2:
            contacts.append((g1, g2))
    return contacts


def count_object_contacts() -> int:
    """Универсально: если есть body 'object' — считаем по bodyid, иначе по ключевому слову."""
    if obj_bid >= 0:
        return int(count_object_contacts_by_body(obj_bid))
    return int(len(get_contacts_by_keyword()))

def reset_simulation(viewer):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(1.0)



def object_pos():
    """Позиция объекта: если есть body 'object' — берём его xpos; иначе ищем по ключевому слову в body/geom (очень мягкий fallback)."""
    if obj_bid >= 0:
        return data.xpos[obj_bid].copy()

    # мягкий fallback: попробуем найти body, имя которого содержит "clock"
    for b in range(model.nbody):
        name = model.body(b).name
        if name and (OBJECT_KEYWORD_FALLBACK in name):
            return data.xpos[b].copy()

    # совсем fallback: вернём NaN
    return np.array([np.nan, np.nan, np.nan], dtype=float)

def object_speed():
    """Скорость объекта: если есть body 'object' — берём линейную скорость из cvel[3:6]. Иначе NaN."""
    if obj_bid >= 0:
        v_now = data.cvel[obj_bid][3:6]
        return float(np.linalg.norm(v_now))
    return float("nan")


# -------------------------------
# METRICS RUN (one trial)
# -------------------------------
def run_trial(trial_idx: int, viewer):
    # metrics accumulators
    energy = 0.0
    smooth = 0.0
    prev_u = np.zeros(model.nu, dtype=float)

    # -------------------------------
    # OPEN
    # -------------------------------
    for _ in range(OPEN_STEPS):
        u = np.zeros(model.nu, dtype=float)
        data.ctrl[:] = u

        # energy/smooth
        energy += float(np.sum(u*u))
        smooth += float(np.sum((u - prev_u) ** 2))
        prev_u = u

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(SLEEP_OPEN)

    # -------------------------------
    # CLOSE (в твоём base: просто 1.0 на пальцы)
    # -------------------------------
    last_touch = {f: 0.0 for f in FINGERS}

    for _ in range(CLOSE_STEPS):
        u = np.zeros(model.nu, dtype=float)
        for idx in FINGER_ACTUATORS:
            u[idx] = 1.0
        data.ctrl[:] = u

        # touch snapshot (последнее значение до конца close)
        for f in FINGERS:
            last_touch[f] = sensor_sum(FINGERTIP_TOUCH[f]) if touch_present[f] else 0.0

        # energy/smooth
        energy += float(np.sum(u*u))
        smooth += float(np.sum((u - prev_u) ** 2))
        prev_u = u

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(SLEEP_CLOSE)

    # after close: contact/touch metrics
    contacts = int(count_object_contacts())
    pos_after_close = object_pos()
    z = float(pos_after_close[2]) if np.isfinite(pos_after_close[2]) else float("nan")

    tip_vals = np.array([last_touch[f] for f in FINGERS], dtype=float)
    tip_sum = float(np.sum(tip_vals))
    tip_active = int(np.sum(tip_vals > 0.0))

    palm_sum = float(np.sum([sensor_sum(n) for n in PALM_SENSOR_NAMES])) if len(PALM_SENSOR_NAMES) else 0.0

    # -------------------------------
    # HOLD
    # -------------------------------
    pos_ref = pos_after_close.copy()
    stable_steps = 0
    max_speed = 0.0
    max_drift = 0.0

    for _ in range(HOLD_STEPS):
        u = np.zeros(model.nu, dtype=float)
        for idx in FINGER_ACTUATORS:
            u[idx] = 1.0
        data.ctrl[:] = u

        # energy/smooth
        energy += float(np.sum(u*u))
        smooth += float(np.sum((u - prev_u) ** 2))
        prev_u = u

        mujoco.mj_step(model, data)

        p_now = object_pos()
        speed = object_speed()
        drift = float(np.linalg.norm(p_now - pos_ref)) if np.all(np.isfinite(p_now)) else float("nan")

        if np.isfinite(speed):
            max_speed = max(max_speed, float(speed))
        if np.isfinite(drift):
            max_drift = max(max_drift, float(drift))


        # стабильность (как в v3_4)
        if np.all(np.isfinite(p_now)):
            if (p_now[2] > Z_MIN) and (speed < V_MAX) and (drift < DRIFT_MAX):
                stable_steps += 1

        viewer.sync()
        time.sleep(SLEEP_HOLD)

    hold_ratio = stable_steps / float(HOLD_STEPS)

    # "реальный" захват: контакты + сенсоры
    real_grasp = (contacts >= MIN_CONTACTS_AFTER_CLOSE) and \
                 (tip_active >= MIN_TIP_ACTIVE_AFTER_CLOSE) and \
                 (tip_sum >= MIN_TIP_SUM_AFTER_CLOSE)

    success = 1 if (hold_ratio >= SUCCESS_HOLD_RATIO and real_grasp) else 0

    return {
        "trial": int(trial_idx),
        "contacts": int(contacts),
        "tip_active": int(tip_active),
        "tip_sum": float(tip_sum),
        "palm_sum": float(palm_sum),
        "z": float(z),
        "hold_ratio": float(hold_ratio),
        "max_speed": float(max_speed),
        "max_drift": float(max_drift),
        "energy": float(energy),
        "smooth": float(smooth),
        "success": int(success),
        "touch_per_finger": {f: float(last_touch[f]) for f in FINGERS},
    }


# -------------------------------
# LAUNCH VIEWER
# -------------------------------
viewer = mujoco.viewer.launch_passive(model, data)

# -------------------------------
# GRASP TRIALS + METRICS
# -------------------------------
all_metrics = []

for t in range(1, NUM_TRIALS + 1):
    print("\n==============================")
    print(f" TRIAL {t}/{NUM_TRIALS}")
    print("==============================")

    reset_simulation(viewer)
    m = run_trial(t, viewer)
    all_metrics.append(m)

    tp = m["touch_per_finger"]
    # компактный лог (можешь расширить)
    print(
        f"SUCCESS={m['success']} hold={m['hold_ratio']:.2f} "
        f"con={m['contacts']} z={m['z']:.3f} "
        f"drift={m['max_drift']:.3f} speed={m['max_speed']:.3f} "
        f"tip_active={m['tip_active']} tip_sum={m['tip_sum']:.3f} palm_sum={m['palm_sum']:.3f} "
        f"energy={m['energy']:.1f} smooth={m['smooth']:.1f} "
        f"FF={tp['FF']:.3f} MF={tp['MF']:.3f} RF={tp['RF']:.3f} LF={tp['LF']:.3f} TH={tp['TH']:.3f}"
    )

# -------------------------------
# SUMMARY
# -------------------------------
succ_rate = float(np.mean([m["success"] for m in all_metrics])) if all_metrics else 0.0
avg_contacts = float(np.mean([m["contacts"] for m in all_metrics])) if all_metrics else 0.0
avg_hold = float(np.mean([m["hold_ratio"] for m in all_metrics])) if all_metrics else 0.0
avg_drift = float(np.mean([m["max_drift"] for m in all_metrics])) if all_metrics else 0.0

print("\n==============================")
print("SUMMARY (all trials)")
print("==============================")
print(f"Success rate: {succ_rate:.2f}")
print(f"Avg contacts: {avg_contacts:.2f}")
print(f"Avg hold_ratio: {avg_hold:.2f}")
print(f"Avg max_drift: {avg_drift:.3f}")

print("\nAll trials finished.")
print("MuJoCo window will stay open.")
print("Close the viewer window manually to exit.")

while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.02)
