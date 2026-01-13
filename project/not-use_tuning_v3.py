# ============================================================
# Shadow Dexterous Hand + Crosley Alarm Clock
# Auto-tune theta via Random Search (ONE CELL)
# - Baseline A, Baseline B, Method (auto theta)
# - Metrics + score
# - LF (мизинец) усиливаем через constraints + бонусы в score
# - Viewer stays open + HOLD best grasp
# ============================================================

import os, time
import numpy as np
import mujoco
import mujoco.viewer

# -------------------------------
# PATHS (ПОД ТВОЮ ЛОКАЛКУ)
# -------------------------------
ROOT_DIR = r"C:\Users\rad\itmo\new_roms"
XML_PATH = os.path.join(ROOT_DIR, "project", "models", "hand", "hand_manipulate_clock.xml")

print("Root dir:", ROOT_DIR)
print("XML exists:", os.path.exists(XML_PATH))
print("XML:", XML_PATH)
if not os.path.exists(XML_PATH):
    raise FileNotFoundError(f"Scene XML not found: {XML_PATH}")

# -------------------------------
# EPISODE CONFIG
# -------------------------------
OPEN_STEPS  = 40
CLOSE_STEPS = 120
HOLD_STEPS  = 300  # чуть больше, чтобы реально проверить удержание

# latch не включать сразу (чтобы не "приклеилось" от касания при падении)
MIN_CLOSE_STEPS_BEFORE_LATCH = 15

# Success критерии в HOLD
Z_MIN      = 0.05
V_MAX      = 2.0
DRIFT_MAX  = 0.25
SUCCESS_HOLD_RATIO = 0.90

# -------------------------------
# OPTIMIZER CONFIG
# -------------------------------
EVAL_TRIALS_PER_THETA = 5
RANDOM_ITERS = 80
SEED0 = 123

# Диапазоны поиска:
W_RANGE = (0.4, 2.0)         # веса по пальцам
TH_TOUCH_RANGE = (0.002, 0.08)
K_HOLD_RANGE   = (0.05, 0.8)

# Constraints для LF (мизинца)
LF_MIN = 0.8     # нижняя граница для wLF (минимальный вес мизинца)

# score weights
BONUS_LF_TOUCH  = 0.10
BONUS_LF_ACTIVE = 0.20
PENALTY_NO_LF   = 0.30

LAMBDA_DRIFT  = 0.90
LAMBDA_ENERGY = 1e-4
LAMBDA_SMOOTH = 5e-5

# -------------------------------
# LOAD MUJOCO MODEL
# -------------------------------
print("Loading model...")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)
print("Model loaded. nu =", model.nu)

# Viewer stays open
viewer = mujoco.viewer.launch_passive(model, data)

# -------------------------------
# UTILITIES
# -------------------------------
def sensor_sum(name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        return 0.0
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return float(np.sum(data.sensordata[adr:adr+dim]))

def build_finger_actuator_map():
    """
    Пытаемся собрать мапу актуаторов пальцев.
    Важно: в разных xml имена могут отличаться.
    В V3 предполагается, что FINGERS = ['FF','MF','RF','LF','TH'] доступны,
    и что есть список актуаторов для "закрытия" каждого пальца.
    """
    act_names = []
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        act_names.append(nm or "")

    finger_act = { "FF": [], "MF": [], "RF": [], "LF": [], "TH": [] }
    wrist_act  = []

    for i, nm in enumerate(act_names):
        low = nm.lower()

        # очень грубые эвристики
        if "wrist" in low or "wrj" in low:
            wrist_act.append(i)
            continue

        if "ff" in low:
            finger_act["FF"].append(i)
        elif "mf" in low:
            finger_act["MF"].append(i)
        elif "rf" in low:
            finger_act["RF"].append(i)
        elif "lf" in low:
            finger_act["LF"].append(i)
        elif "th" in low:
            finger_act["TH"].append(i)

    # fallback: как в твоём простом тесте 0–1 wrist, 2–... fingers
    # если не нашли по имени ни одного пальца:
    total_found = sum(len(v) for v in finger_act.values())
    if total_found == 0 and model.nu >= 3:
        # назначим всем пальцам общий диапазон "пальцевых" актуаторов
        # (это хуже, но не даст коду упасть)
        all_fingers = list(range(2, model.nu))
        # просто делим равномерно по 5 пальцам:
        chunks = np.array_split(all_fingers, 5)
        for f, ch in zip(["FF","MF","RF","LF","TH"], chunks):
            finger_act[f] = list(map(int, ch))
        wrist_act = [0, 1] if model.nu >= 2 else []

    return finger_act, wrist_act

FINGERS = ["FF","MF","RF","LF","TH"]
FINGER_ACTS, WRIST_ACTS = build_finger_actuator_map()

# -------------------------------
# SENSOR NAMES (как в V3-логике)
# -------------------------------
# В gymnasium-активах обычно есть такие сенсоры. Если в твоём XML названия другие —
# sensor_sum вернёт 0.0 и это не сломает код, но метрики будут беднее.
FINGERTIP_TOUCH = {
    "FF": "FF_tip_touch",
    "MF": "MF_tip_touch",
    "RF": "RF_tip_touch",
    "LF": "LF_tip_touch",
    "TH": "TH_tip_touch",
}

# ладонные сенсоры (если есть)
PALM_SENSOR_NAMES = [
    "palm_touch", "palm_touch_1", "palm_touch_2"
]

# -------------------------------
# OBJECT NAMES (как в V3: часы уже в XML)
# -------------------------------
OBJ_BODY_NAME = "object"      # тело объекта
OBJ_GEOM_NAME = "object"      # геом объекта

obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJ_BODY_NAME)
obj_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJ_GEOM_NAME)
if obj_bid < 0:
    # fallback: попробуем "clock"
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "clock")
if obj_gid < 0:
    obj_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "clock")

if obj_bid < 0 or obj_gid < 0:
    print("[WARN] object body/geom not found by names (object/clock).")
    print("You may need to set correct OBJ_BODY_NAME / OBJ_GEOM_NAME for your XML.")

def count_object_contacts():
    if obj_gid < 0:
        return 0
    c = 0
    for i in range(data.ncon):
        con = data.contact[i]
        if con.geom1 == obj_gid or con.geom2 == obj_gid:
            c += 1
    return c

def set_object_pose_random(seed):
    """
    В V3 объект "часы" рандомизировался.
    Здесь оставляем совместимую заготовку:
    слегка вращаем и сдвигаем по xy.
    """
    if obj_bid < 0:
        return
    rng = np.random.RandomState(seed)
    # небольшой сдвиг
    dx = rng.uniform(-0.02, 0.02)
    dy = rng.uniform(-0.02, 0.02)

    # небольшое вращение вокруг z
    ang = rng.uniform(-0.6, 0.6)
    quat = np.array([np.cos(ang/2), 0, 0, np.sin(ang/2)], dtype=float)

    # позиция/кватернион тела
    data.xpos[obj_bid][0] += dx
    data.xpos[obj_bid][1] += dy
    data.xquat[obj_bid][:] = quat

def controller_step(t_close, theta, latched):
    """
    theta = [wFF, wMF, wRF, wLF, wTH, th_touch, k_hold]
    """
    wFF, wMF, wRF, wLF, wTH, th_touch, k_hold = theta
    w = {"FF": wFF, "MF": wMF, "RF": wRF, "LF": wLF, "TH": wTH}

    # ramp base усилия закрытия
    u_base = float(t_close) / float(CLOSE_STEPS)
    u_base = np.clip(u_base, 0.0, 1.0)

    u = np.zeros(model.nu, dtype=float)

    # wrist (если надо) оставляем 0
    for i in WRIST_ACTS:
        if 0 <= i < model.nu:
            u[i] = 0.0

    # пальцы
    for f in FINGERS:
        # latch: после касания — усиливаем удержание
        if latched[f]:
            amp = np.clip(u_base + k_hold, 0.0, 1.0)
        else:
            amp = u_base

        # распределяем по актуаторам данного пальца
        for aid in FINGER_ACTS[f]:
            if 0 <= aid < model.nu:
                u[aid] = np.clip(w[f] * amp, 0.0, 1.0)

    # обновим latch по сенсорам (после MIN_CLOSE_STEPS_BEFORE_LATCH)
    if t_close >= MIN_CLOSE_STEPS_BEFORE_LATCH:
        for f in FINGERS:
            sname = FINGERTIP_TOUCH.get(f, "")
            if sname:
                val = sensor_sum(sname)
                if val >= th_touch:
                    latched[f] = True

    return u

def run_trial(theta, seed, viewer=None, do_render=False,
             sleep_open=0.0, sleep_close=0.0, sleep_hold=0.0):

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    set_object_pose_random(seed)
    mujoco.mj_forward(model, data)

    latched = {f: False for f in FINGERS}

    energy = 0.0
    smooth = 0.0
    prev_u = np.zeros(model.nu, dtype=float)

    # какие сенсоры реально есть
    touch_present = {f: (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, FINGERTIP_TOUCH[f]) >= 0) for f in FINGERS}
    last_touch = {f: 0.0 for f in FINGERS}

    # ---- OPEN ----
    for _ in range(OPEN_STEPS):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        if viewer is not None and do_render:
            viewer.sync()
            time.sleep(sleep_open)

    # ---- CLOSE ----
    for t in range(1, CLOSE_STEPS+1):
        u = controller_step(t_close=t, theta=theta, latched=latched)
        data.ctrl[:] = u

        for f in FINGERS:
            last_touch[f] = sensor_sum(FINGERTIP_TOUCH[f]) if touch_present[f] else 0.0

        energy += float(np.sum(u*u))
        smooth += float(np.sum((u-prev_u)**2))
        prev_u = u

        mujoco.mj_step(model, data)
        if viewer is not None and do_render:
            viewer.sync()
            time.sleep(sleep_close)

    contacts = int(count_object_contacts())
    z = float(data.xpos[obj_bid][2]) if obj_bid >= 0 else 0.0

    tip_vals = np.array([last_touch[f] for f in FINGERS], dtype=float)
    tip_sum = float(np.sum(tip_vals))
    tip_active = int(np.sum(tip_vals > 0.0))

    palm_sum = float(np.sum([sensor_sum(n) for n in PALM_SENSOR_NAMES])) if len(PALM_SENSOR_NAMES) else 0.0

    # ---- HOLD ----
    if obj_bid >= 0:
        pos_ref = data.xpos[obj_bid].copy()
    else:
        pos_ref = np.zeros(3)

    ok = 0
    max_speed = 0.0
    max_drift = 0.0

    for _ in range(HOLD_STEPS):
        u = controller_step(t_close=CLOSE_STEPS, theta=theta, latched=latched)
        data.ctrl[:] = u

        energy += float(np.sum(u*u))
        smooth += float(np.sum((u-prev_u)**2))
        prev_u = u

        mujoco.mj_step(model, data)

        if obj_bid >= 0:
            p = data.xpos[obj_bid]
            # cvel: [angvel(3), linvel(3)] in body frame-ish; берём линейную
            v = data.cvel[obj_bid][3:6]
            speed = float(np.linalg.norm(v))
            drift = float(np.linalg.norm(p - pos_ref))
        else:
            speed = 0.0
            drift = 0.0
            p = np.zeros(3)

        max_speed = max(max_speed, speed)
        max_drift = max(max_drift, drift)

        z_now = float(p[2]) if obj_bid >= 0 else 0.0

        cond = (z_now >= Z_MIN) and (speed <= V_MAX) and (drift <= DRIFT_MAX)
        ok += int(cond)

        if viewer is not None and do_render:
            viewer.sync()
            time.sleep(sleep_hold)

    hold_ratio = ok / float(HOLD_STEPS)
    success = int(hold_ratio >= SUCCESS_HOLD_RATIO)

    return {
        "contacts": contacts,
        "tip_active": tip_active,
        "tip_sum": tip_sum,
        "palm_sum": palm_sum,
        "z": z,
        "hold_ratio": float(hold_ratio),
        "max_speed": float(max_speed),
        "max_drift": float(max_drift),
        "energy": float(energy),
        "smooth": float(smooth),
        "success": int(success),
        "touch_per_finger": {f: float(last_touch[f]) for f in FINGERS},
    }

# -------------------------------
# SCORE FUNCTION (higher better)
# --------------------
def score_metrics(m):
    score = 0.0
    score += 2.0 * m["success"]
    score += 1.0 * m["hold_ratio"]

    score += 0.05 * m["contacts"]
    score += 0.03 * m["tip_active"]

    # LF priority (мизинец)
    lf_touch = m["touch_per_finger"]["LF"]
    score += BONUS_LF_TOUCH * lf_touch
    if lf_touch > 0.0:
        score += BONUS_LF_ACTIVE
    else:
        score -= PENALTY_NO_LF

    # penalties
    score -= LAMBDA_DRIFT  * m["max_drift"]
    score -= LAMBDA_ENERGY * m["energy"]
    score -= LAMBDA_SMOOTH * m["smooth"]
    return float(score)

def eval_theta(theta, base_seed):
    scores = []
    metrics = []
    for k in range(EVAL_TRIALS_PER_THETA):
        m = run_trial(theta=theta, seed=base_seed + k, viewer=None, do_render=False)
        metrics.append(m)
        scores.append(score_metrics(m))
    return float(np.mean(scores)), metrics

def print_trial_line(tag, m, score=None):
    if score is None:
        score = score_metrics(m)
    print(f"{tag:>10s} | score {score:7.4f} | succ {m['success']} | hold {m['hold_ratio']:.2f} | "
          f"con {m['contacts']:2d} | tipAct {m['tip_active']:d} | drift {m['max_drift']:.3f} | "
          f"v {m['max_speed']:.3f} | E {m['energy']:.1f}")

# -------------------------------
# BASELINES
# -------------------------------
baseline_A = np.array([1.0, 1.0, 1.0, max(0.9, LF_MIN), 1.2, 0.010, 0.30], dtype=float)
baseline_B = np.array([1.2, 0.9, 1.1, max(1.2, LF_MIN), 1.4, 0.015, 0.45], dtype=float)

print("\n--- Evaluate baselines ---")
sA, mA = eval_theta(baseline_A, SEED0 + 10000)
sB, mB = eval_theta(baseline_B, SEED0 + 20000)
print("Baseline A theta:", baseline_A, "score:", round(sA, 4))
print("Baseline B theta:", baseline_B, "score:", round(sB, 4))

best_theta = baseline_A.copy()
best_score = sA
best_pack  = mA
if sB > best_score:
    best_theta, best_score, best_pack = baseline_B.copy(), sB, mB

# -------------------------------
# RANDOM SEARCH
# -------------------------------
rng = np.random.RandomState(SEED0)

print("\n--- Random search (LF constrained) ---")
for it in range(1, RANDOM_ITERS + 1):
    wFF = rng.uniform(*W_RANGE)
    wMF = rng.uniform(*W_RANGE)
    wRF = rng.uniform(*W_RANGE)
    wLF = max(rng.uniform(*W_RANGE), LF_MIN)  # constraint
    wTH = rng.uniform(*W_RANGE)

    th_touch = rng.uniform(*TH_TOUCH_RANGE)
    k_hold   = rng.uniform(*K_HOLD_RANGE)

    theta = np.array([wFF, wMF, wRF, wLF, wTH, th_touch, k_hold], dtype=float)
    s, pack = eval_theta(theta, SEED0 + 30000 + it*100)

    if s > best_score:
        best_theta = theta.copy()
        best_score = s
        best_pack  = pack
        print(f"[BEST] iter {it:03d} score {best_score:.4f} theta {best_theta}")

print("\n=== BEST THETA FOUND ===")
print("best_score:", best_score)
print("best_theta:", best_theta)

# покажем несколько trial-метрик для лучшего
print("\n--- Best theta trials ---")
for k, mm in enumerate(best_pack[:min(5, len(best_pack))]):
    print_trial_line(f"trial{k}", mm)

# -------------------------------
# VISUALIZE + HOLD (чтобы реально удерживал)
# -------------------------------
print("\n--- Visualize best theta (close + hold) ---")
m_vis = run_trial(theta=best_theta, seed=SEED0 + 999, viewer=viewer, do_render=True,
                  sleep_open=0.01, sleep_close=0.01, sleep_hold=0.01)
print_trial_line("VIS", m_vis)

print("\nHolding BEST grasp. Close the viewer window to exit.")
# Прямо удерживаем в бесконечном цикле тем же контроллером
latched = {f: True for f in FINGERS}  # на удержании считаем что latch уже есть
while viewer.is_running():
    u = controller_step(t_close=CLOSE_STEPS, theta=best_theta, latched=latched)
    data.ctrl[:] = u
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.01)
