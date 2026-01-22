import os, time
import numpy as np
import mujoco
import mujoco.viewer


# -------------------------------
# PATHS
# -------------------------------
ROOT_DIR = r"C:\Users\rad\itmo\new_roms"

XML_LIST = [
    os.path.join(ROOT_DIR, "project", "models", "hand", "hand_manipulate_clock.xml"),

    os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_block_touch_sensors.xml"),
    os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_egg_touch_sensors.xml"),
    os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_pen_touch_sensors.xml"),
    
]

# -------------------------------
# EPISODE CONFIG
# -------------------------------
OPEN_STEPS  = 40
CLOSE_STEPS = 120
HOLD_STEPS  = 200

MIN_CLOSE_STEPS_BEFORE_LATCH = 15

Z_MIN      = 0.05
V_MAX      = 2.0
DRIFT_MAX  = 0.25
SUCCESS_HOLD_RATIO = 0.90

# дополнительные ограничения на "реальный" захват
MIN_CONTACTS_AFTER_CLOSE = 3
MIN_TIP_ACTIVE_AFTER_CLOSE = 1
MIN_TIP_SUM_AFTER_CLOSE    = 0.25

# -------------------------------
# OPTIMIZER CONFIG
# -------------------------------
EVAL_TRIALS_PER_THETA = 5      # <-- КАК ТЫ ПРОСИЛА: каждый theta оцениваем 5 раз
RANDOM_ITERS = 25              # <-- чтобы не было тысяч запусков (поднимешь позже)
SEED0 = 2000

# -------------------------------
# SCORE WEIGHTS
# -------------------------------
LAMBDA_ENERGY = 1e-4
LAMBDA_SMOOTH = 1e-3
LAMBDA_DRIFT  = 1.0

# search ranges
TH_TOUCH_RANGE = (0.001, 0.05)
K_HOLD_RANGE   = (0.10, 0.60)
W_RANGE        = (0.5, 1.5)

# ---- LF (мизинец) усиление ----
LF_MIN = 1.10
LF_RANGE = (1.10, 1.80)
BONUS_LF_TOUCH  = 0.02
BONUS_LF_ACTIVE = 0.08
PENALTY_NO_LF   = 0.15

# -------------------------------
# RENDER SETTINGS
# -------------------------------
SLOWDOWN = 2.0               # замедление визуализации в 2 раза
RENDER_BASELINES = True      # показать baseline A/B (по 1 траю)
RENDER_EVERY_N_THETA = 10    # в поиске показывать каждый N-й theta (по 1 траю)
VISUAL_TRIALS = 5            # показать best_theta 5 раз (как в эталоне)


def run_for_xml(XML_PATH: str):
    xml_tag = os.path.basename(XML_PATH)

    def p(msg: str):
        print(f"[{xml_tag}] {msg}")

    print("\n" + "="*94)
    p(f"XML: {XML_PATH}")
    p(f"Exists: {os.path.exists(XML_PATH)}")
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(XML_PATH)

    # -------------------------------
    # LOAD MODEL
    # -------------------------------
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    DT = float(model.opt.timestep)
    # "замедление" = добавляем sleep после каждого mj_step
    EXTRA_SLEEP_PER_STEP = max(0.0, DT * (SLOWDOWN - 1.0))

    p("Scene loaded successfully")
    p(f"Bodies={model.nbody} Joints={model.njnt} Actuators={model.nu} Sensors={model.nsensor} timestep={DT}")

    # -------------------------------
    # OBJECT IDs
    # -------------------------------
    OBJ_BODY_NAME  = "object"
    OBJ_JOINT_NAME = "object:joint"

    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJ_BODY_NAME)
    if obj_bid < 0:
        raise RuntimeError(f"[{xml_tag}] Body '{OBJ_BODY_NAME}' not found in XML.")

    obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJ_JOINT_NAME)
    if obj_jid < 0:
        raise RuntimeError(f"[{xml_tag}] Joint '{OBJ_JOINT_NAME}' not found. Check your XML joint name.")

    obj_qpos_adr = int(model.jnt_qposadr[obj_jid])  # free joint: qpos[adr:adr+7]

    # -------------------------------
    # BUILD SENSOR SLICES
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
        return float(np.sum(data.sensordata[st:st+dim]))

    # fingertip touch sensors
    FINGERTIP_TOUCH = {
        "FF": "robot0:ST_Tch_fftip",
        "MF": "robot0:ST_Tch_mftip",
        "RF": "robot0:ST_Tch_rftip",
        "LF": "robot0:ST_Tch_lftip",
        "TH": "robot0:ST_Tch_thtip",
    }
    FINGERS = ["FF", "MF", "RF", "LF", "TH"]
    touch_present = {f: (FINGERTIP_TOUCH[f] in sensor_slices) for f in FINGERS}
    p(f"Fingertip touch present: {touch_present}")

    PALM_SENSOR_NAMES = [n for n in sensor_slices.keys() if ("ts_palm" in n.lower()) or ("palm" in n.lower())]

    # -------------------------------
    # ACTUATORS PER FINGER
    # -------------------------------
    def build_finger_actuator_map():
        m = {"FF": [], "MF": [], "RF": [], "LF": [], "TH": []}
        for i in range(model.nu):
            name = model.actuator(i).name
            if ":A_FF" in name: m["FF"].append(i)
            if ":A_MF" in name: m["MF"].append(i)
            if ":A_RF" in name: m["RF"].append(i)
            if ":A_LF" in name: m["LF"].append(i)
            if ":A_TH" in name: m["TH"].append(i)
        return m

    FINGER_ACT = build_finger_actuator_map()
    p(f"Actuators per finger: { {k: len(v) for k, v in FINGER_ACT.items()} }")

    # -------------------------------
    # CONTACT COUNT (hand-object)
    # -------------------------------
    def count_object_contacts():
        cnt = 0
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[int(c.geom1)])
            b2 = int(model.geom_bodyid[int(c.geom2)])
            if (b1 == obj_bid and b2 != obj_bid) or (b2 == obj_bid and b1 != obj_bid):
                cnt += 1
        return cnt

    # -------------------------------
    # RANDOMIZE OBJECT START (XY + yaw)
    # -------------------------------
    def set_object_pose_random(seed: int, xy_sigma=0.015, yaw_range=np.deg2rad(25)):
        rng = np.random.default_rng(seed)
        qpos = data.qpos

        base_pos  = qpos[obj_qpos_adr:obj_qpos_adr+3].copy()
        base_quat = qpos[obj_qpos_adr+3:obj_qpos_adr+7].copy()

        dx, dy = rng.normal(0, xy_sigma, size=2)
        pos = base_pos.copy()
        pos[0] += dx
        pos[1] += dy

        yaw = float(rng.uniform(-yaw_range, yaw_range))
        q_yaw = np.zeros(4, dtype=float)
        mujoco.mju_axisAngle2Quat(q_yaw, np.array([0.0, 0.0, 1.0], dtype=float), yaw)

        q = np.zeros(4, dtype=float)
        mujoco.mju_mulQuat(q, q_yaw, base_quat)

        qpos[obj_qpos_adr:obj_qpos_adr+3] = pos
        qpos[obj_qpos_adr+3:obj_qpos_adr+7] = q

    # -------------------------------
    # CONTROLLER (theta)
    # theta = [wFF,wMF,wRF,wLF,wTH, th_touch, k_hold]
    # -------------------------------
    def controller_step(t_close, theta, latched):
        w = np.array(theta[:5], dtype=float)
        th_touch = float(theta[5])
        k_hold   = float(theta[6])

        u_base = np.clip(t_close / max(1, CLOSE_STEPS), 0.0, 1.0)

        if t_close >= MIN_CLOSE_STEPS_BEFORE_LATCH:
            for f in FINGERS:
                if touch_present[f]:
                    tv = sensor_sum(FINGERTIP_TOUCH[f])
                    if (not latched[f]) and (tv > th_touch):
                        latched[f] = True

        u = np.zeros(model.nu, dtype=float)
        for i, f in enumerate(FINGERS):
            wf = float(w[i])
            amp = (u_base + k_hold) if latched[f] else u_base
            uf = float(np.clip(wf * amp, 0.0, 1.0))
            for ai in FINGER_ACT.get(f, []):
                u[ai] = uf
        return u

    def render_step(viewer):
        viewer.sync()
        if EXTRA_SLEEP_PER_STEP > 0:
            time.sleep(EXTRA_SLEEP_PER_STEP)

    # -------------------------------
    # RUN ONE TRIAL (with metrics)
    # -------------------------------
    def run_trial(theta, seed, viewer=None, do_render=False):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        set_object_pose_random(seed)
        mujoco.mj_forward(model, data)

        latched = {f: False for f in FINGERS}
        energy = 0.0
        smooth = 0.0
        prev_u = np.zeros(model.nu, dtype=float)

        # OPEN
        for _ in range(OPEN_STEPS):
            u = np.zeros(model.nu, dtype=float)
            data.ctrl[:] = u
            energy += float(np.sum(u*u))
            smooth += float(np.sum((u-prev_u)**2))
            prev_u = u
            mujoco.mj_step(model, data)
            if viewer is not None and do_render:
                render_step(viewer)

        # CLOSE
        last_touch = {f: 0.0 for f in FINGERS}
        for t in range(1, CLOSE_STEPS + 1):
            u = controller_step(t_close=t, theta=theta, latched=latched)
            data.ctrl[:] = u

            for f in FINGERS:
                last_touch[f] = sensor_sum(FINGERTIP_TOUCH[f]) if touch_present[f] else 0.0

            energy += float(np.sum(u*u))
            smooth += float(np.sum((u-prev_u)**2))
            prev_u = u

            mujoco.mj_step(model, data)
            if viewer is not None and do_render:
                render_step(viewer)

        contacts = int(count_object_contacts())
        obj_pos = data.xpos[obj_bid].copy()
        z = float(obj_pos[2])

        tip_vals = np.array([last_touch[f] for f in FINGERS], dtype=float)
        tip_sum = float(np.sum(tip_vals))
        tip_active = int(np.sum(tip_vals > 0.0))

        palm_sum = float(np.sum([sensor_sum(n) for n in PALM_SENSOR_NAMES])) if len(PALM_SENSOR_NAMES) else 0.0

        # HOLD
        pos_ref = obj_pos.copy()
        stable_steps = 0
        max_speed = 0.0
        max_drift = 0.0

        for _ in range(HOLD_STEPS):
            u = controller_step(t_close=CLOSE_STEPS, theta=theta, latched=latched)
            data.ctrl[:] = u

            energy += float(np.sum(u*u))
            smooth += float(np.sum((u-prev_u)**2))
            prev_u = u

            mujoco.mj_step(model, data)

            p_now = data.xpos[obj_bid]
            v_now = data.cvel[obj_bid][3:6]
            speed = float(np.linalg.norm(v_now))
            drift = float(np.linalg.norm(p_now - pos_ref))

            max_speed = max(max_speed, speed)
            max_drift = max(max_drift, drift)

            # стабильность
            if (p_now[2] > Z_MIN) and (speed < V_MAX) and (drift < DRIFT_MAX):
                stable_steps += 1

            if viewer is not None and do_render:
                render_step(viewer)

        hold_ratio = stable_steps / float(HOLD_STEPS)

        # "реальный" захват: есть хотя бы контакты и хоть какие-то сенсоры
        real_grasp = (contacts >= MIN_CONTACTS_AFTER_CLOSE) and \
                     (tip_active >= MIN_TIP_ACTIVE_AFTER_CLOSE) and \
                     (tip_sum >= MIN_TIP_SUM_AFTER_CLOSE)

        success = 1 if (hold_ratio >= SUCCESS_HOLD_RATIO and real_grasp) else 0

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
    # SCORE FUNCTION
    # -------------------------------
    def score_metrics(m):
        score = 0.0
        score += 2.0 * m["success"]
        score += 1.0 * m["hold_ratio"]

        score += 0.05 * m["contacts"]
        score += 0.03 * m["tip_active"]
        score += 0.005 * m["tip_sum"]

        # LF priority
        lf_touch = m["touch_per_finger"]["LF"]
        score += BONUS_LF_TOUCH * lf_touch
        if lf_touch > 0.0:
            score += BONUS_LF_ACTIVE
        else:
            score -= PENALTY_NO_LF

        score -= LAMBDA_DRIFT  * m["max_drift"]
        score -= LAMBDA_ENERGY * m["energy"]
        score -= LAMBDA_SMOOTH * m["smooth"]
        return float(score)

    # -------------------------------
    # eval_theta: 5 траев на theta (как ты просила)
    # + можно отрендерить только 1-й трай
    # -------------------------------
    def eval_theta(theta, base_seed, viewer=None, render_one=False):
        scores = []
        metrics_pack = []
        for k in range(EVAL_TRIALS_PER_THETA):
            seed = int(base_seed + 1000*k + 17*k*k)
            do_render = (render_one and k == 0 and viewer is not None)
            m = run_trial(theta=theta, seed=seed, viewer=viewer, do_render=do_render)
            metrics_pack.append(m)
            scores.append(score_metrics(m))
        return float(np.mean(scores)), metrics_pack

    # -------------------------------
    # BASELINES + RANDOM SEARCH (как в твоём коде)
    # -------------------------------
    rng = np.random.default_rng(123)

    baseline_A = np.array([1.0, 1.0, 1.0, max(1.0, LF_MIN), 1.0, 0.010, 0.30], dtype=float)
    baseline_B = np.array([1.2, 0.9, 0.9, max(1.15, LF_MIN), 1.4, 0.010, 0.30], dtype=float)

    # ---------- VIEWER STARTS IMMEDIATELY ----------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        p("Viewer opened.")

        # прогрев окна
        for _ in range(20):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.003)

        p("--- Evaluate baselines ---")
        sA, mA = eval_theta(baseline_A, SEED0 + 10000, viewer=viewer, render_one=RENDER_BASELINES)
        sB, mB = eval_theta(baseline_B, SEED0 + 20000, viewer=viewer, render_one=RENDER_BASELINES)
        p(f"Baseline A score={sA:.4f} theta={baseline_A}")
        p(f"Baseline B score={sB:.4f} theta={baseline_B}")

        best_theta = baseline_A.copy()
        best_score = sA
        best_pack  = mA
        if sB > best_score:
            best_theta, best_score, best_pack = baseline_B.copy(), sB, mB

        p(f"--- Random search (iters={RANDOM_ITERS}, trials/theta={EVAL_TRIALS_PER_THETA}) ---")
        for it in range(1, RANDOM_ITERS + 1):
            w = rng.uniform(W_RANGE[0], W_RANGE[1], size=5)
            w[3] = rng.uniform(LF_RANGE[0], LF_RANGE[1])  # wLF
            if w[3] < LF_MIN:
                continue

            th_touch = rng.uniform(TH_TOUCH_RANGE[0], TH_TOUCH_RANGE[1])
            k_hold   = rng.uniform(K_HOLD_RANGE[0], K_HOLD_RANGE[1])
            theta = np.array([*w, th_touch, k_hold], dtype=float)

            render_now = (it % RENDER_EVERY_N_THETA == 0)
            s, pack = eval_theta(theta, SEED0 + 30000 + it*100, viewer=viewer, render_one=render_now)

            if s > best_score:
                best_theta, best_score, best_pack = theta, s, pack
                p(f"[NEW BEST] it={it:03d} score={best_score:.4f} theta={best_theta}")

                # сразу показать новый best одним трейлом
                _ = eval_theta(best_theta, SEED0 + 900000 + it, viewer=viewer, render_one=True)

        p("==============================")
        p(" BEST RESULT")
        p("==============================")
        p(f"best_score={best_score:.4f}")
        p(f"best_theta=[wFF,wMF,wRF,wLF,wTH, th_touch, k_hold]={best_theta}")

        # -------------------------------
        # SHOW BEST IN VIEWER (5 trials визуально)
        # -------------------------------
        p(f"--- Visualize best_theta ({VISUAL_TRIALS} trials) ---")
        all_visual = []
        for t in range(1, VISUAL_TRIALS + 1):
            m = run_trial(best_theta, seed=SEED0 + 50000 + t, viewer=viewer, do_render=True)
            all_visual.append(m)

            tp = m["touch_per_finger"]
            p(f"TRIAL {t}: SUCCESS={m['success']} hold={m['hold_ratio']:.2f} con={m['contacts']} "
              f"z={m['z']:.3f} drift={m['max_drift']:.3f} speed={m['max_speed']:.3f} "
              f"tip_active={m['tip_active']} tip_sum={m['tip_sum']:.3f} "
              f"FF={tp['FF']:.3f} MF={tp['MF']:.3f} RF={tp['RF']:.3f} LF={tp['LF']:.3f} TH={tp['TH']:.3f}")

        succ_rate = float(np.mean([m["success"] for m in all_visual]))
        avg_contacts = float(np.mean([m["contacts"] for m in all_visual]))
        avg_hold = float(np.mean([m["hold_ratio"] for m in all_visual]))
        p("==============================")
        p(" VISUAL SUMMARY")
        p("==============================")
        p(f"Success rate: {succ_rate:.2f} | Avg contacts: {avg_contacts:.2f} | Avg hold_ratio: {avg_hold:.2f}")

        p("Close the viewer window to continue to next XML.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.02)

    return {
        "xml": XML_PATH,
        "best_score": float(best_score),
        "best_theta": best_theta.copy(),
    }


def main():
    results = []
    for xml in XML_LIST:
        results.append(run_for_xml(xml))

    print("\n" + "="*94)
    print("SUMMARY (best per XML)")
    print("="*94)
    for r in results:
        tag = os.path.basename(r["xml"])
        print(f"[{tag}] best_score={r['best_score']:.4f} best_theta={r['best_theta']}")

    best = max(results, key=lambda x: x["best_score"])
    print("\nBest overall:", os.path.basename(best["xml"]), "score:", best["best_score"])


if __name__ == "__main__":
    main()
