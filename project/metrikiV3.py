import os, time
import numpy as np
import mujoco
import mujoco.viewer


# -------------------------------
# PATHS
# -------------------------------
ROOT_DIR = r"C:\Users\rad\itmo\new_roms"

XML_LIST = [
    # os.path.join(ROOT_DIR, "project", "models", "hand", "hand_manipulate_clock.xml"),
    # os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_block_touch_sensors.xml"),
    os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_egg_touch_sensors.xml"),
    # os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_pen_touch_sensors.xml"),
]

# -------------------------------
# EPISODE CONFIG
# -------------------------------
OPEN_STEPS  = 40
CLOSE_STEPS = 120
HOLD_STEPS  = 200
Z_START_CLOSE = 0.22
MAX_OPEN_STEPS = 200

MIN_CLOSE_STEPS_BEFORE_LATCH = 15

Z_MIN      = 0.05
V_MAX      = 2.0
DRIFT_MAX  = 0.25
SUCCESS_HOLD_RATIO = 0.90

# ---- NEW: settle window (не штрафуем первые шаги HOLD) ----
SETTLE_STEPS = 40  # первые шаги HOLD игнорируем в p95/ratio

# ---- NEW: thresholds for "real grasp" in HOLD ----
MIN_TIP_ACTIVE_HOLD = 2          # хотим >=2 пальцев
MIN_TIP_SUM_HOLD_AVG = 0.25      # средний tip_sum в HOLD
TIP_CONTACT_THR = 0.05           # порог "есть контакт" для streak
MIN_TIP_STREAK = 15              # мин. длина непрерывного контакта (шаги)

# ---- NEW: contact thresholds ----
MIN_CONTACTS_HAND = 3            # контакты именно с рукой
MAX_CONTACTS_ENV  = 0            # запрет "подпора" окружением (можно сделать 1)

# -------------------------------
# OPTIMIZER CONFIG
# -------------------------------
EVAL_TRIALS_PER_THETA = 5
RANDOM_ITERS = 25
SEED0 = 2000

# -------------------------------
# SCORE WEIGHTS
# -------------------------------
LAMBDA_ENERGY = 1e-4
LAMBDA_SMOOTH = 1e-3
LAMBDA_DRIFT  = 1.0

# NEW weights (robust metrics)
LAMBDA_SPEED_P95 = 0.4
LAMBDA_DRIFT_P95 = 1.2
LAMBDA_BAD_SPEED = 0.6
LAMBDA_BAD_DRIFT = 0.8

BONUS_STREAK     = 0.02
BONUS_OPPOSITION = 0.25

PENALTY_ENV_CONTACT = 0.25  # штраф за контакты с "не рукой"

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
SLOWDOWN = 2.0
RENDER_BASELINES = True
RENDER_EVERY_N_THETA = 10
VISUAL_TRIALS = 5


def run_for_xml(XML_PATH: str):
    xml_tag = os.path.basename(XML_PATH)

    def p(msg: str):
        print(f"[{xml_tag}] {msg}")

    print("\n" + "=" * 94)
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
        return float(np.sum(data.sensordata[st:st + dim]))

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
    # CONTACT COUNT (split hand/env)
    # -------------------------------
    def is_hand_body(bid: int) -> bool:
        """
        Эвристика: у Shadow Hand тела обычно содержат 'robot0'.
        Если у тебя есть части руки без 'robot0' в имени, добавь сюда условия.
        """
        name = model.body(bid).name
        if not name:
            return False
        if ("robot0" in name) or name.startswith("robot0:"):
            return True
        # иногда полезно считать предплечье частью руки:
        if "forearm" in name.lower() or "hand" in name.lower() or "wrist" in name.lower():
            return True
        return False

    def count_object_contacts_split():
        """Возвращает (contacts_hand, contacts_env, contacts_total)"""
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
    # RANDOMIZE OBJECT START (XY + yaw)
    # -------------------------------
    def set_object_pose_random(seed: int, xy_sigma=0.015, yaw_range=np.deg2rad(25)):
        rng = np.random.default_rng(seed)
        qpos = data.qpos

        base_pos  = qpos[obj_qpos_adr:obj_qpos_adr + 3].copy()
        base_quat = qpos[obj_qpos_adr + 3:obj_qpos_adr + 7].copy()

        dx, dy = rng.normal(0, xy_sigma, size=2)
        pos = base_pos.copy()
        pos[0] += dx
        pos[1] += dy

        yaw = float(rng.uniform(-yaw_range, yaw_range))
        q_yaw = np.zeros(4, dtype=float)
        mujoco.mju_axisAngle2Quat(q_yaw, np.array([0.0, 0.0, 1.0], dtype=float), yaw)

        q = np.zeros(4, dtype=float)
        mujoco.mju_mulQuat(q, q_yaw, base_quat)

        qpos[obj_qpos_adr:obj_qpos_adr + 3] = pos
        qpos[obj_qpos_adr + 3:obj_qpos_adr + 7] = q

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
    # RUN ONE TRIAL (with NEW metrics)
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

        # OPEN (ждём пока объект опустится до Z_START_CLOSE или пока не истечёт лимит)
        for _ in range(MAX_OPEN_STEPS):
            u = np.zeros(model.nu, dtype=float)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

            z_obj = float(data.xpos[obj_bid][2])
            if z_obj <= Z_START_CLOSE:
                break

            if viewer is not None and do_render:
                render_step(viewer)

        # CLOSE
        last_touch = {f: 0.0 for f in FINGERS}
        for t in range(1, CLOSE_STEPS + 1):
            u = controller_step(t_close=t, theta=theta, latched=latched)
            data.ctrl[:] = u

            for f in FINGERS:
                last_touch[f] = sensor_sum(FINGERTIP_TOUCH[f]) if touch_present[f] else 0.0

            energy += float(np.sum(u * u))
            smooth += float(np.sum((u - prev_u) ** 2))
            prev_u = u

            mujoco.mj_step(model, data)
            if viewer is not None and do_render:
                render_step(viewer)

        contacts_hand, contacts_env, contacts_total = count_object_contacts_split()

        obj_pos = data.xpos[obj_bid].copy()
        z = float(obj_pos[2])

        tip_vals = np.array([last_touch[f] for f in FINGERS], dtype=float)
        tip_sum = float(np.sum(tip_vals))
        tip_active = int(np.sum(tip_vals > 0.0))

        palm_sum = float(np.sum([sensor_sum(n) for n in PALM_SENSOR_NAMES])) if len(PALM_SENSOR_NAMES) else 0.0

        # HOLD (NEW robust metrics)
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

        for t_hold in range(HOLD_STEPS):
            u = controller_step(t_close=CLOSE_STEPS, theta=theta, latched=latched)
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

            # streak of meaningful contact
            if ts > TIP_CONTACT_THR:
                streak += 1
                tip_streak_max = max(tip_streak_max, streak)
            else:
                streak = 0

            # thumb opposition: TH + any other finger simultaneously
            if (touch_now["TH"] > 0.0) and (
                (touch_now["FF"] > 0.0) or (touch_now["MF"] > 0.0) or
                (touch_now["RF"] > 0.0) or (touch_now["LF"] > 0.0)
            ):
                opposition_seen = True

            if viewer is not None and do_render:
                render_step(viewer)

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

        # NEW "real grasp" + SUCCESS
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
            # legacy fields (чтобы старые таблицы не ломались)
            "contacts": int(contacts_total),
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

            # NEW fields
            "contacts_hand": int(contacts_hand),
            "contacts_env": int(contacts_env),
            "speed_p95": float(speed_p95),
            "drift_p95": float(drift_p95),
            "bad_speed_ratio": float(bad_speed_ratio),
            "bad_drift_ratio": float(bad_drift_ratio),
            "tip_sum_hold_avg": float(tip_sum_hold_avg),
            "tip_active_hold_avg": float(tip_active_hold_avg),
            "tip_streak_max": int(tip_streak_max),
            "thumb_opposition": int(thumb_opposition),
        }

    # -------------------------------
    # SCORE FUNCTION (UPDATED)
    # -------------------------------
    def score_metrics(m):
        score = 0.0

        # главный сигнал: success (по новым правилам)
        score += 3.0 * m["success"]

        # стабильность всё ещё важна
        score += 1.2 * m["hold_ratio"]

        # контакты именно с рукой (а не с окружением)
        score += 0.06 * m["contacts_hand"]
        score -= PENALTY_ENV_CONTACT * m["contacts_env"]

        # робастные метрики (после settle)
        score -= LAMBDA_DRIFT_P95 * m["drift_p95"]
        score -= LAMBDA_SPEED_P95 * m["speed_p95"]
        score -= LAMBDA_BAD_DRIFT * m["bad_drift_ratio"]
        score -= LAMBDA_BAD_SPEED * m["bad_speed_ratio"]

        # качество контакта в HOLD
        score += 0.01 * m["tip_sum_hold_avg"]
        score += BONUS_STREAK * (m["tip_streak_max"] / max(1.0, HOLD_STEPS))
        score += BONUS_OPPOSITION * (1.0 if m["thumb_opposition"] else 0.0)

        # LF priority (оставляем твою идею)
        lf_touch = m["touch_per_finger"]["LF"]
        score += BONUS_LF_TOUCH * lf_touch
        if lf_touch > 0.0:
            score += BONUS_LF_ACTIVE
        else:
            score -= PENALTY_NO_LF

        # штрафы на управление
        score -= LAMBDA_ENERGY * m["energy"]
        score -= LAMBDA_SMOOTH * m["smooth"]

        return float(score)

    # -------------------------------
    # eval_theta: 5 траев на theta
    # -------------------------------
    def eval_theta(theta, base_seed, viewer=None, render_one=False):
        scores = []
        metrics_pack = []
        for k in range(EVAL_TRIALS_PER_THETA):
            seed = int(base_seed + 1000 * k + 17 * k * k)
            do_render = (render_one and k == 0 and viewer is not None)
            m = run_trial(theta=theta, seed=seed, viewer=viewer, do_render=do_render)
            metrics_pack.append(m)
            scores.append(score_metrics(m))
        return float(np.mean(scores)), metrics_pack

    # -------------------------------
    # BASELINES + RANDOM SEARCH
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
            time.sleep(1)

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
            s, pack = eval_theta(theta, SEED0 + 30000 + it * 100, viewer=viewer, render_one=render_now)

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
        # SHOW BEST IN VIEWER (5 trials)
        # -------------------------------
        p(f"--- Visualize best_theta ({VISUAL_TRIALS} trials) ---")
        all_visual = []
        for t in range(1, VISUAL_TRIALS + 1):
            m = run_trial(best_theta, seed=SEED0 + 50000 + t, viewer=viewer, do_render=True)
            all_visual.append(m)

            tp = m["touch_per_finger"]
            p(
                f"TRIAL {t}: SUCCESS={m['success']} hold={m['hold_ratio']:.2f} "
                f"con_hand={m['contacts_hand']} con_env={m['contacts_env']} con={m['contacts']} "
                f"z={m['z']:.3f} driftP95={m['drift_p95']:.3f} speedP95={m['speed_p95']:.3f} "
                f"bad_drift={m['bad_drift_ratio']:.2f} bad_speed={m['bad_speed_ratio']:.2f} "
                f"tip_active={m['tip_active']} tip_sum={m['tip_sum']:.3f} "
                f"hold_tip_avg={m['tip_sum_hold_avg']:.3f} streak={m['tip_streak_max']} opp={m['thumb_opposition']} "
                f"FF={tp['FF']:.3f} MF={tp['MF']:.3f} RF={tp['RF']:.3f} LF={tp['LF']:.3f} TH={tp['TH']:.3f}"
            )

        succ_rate = float(np.mean([m["success"] for m in all_visual]))
        avg_contacts_hand = float(np.mean([m["contacts_hand"] for m in all_visual]))
        avg_hold = float(np.mean([m["hold_ratio"] for m in all_visual]))
        avg_drift_p95 = float(np.mean([m["drift_p95"] for m in all_visual]))
        avg_speed_p95 = float(np.mean([m["speed_p95"] for m in all_visual]))

        p("==============================")
        p(" VISUAL SUMMARY")
        p("==============================")
        p(
            f"Success rate: {succ_rate:.2f} | Avg hand contacts: {avg_contacts_hand:.2f} | "
            f"Avg hold_ratio: {avg_hold:.2f} | Avg driftP95: {avg_drift_p95:.3f} | Avg speedP95: {avg_speed_p95:.3f}"
        )

        p("Close the viewer window to continue to next XML.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(3)

    return {
        "xml": XML_PATH,
        "best_score": float(best_score),
        "best_theta": best_theta.copy(),
    }


def main():
    results = []
    for xml in XML_LIST:
        results.append(run_for_xml(xml))

    print("\n" + "=" * 94)
    print("SUMMARY (best per XML)")
    print("=" * 94)
    for r in results:
        tag = os.path.basename(r["xml"])
        print(f"[{tag}] best_score={r['best_score']:.4f} best_theta={r['best_theta']}")

    best = max(results, key=lambda x: x["best_score"])
    print("\nBest overall:", os.path.basename(best["xml"]), "score:", best["best_score"])


if __name__ == "__main__":
    main()
