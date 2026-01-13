import os, time
import numpy as np
import mujoco
import mujoco.viewer


ROOT_DIR = r"C:\Users\rad\itmo\new_roms"

XML_LIST = [
    os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_block_touch_sensors.xml"),
    os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_egg_touch_sensors.xml"),
    os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_pen_touch_sensors.xml"),
    os.path.join(ROOT_DIR, "project", "models", "hand", "hand_manipulate_clock.xml"),
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

MIN_CONTACTS_AFTER_CLOSE = 4
MIN_CONTACTS_DURING_HOLD = 1
MIN_CONTACT_HOLD_RATIO   = 0.80

MIN_TIP_ACTIVE_AFTER_CLOSE = 1
MIN_TIP_SUM_AFTER_CLOSE    = 0.50

USE_PERTURB = True
PERTURB_F = 1.5
PERTURB_PERIOD = 20

SLIP_CONTACTS_TH = 2
GRIP_BOOST_WHEN_SLIP = 0.25

SLIP_SPEED_TH = 0.35
SLIP_DRIFT_TH = 0.06
GRIP_BOOST_WHEN_SLIP_STRONG = 0.45

EVAL_TRIALS_PER_THETA = 5
RANDOM_ITERS = 220
SEED0 = 2000

LAMBDA_ENERGY = 1e-4
LAMBDA_SMOOTH = 1e-3

TH_TOUCH_RANGE = (0.001, 0.05)
K_HOLD_RANGE   = (0.10, 0.75)
W_RANGE        = (0.5, 1.6)

FF_MIN = 0.90

LF_MIN = 1.10
LF_RANGE = (1.10, 2.00)
BONUS_LF_TOUCH = 0.02
BONUS_LF_ACTIVE = 0.08
PENALTY_NO_LF = 0.15

# -------------------------------
# RENDER SETTINGS
# -------------------------------
RENDER_BASELINES = True
RENDER_EVERY_N_THETA = 12      # рендерить каждый N-й theta (первый трай в eval), чтобы не тормозить поиск
KEEP_VIEWER_OPEN_AFTER_XML = False

# -------------------------------
# SLOWDOWN (ВАЖНО)
# -------------------------------
SLOWDOWN = 2.0  # 2x медленнее (только при рендере)


def run_for_xml(XML_PATH: str):
    xml_tag = os.path.basename(XML_PATH)

    def p(msg: str):
        print(f"[{xml_tag}] {msg}")

    print("\n" + "="*90)
    p(f"XML: {XML_PATH}")
    p(f"Exists: {os.path.exists(XML_PATH)}")
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(XML_PATH)

    # --- load model/data
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    DT = float(model.opt.timestep)
    EXTRA_SLEEP_PER_STEP = DT * (SLOWDOWN - 1.0)  # добавка к “реальному” времени на каждый шаг при рендере

    p(f"Scene loaded. Bodies: {model.nbody} Joints: {model.njnt} Actuators: {model.nu} Sensors: {model.nsensor}")
    p(f"timestep (DT): {DT:.6f} | SLOWDOWN: {SLOWDOWN:.2f} | extra_sleep/step: {EXTRA_SLEEP_PER_STEP:.6f}")

    # --- object ids
    OBJ_BODY_NAME  = "object"
    OBJ_JOINT_NAME = "object:joint"

    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJ_BODY_NAME)
    if obj_bid < 0:
        raise RuntimeError(f"[{xml_tag}] Body '{OBJ_BODY_NAME}' not found in XML.")

    obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJ_JOINT_NAME)
    if obj_jid < 0:
        raise RuntimeError(f"[{xml_tag}] Joint '{OBJ_JOINT_NAME}' not found. Check your XML joint name.")

    obj_qpos_adr = int(model.jnt_qposadr[obj_jid])

    # --- enable object collisions
    obj_geom_ids = np.where(model.geom_bodyid == obj_bid)[0].astype(int).tolist()
    p(f"Object geoms: {len(obj_geom_ids)}")
    if len(obj_geom_ids) == 0:
        p("[WARN] Object body has 0 geoms. Contacts may stay 0.")

    for gid in obj_geom_ids:
        model.geom_contype[gid] = 1
        model.geom_conaffinity[gid] = 1
    for gid in range(model.ngeom):
        model.geom_conaffinity[gid] |= 1

    # --- sensor slices
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

    # --- fingertip touch mapping
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

    def count_object_contacts():
        cnt = 0
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[int(c.geom1)])
            b2 = int(model.geom_bodyid[int(c.geom2)])
            if (b1 == obj_bid and b2 != obj_bid) or (b2 == obj_bid and b1 != obj_bid):
                cnt += 1
        return cnt

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

    def render_step_if_needed(viewer, do_render: bool):
        if viewer is None or not do_render:
            return
        viewer.sync()
        if EXTRA_SLEEP_PER_STEP > 0.0:
            time.sleep(EXTRA_SLEEP_PER_STEP)

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
            render_step_if_needed(viewer, do_render)

        # CLOSE
        last_touch = {f: 0.0 for f in FINGERS}
        for t in range(1, CLOSE_STEPS+1):
            u = controller_step(t_close=t, theta=theta, latched=latched)
            data.ctrl[:] = u

            for f in FINGERS:
                last_touch[f] = sensor_sum(FINGERTIP_TOUCH[f]) if touch_present[f] else 0.0

            energy += float(np.sum(u*u))
            smooth += float(np.sum((u-prev_u)**2))
            prev_u = u

            mujoco.mj_step(model, data)
            render_step_if_needed(viewer, do_render)

        contacts_after_close = int(count_object_contacts())
        obj_pos = data.xpos[obj_bid].copy()
        z = float(obj_pos[2])

        tip_vals = np.array([last_touch[f] for f in FINGERS], dtype=float)
        tip_sum = float(np.sum(tip_vals))
        tip_active = int(np.sum(tip_vals > 0.0))

        palm_sum = float(np.sum([sensor_sum(n) for n in PALM_SENSOR_NAMES])) if len(PALM_SENSOR_NAMES) else 0.0
        real_finger_grasp = (tip_active >= MIN_TIP_ACTIVE_AFTER_CLOSE) and (tip_sum >= MIN_TIP_SUM_AFTER_CLOSE)

        pos_ref = obj_pos.copy()
        stable_steps = 0
        hold_contact_steps = 0
        max_speed = 0.0
        max_drift = 0.0

        # HOLD
        for t_hold in range(HOLD_STEPS):
            u = controller_step(t_close=CLOSE_STEPS, theta=theta, latched=latched)

            cnow_pre = int(count_object_contacts())
            if cnow_pre < SLIP_CONTACTS_TH:
                u = np.clip(u * (1.0 + GRIP_BOOST_WHEN_SLIP), 0.0, 1.0)

            p_pre = data.xpos[obj_bid].copy()
            v_pre = data.cvel[obj_bid][3:6].copy()
            speed_pre = float(np.linalg.norm(v_pre))
            drift_pre = float(np.linalg.norm(p_pre - pos_ref))
            if (speed_pre > SLIP_SPEED_TH) or (drift_pre > SLIP_DRIFT_TH):
                u = np.clip(u * (1.0 + GRIP_BOOST_WHEN_SLIP_STRONG), 0.0, 1.0)

            data.ctrl[:] = u

            energy += float(np.sum(u*u))
            smooth += float(np.sum((u-prev_u)**2))
            prev_u = u

            if USE_PERTURB:
                data.xfrc_applied[obj_bid, :3] = 0.0
                data.xfrc_applied[obj_bid, 3:] = 0.0
                sgn = 1.0 if ((t_hold // PERTURB_PERIOD) % 2 == 0) else -1.0
                data.xfrc_applied[obj_bid, 0] = sgn * PERTURB_F

            mujoco.mj_step(model, data)

            cnow = int(count_object_contacts())
            if cnow >= MIN_CONTACTS_DURING_HOLD:
                hold_contact_steps += 1

            p_now = data.xpos[obj_bid]
            v_now = data.cvel[obj_bid][3:6]
            speed = float(np.linalg.norm(v_now))
            drift = float(np.linalg.norm(p_now - pos_ref))

            max_speed = max(max_speed, speed)
            max_drift = max(max_drift, drift)

            if (p_now[2] > Z_MIN) and (speed < V_MAX) and (drift < DRIFT_MAX) and (cnow >= MIN_CONTACTS_DURING_HOLD):
                stable_steps += 1

            render_step_if_needed(viewer, do_render)

        hold_ratio = stable_steps / float(HOLD_STEPS)
        contact_hold_ratio = hold_contact_steps / float(HOLD_STEPS)

        success = 1 if (
            (hold_ratio >= SUCCESS_HOLD_RATIO) and
            (contacts_after_close >= MIN_CONTACTS_AFTER_CLOSE) and
            (contact_hold_ratio >= MIN_CONTACT_HOLD_RATIO) and
            real_finger_grasp
        ) else 0

        return {
            "contacts": int(contacts_after_close),
            "contact_hold_ratio": float(contact_hold_ratio),
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

    def score_metrics(m):
        score = 0.0
        score += 2.0 * m["success"]
        score += 1.0 * m["hold_ratio"]

        score += 0.08 * m["contacts"]
        score += 1.00 * m.get("contact_hold_ratio", 0.0)
        score += 0.05 * m["tip_active"]
        score += 0.01 * m["tip_sum"]

        tp = m["touch_per_finger"]
        if tp["TH"] > 0.0:
            score += 0.20
        else:
            score -= 0.15
        if tp["FF"] > 0.0:
            score += 0.15

        lf_touch = tp["LF"]
        score += BONUS_LF_TOUCH * lf_touch
        if lf_touch > 0.0:
            score += BONUS_LF_ACTIVE
        else:
            score -= PENALTY_NO_LF

        if (m["tip_active"] < MIN_TIP_ACTIVE_AFTER_CLOSE) or (m["tip_sum"] < MIN_TIP_SUM_AFTER_CLOSE):
            score -= 1.5

        if m["contacts"] < MIN_CONTACTS_AFTER_CLOSE:
            score -= 1.0

        score -= 2.00 * m["max_drift"]
        score -= 0.30 * m["max_speed"]
        score -= LAMBDA_ENERGY * m["energy"]
        score -= LAMBDA_SMOOTH * m["smooth"]
        return float(score)

    # ---------- START VIEWER IMMEDIATELY ----------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        p("Viewer opened (simulation starts immediately).")
        for _ in range(30):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.005)

        theta_eval_counter = 0

        def eval_theta(theta, base_seed, render_this_theta=False):
            scores = []
            succ = 0

            for k in range(EVAL_TRIALS_PER_THETA):
                seed = int(base_seed + 1000*k + 17*k*k)
                do_render = (k == 0) and render_this_theta  # только первый trial рисуем
                m = run_trial(theta=theta, seed=seed, viewer=viewer, do_render=do_render)
                scores.append(score_metrics(m))
                succ += int(m["success"])

            scores = np.array(scores, dtype=float)
            q20 = float(np.quantile(scores, 0.20))
            mean = float(np.mean(scores))

            fail_penalty = 0.0
            if succ < EVAL_TRIALS_PER_THETA:
                fail_penalty = 0.6 * (EVAL_TRIALS_PER_THETA - succ)

            final = q20 + 0.20 * mean - fail_penalty
            return final

        rng = np.random.default_rng(123)

        baseline_A = np.array([1.0, 1.0, 1.0, max(1.0, LF_MIN), 1.0, 0.010, 0.30], dtype=float)
        baseline_B = np.array([1.2, 0.9, 0.9, max(1.15, LF_MIN), 1.4, 0.010, 0.30], dtype=float)

        p("--- Evaluate baselines ---")
        sA = eval_theta(baseline_A, SEED0 + 10000, render_this_theta=RENDER_BASELINES)
        sB = eval_theta(baseline_B, SEED0 + 20000, render_this_theta=RENDER_BASELINES)
        p(f"Baseline A score: {sA:.4f} theta: {baseline_A}")
        p(f"Baseline B score: {sB:.4f} theta: {baseline_B}")

        best_theta = baseline_A.copy()
        best_score = sA
        if sB > best_score:
            best_theta, best_score = baseline_B.copy(), sB

        p("--- Random search ---")
        for it in range(1, RANDOM_ITERS + 1):
            if it <= int(0.6 * RANDOM_ITERS):
                w = rng.uniform(W_RANGE[0], W_RANGE[1], size=5)
                w[0] = max(w[0], FF_MIN)
                w[3] = rng.uniform(LF_RANGE[0], LF_RANGE[1])
                if w[3] < LF_MIN:
                    continue
                th_touch = rng.uniform(TH_TOUCH_RANGE[0], TH_TOUCH_RANGE[1])
                k_hold   = rng.uniform(K_HOLD_RANGE[0], K_HOLD_RANGE[1])
                theta = np.array([*w, th_touch, k_hold], dtype=float)
            else:
                theta = best_theta.copy()
                theta[:5] += rng.normal(0.0, 0.12, size=5)
                theta[5]  += rng.normal(0.0, 0.006)
                theta[6]  += rng.normal(0.0, 0.05)

                theta[:5] = np.clip(theta[:5], W_RANGE[0], W_RANGE[1])
                theta[0]  = max(theta[0], FF_MIN)
                theta[3]  = np.clip(theta[3], LF_RANGE[0], LF_RANGE[1])
                theta[5]  = float(np.clip(theta[5], TH_TOUCH_RANGE[0], TH_TOUCH_RANGE[1]))
                theta[6]  = float(np.clip(theta[6], K_HOLD_RANGE[0], K_HOLD_RANGE[1]))

            theta_eval_counter += 1
            render_now = (theta_eval_counter % RENDER_EVERY_N_THETA == 0)

            s = eval_theta(theta, SEED0 + 30000 + it*100, render_this_theta=render_now)

            if s > best_score:
                best_theta, best_score = theta, s
                p(f"[NEW BEST] it={it:03d} score={best_score:.4f} theta={best_theta}")

                # показать новый best визуально (медленнее в 2 раза)
                _ = eval_theta(best_theta, SEED0 + 900000 + it, render_this_theta=True)

        p("BEST for this XML:")
        p(f"best_score: {best_score:.4f}")
        p(f"best_theta [wFF,wMF,wRF,wLF,wTH, th_touch, k_hold] = {best_theta}")

        if KEEP_VIEWER_OPEN_AFTER_XML:
            p("KEEP_VIEWER_OPEN_AFTER_XML=True -> окно будет открыто, закрывай вручную чтобы продолжить.")
            while viewer.is_running():
                data.ctrl[:] = 0.0
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
    for pth in XML_LIST:
        res = run_for_xml(pth)
        results.append(res)

    print("\n" + "="*90)
    print("SUMMARY (best per XML)")
    print("="*90)
    for r in results:
        tag = os.path.basename(r["xml"])
        print(f"[{tag}]  score={r['best_score']:.4f}  theta={r['best_theta']}")

    best = max(results, key=lambda x: x["best_score"])
    print("\nBest overall:", os.path.basename(best["xml"]), "score:", best["best_score"])


if __name__ == "__main__":
    main()
