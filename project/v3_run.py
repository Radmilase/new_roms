

import os, time
import numpy as np
import mujoco
import mujoco.viewer

ROOT_DIR = r"C:\Users\rad\itmo\new_roms"

XML_LIST = [
    # os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_block_touch_sensors.xml"),
    # os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_egg_touch_sensors.xml"),
    os.path.join(ROOT_DIR, "project", "models", "hand", "manipulate_pen_touch_sensors.xml"),
]


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


MIN_CONTACTS_AFTER_CLOSE = 3
MIN_TIP_ACTIVE_AFTER_CLOSE = 1
MIN_TIP_SUM_AFTER_CLOSE    = 0.25

# CGF thresholding
CGF_MIN_TIP_ACTIVE = 2
CGF_TAU_TIP_SUM    = 0.05   # порог "суммарного контакта" для grasp
CGF_STREAK_MIN     = 10     # минимальная длина непрерывного grasp, чтобы считать CGF_stable=1

# For "bad drift / bad speed" counters (diagnostic)
DRIFT_BAD_THR = 0.10
SPEED_BAD_THR = 0.80

EVAL_TRIALS_PER_THETA = 5
RANDOM_ITERS = 25
SEED0 = 2000


LAMBDA_ENERGY = 1e-4
LAMBDA_SMOOTH = 1e-3
LAMBDA_DRIFT  = 1.0

TH_TOUCH_RANGE = (0.001, 0.05)
K_HOLD_RANGE   = (0.10, 0.60)
W_RANGE        = (0.5, 1.5)

LF_MIN = 1.10
LF_RANGE = (1.10, 1.80)
BONUS_LF_TOUCH  = 0.02
BONUS_LF_ACTIVE = 0.08
PENALTY_NO_LF   = 0.15


SLOWDOWN = 2.0
RENDER_BASELINES = True
RENDER_EVERY_N_THETA = 10
VISUAL_TRIALS = 5


def run_for_xml(XML_PATH: str):
    xml_tag = os.path.basename(XML_PATH)

    def p(msg: str):
        print(f"[{xml_tag}] {msg}")

    print("\n" + "="*94)
    p(f"XML: {XML_PATH}")
    p(f"Exists: {os.path.exists(XML_PATH)}")
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(XML_PATH)


    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    DT = float(model.opt.timestep)
    EXTRA_SLEEP_PER_STEP = max(0.0, DT * (SLOWDOWN - 1.0))

    p("Scene loaded successfully")
    p(f"Bodies={model.nbody} Joints={model.njnt} Actuators={model.nu} Sensors={model.nsensor} timestep={DT}")

    OBJ_BODY_NAME  = "object"
    OBJ_JOINT_NAME = "object:joint"

    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJ_BODY_NAME)
    if obj_bid < 0:
        raise RuntimeError(f"[{xml_tag}] Body '{OBJ_BODY_NAME}' not found in XML.")
    obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJ_JOINT_NAME)
    if obj_jid < 0:
        raise RuntimeError(f"[{xml_tag}] Joint '{OBJ_JOINT_NAME}' not found. Check your XML joint name.")
    obj_qpos_adr = int(model.jnt_qposadr[obj_jid])  


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


    HAND_BODY_KEYWORDS = ("robot0", "hand", "palm", "finger", "thumb", "ff", "mf", "rf", "lf", "th", "wrist", "forearm")

    def is_hand_body_id(bid: int) -> bool:
        name = (model.body(bid).name or "").lower()
        return any(k in name for k in HAND_BODY_KEYWORDS)

    def count_object_contacts_split():
        con_total = 0
        con_hand  = 0
        con_env   = 0
        for i in range(data.ncon):
            c = data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            b1 = int(model.geom_bodyid[g1])
            b2 = int(model.geom_bodyid[g2])

            if (b1 == obj_bid and b2 != obj_bid):
                con_total += 1
                if is_hand_body_id(b2): con_hand += 1
                else: con_env += 1
            elif (b2 == obj_bid and b1 != obj_bid):
                con_total += 1
                if is_hand_body_id(b1): con_hand += 1
                else: con_env += 1

        return con_total, con_hand, con_env


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

    def render_step(viewer):
        viewer.sync()
        if EXTRA_SLEEP_PER_STEP > 0:
            time.sleep(EXTRA_SLEEP_PER_STEP)

    def contact_entropy(touch_map: dict, eps: float = 1e-12) -> float:
        vec = np.array([max(0.0, float(touch_map[f])) for f in FINGERS], dtype=float)
        s = float(np.sum(vec))
        if s <= eps:
            return 0.0
        pvec = vec / s
        return float(-np.sum(pvec * np.log(pvec + eps)))


    def run_trial(theta, seed, viewer=None, do_render=False):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        set_object_pose_random(seed)
        mujoco.mj_forward(model, data)

        latched = {f: False for f in FINGERS}
        energy = 0.0
        smooth = 0.0
        prev_u = np.zeros(model.nu, dtype=float)


        for _ in range(MAX_OPEN_STEPS):
            u = np.zeros(model.nu, dtype=float)
            data.ctrl[:] = u

            energy += float(np.sum(u*u))
            smooth += float(np.sum((u-prev_u)**2))
            prev_u = u

            mujoco.mj_step(model, data)
            z_obj = float(data.xpos[obj_bid][2])
            if z_obj <= Z_START_CLOSE:
                break

            if viewer is not None and do_render:
                render_step(viewer)

   
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

      
        con_total0, con_hand0, con_env0 = count_object_contacts_split()
        obj_pos = data.xpos[obj_bid].copy()
        z = float(obj_pos[2])

        tip_vals = np.array([last_touch[f] for f in FINGERS], dtype=float)
        tip_sum = float(np.sum(tip_vals))
        tip_active = int(np.sum(tip_vals > 0.0))

        palm_sum = float(np.sum([sensor_sum(n) for n in PALM_SENSOR_NAMES])) if len(PALM_SENSOR_NAMES) else 0.0

        pos_ref = obj_pos.copy()
        stable_steps = 0
        max_speed = 0.0
        max_drift = 0.0

        # new accumulators (HOLD)
        hold_con_total = 0
        hold_con_hand  = 0
        hold_con_env   = 0

        hold_tip_sums = []

        # CGF / GD + grasp-only slices
        cgf_streak = 0
        gd_max = 0
        drift_grasp = []
        speed_grasp = []
        tip_sum_grasp = []
        ce_grasp = []

        bad_drift_cnt = 0
        bad_speed_cnt = 0

        for _ in range(HOLD_STEPS):
            u = controller_step(t_close=CLOSE_STEPS, theta=theta, latched=latched)
            data.ctrl[:] = u

            energy += float(np.sum(u*u))
            smooth += float(np.sum((u-prev_u)**2))
            prev_u = u

            mujoco.mj_step(model, data)

            # contacts split this step
            con_total, con_hand, con_env = count_object_contacts_split()
            hold_con_total += con_total
            hold_con_hand  += con_hand
            hold_con_env   += con_env

            # object kinematics
            p_now = data.xpos[obj_bid]
            v_now = data.cvel[obj_bid][3:6]
            speed = float(np.linalg.norm(v_now))
            drift = float(np.linalg.norm(p_now - pos_ref))

            max_speed = max(max_speed, speed)
            max_drift = max(max_drift, drift)

            # stability (old metric)
            if (p_now[2] > Z_MIN) and (speed < V_MAX) and (drift < DRIFT_MAX):
                stable_steps += 1

            # per-step fingertip touches for CGF/CE
            touch_now = {f: (sensor_sum(FINGERTIP_TOUCH[f]) if touch_present[f] else 0.0) for f in FINGERS}
            tip_sum_now = float(sum(touch_now.values()))
            tip_active_now = int(sum(1 for f in FINGERS if touch_now[f] > 0.0))
            ce_now = contact_entropy(touch_now)
            hold_tip_sums.append(tip_sum_now)

            # CGF condition: real finger grasp without environment support
            cgf_now = (tip_active_now >= CGF_MIN_TIP_ACTIVE) and (tip_sum_now >= CGF_TAU_TIP_SUM) and (con_env == 0) and (con_hand > 0)

            if cgf_now:
                cgf_streak += 1
                drift_grasp.append(drift)
                speed_grasp.append(speed)
                tip_sum_grasp.append(tip_sum_now)
                ce_grasp.append(ce_now)
            else:
                cgf_streak = 0

            gd_max = max(gd_max, cgf_streak)

            if drift > DRIFT_BAD_THR:
                bad_drift_cnt += 1
            if speed > SPEED_BAD_THR:
                bad_speed_cnt += 1

            if viewer is not None and do_render:
                render_step(viewer)

        hold_ratio = stable_steps / float(HOLD_STEPS)

        edr = (hold_con_env / hold_con_total) if hold_con_total > 0 else 0.0


        cgf_stable = 1 if gd_max >= CGF_STREAK_MIN else 0


        if len(drift_grasp) > 0:
            driftP95 = float(np.percentile(drift_grasp, 95))
            speedP95 = float(np.percentile(speed_grasp, 95))
            tip_sum_grasp_avg = float(np.mean(tip_sum_grasp))
            ce_grasp_avg = float(np.mean(ce_grasp))
        else:
            driftP95 = 0.0
            speedP95 = 0.0
            tip_sum_grasp_avg = 0.0
            ce_grasp_avg = 0.0

        real_grasp_old = (con_total0 >= MIN_CONTACTS_AFTER_CLOSE) and \
                         (tip_active >= MIN_TIP_ACTIVE_AFTER_CLOSE) and \
                         (tip_sum >= MIN_TIP_SUM_AFTER_CLOSE)

        success = 1 if (hold_ratio >= SUCCESS_HOLD_RATIO and real_grasp_old) else 0

        return {
            "contacts": int(con_total0),
            "contacts_hand": float(hold_con_hand) / float(HOLD_STEPS),
            "contacts_env": float(hold_con_env) / float(HOLD_STEPS),
            "contacts_total_avg": float(hold_con_total) / float(HOLD_STEPS),

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

            # NEW metrics
            "CGF_stable": int(cgf_stable),
            "GD": int(gd_max),
            "EDR": float(edr),
            "driftP95_grasp": float(driftP95),
            "speedP95_grasp": float(speedP95),
            "tip_sum_grasp_avg": float(tip_sum_grasp_avg),
            "CE_grasp_avg": float(ce_grasp_avg),

            # diagnostics
            "bad_drift_ratio": float(bad_drift_cnt) / float(HOLD_STEPS),
            "bad_speed_ratio": float(bad_speed_cnt) / float(HOLD_STEPS),
            "hold_tip_avg": float(np.mean(tip_sum_grasp)) if len(tip_sum_grasp) else 0.0,
            "opp": 0,  # keep field for compatibility if you later add opposition
        }


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

    rng = np.random.default_rng(123)

    baseline_A = np.array([1.0, 1.0, 1.0, max(1.0, LF_MIN), 1.0, 0.010, 0.30], dtype=float)
    baseline_B = np.array([1.2, 0.9, 0.9, max(1.15, LF_MIN), 1.4, 0.010, 0.30], dtype=float)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        p("Viewer opened.")

        # warmup steps (no long sleep)
        for _ in range(20):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()

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
                _ = eval_theta(best_theta, SEED0 + 900000 + it, viewer=viewer, render_one=True)

   
        p(" BEST RESULT")

        p(f"best_score={best_score:.4f}")
        p(f"best_theta=[wFF,wMF,wRF,wLF,wTH, th_touch, k_hold]={best_theta}")

 
        p(f"--- Visualize best_theta ({VISUAL_TRIALS} trials) ---")
        all_visual = []
        for t in range(1, VISUAL_TRIALS + 1):
            m = run_trial(best_theta, seed=SEED0 + 50000 + t, viewer=viewer, do_render=True)
            all_visual.append(m)

            tp = m["touch_per_finger"]
            # formatting like BASE
            p(
                f"TRIAL {t}: "
                f"SUCCESS={m['success']} hold={m['hold_ratio']:.2f} "
                f"con_hand={m['contacts_hand']:.2f} con_env={m['contacts_env']:.2f} con={m['contacts']} "
                f"z={m['z']:.3f} "
                f"drift={m['max_drift']:.3f} speed={m['max_speed']:.3f} "
                f"driftP95={m['driftP95_grasp']:.3f} speedP95={m['speedP95_grasp']:.3f} "
                f"bad_drift={m['bad_drift_ratio']:.2f} bad_speed={m['bad_speed_ratio']:.2f} "
                f"tip_active={m['tip_active']} tip_sum={m['tip_sum']:.3f} palm_sum={m['palm_sum']:.3f} "
                f"hold_tip_avg={m['tip_sum_grasp_avg']:.3f} streak={m['GD']} "
                f"EDR={m['EDR']:.2f} CGF={m['CGF_stable']} CEg={m['CE_grasp_avg']:.3f} tipg={m['tip_sum_grasp_avg']:.3f} "
                f"opp={m['opp']} "
                f"energy={m['energy']:.1f} smooth={m['smooth']:.1f} "
                f"FF={tp['FF']:.3f} MF={tp['MF']:.3f} RF={tp['RF']:.3f} LF={tp['LF']:.3f} TH={tp['TH']:.3f}"
            )

        # VISUAL SUMMARY (like BASE)
        succ_rate = float(np.mean([m["success"] for m in all_visual]))
        avg_con   = float(np.mean([m["contacts"] for m in all_visual]))
        avg_hand  = float(np.mean([m["contacts_hand"] for m in all_visual]))
        avg_env   = float(np.mean([m["contacts_env"] for m in all_visual]))
        avg_hold  = float(np.mean([m["hold_ratio"] for m in all_visual]))
        avg_drift = float(np.mean([m["max_drift"] for m in all_visual]))
        avg_dp95  = float(np.mean([m["driftP95_grasp"] for m in all_visual]))
        avg_sp95  = float(np.mean([m["speedP95_grasp"] for m in all_visual]))
        avg_gd    = float(np.mean([m["GD"] for m in all_visual]))
        avg_edr   = float(np.mean([m["EDR"] for m in all_visual]))
        avg_ceg   = float(np.mean([m["CE_grasp_avg"] for m in all_visual]))
        avg_tipg  = float(np.mean([m["tip_sum_grasp_avg"] for m in all_visual]))
        avg_cgf   = float(np.mean([m["CGF_stable"] for m in all_visual]))


        p(" SUMMARY (all trials)")

        p(f"Success rate: {succ_rate:.2f}")
        p(f"Avg contacts (total): {avg_con:.2f}")
        p(f"Avg contacts_hand: {avg_hand:.2f} | Avg contacts_env: {avg_env:.2f}")
        p(f"Avg hold_ratio: {avg_hold:.2f}")
        p(f"Avg max_drift: {avg_drift:.3f}")
        p(f"Avg driftP95: {avg_dp95:.3f} | Avg speedP95: {avg_sp95:.3f}")
        p(f"Avg GD: {avg_gd:.1f} | Avg CGF_stable: {avg_cgf:.2f} | Avg EDR: {avg_edr:.2f}")
        p(f"Avg CE@Grasp: {avg_ceg:.3f} | Avg tip_sum@Grasp: {avg_tipg:.3f}")

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
