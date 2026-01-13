# ============================================================
# run_all.py
# Robust runner for MuJoCo scenes inside new_roms
# - Auto-finds repo root (folder that contains "project/")
# - Loads: project/models/hand/clock_only.xml  (by default)
# - Tries to drop the clock from above by setting FREEJOINT pose
# - Opens mujoco viewer
# ============================================================

import os
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer


# -------------------------------
# PATH HELPERS
# -------------------------------
def find_repo_root(start_dir: str) -> str:
    """Walk up until we find a folder containing 'project'."""
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, "project")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise FileNotFoundError("Repo root not found (folder with 'project/' not found).")
        cur = parent


def resolve_xml_path(repo_root: str, xml_rel: str) -> str:
    """Resolve xml relative to repo root and normalize."""
    xml_path = os.path.normpath(os.path.join(repo_root, xml_rel))
    return xml_path


# -------------------------------
# MODEL HELPERS
# -------------------------------
def list_bodies(model: mujoco.MjModel, limit: int = 80) -> None:
    print("\nBodies in model (first {}):".format(min(limit, model.nbody)))
    for i in range(min(limit, model.nbody)):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if n:
            print(" -", n)


def has_freejoint_on_body(model: mujoco.MjModel, body_id: int) -> bool:
    jadr = model.body_jntadr[body_id]
    jnum = model.body_jntnum[body_id]
    if jnum <= 0:
        return False
    # assume first joint is the body's joint
    jid = jadr
    return model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE


def guess_clock_body(model: mujoco.MjModel) -> str | None:
    """
    Try to find a body that looks like the clock AND has a freejoint.
    """
    keywords = ["clock", "crosley", "alarm", "obj", "object", "model", "mesh"]
    candidates = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name:
            continue
        lname = name.lower()
        if any(k in lname for k in keywords) and has_freejoint_on_body(model, bid):
            candidates.append(name)
    return candidates[0] if candidates else None


def set_freejoint_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str,
                       pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> None:
    """
    Set qpos/qvel for a body that has a FREEJOINT:
    qpos: [x y z qw qx qy qz]
    qvel: [vx vy vz wx wy wz]
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body '{body_name}' not found.")

    jadr = model.body_jntadr[bid]
    jnum = model.body_jntnum[bid]
    if jnum <= 0:
        raise ValueError(f"Body '{body_name}' has no joints (expected a freejoint).")

    jid = jadr
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Body '{body_name}' does not have a FREEJOINT.")

    qpos_adr = model.jnt_qposadr[jid]
    qvel_adr = model.jnt_dofadr[jid]

    data.qpos[qpos_adr:qpos_adr + 3] = pos_xyz
    data.qpos[qpos_adr + 3:qpos_adr + 7] = quat_wxyz
    data.qvel[qvel_adr:qvel_adr + 6] = 0.0


# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        default=os.path.join("project", "models", "hand", "clock_only.xml"),
        help="XML path relative to repo root (default: project/models/hand/clock_only.xml)"
    )
    parser.add_argument("--drop_z", type=float, default=0.35, help="Drop height (meters)")
    parser.add_argument("--drop_x", type=float, default=0.00, help="Drop x")
    parser.add_argument("--drop_y", type=float, default=0.00, help="Drop y")
    parser.add_argument("--no_drop", action="store_true", help="Do not reposition clock")
    parser.add_argument("--clock_body", type=str, default="", help="Force body name for clock (optional)")
    args = parser.parse_args()

    # Find repo root from THIS file's location (run_all.py is in repo root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = find_repo_root(script_dir)

    xml_path = resolve_xml_path(repo_root, args.xml)

    print("Repo root :", repo_root)
    print("XML path  :", xml_path)
    print("XML exists:", os.path.exists(xml_path))

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML not found: {xml_path}")

    print("\nLoading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print("Loaded OK. nbody:", model.nbody, "njnt:", model.njnt)

    # Drop clock (if possible)
    if not args.no_drop:
        clock_body = args.clock_body.strip() or guess_clock_body(model)

        if not clock_body:
            print("\n[WARN] Could not find a clock body WITH freejoint => can't 'drop' by teleport.")
            print("This usually means the clock is welded / has no freejoint in the XML.")
            list_bodies(model, limit=80)
            print("\nIf you know the body name, run:")
            print('  python run_all.py --clock_body "<NAME>"')
        else:
            drop_xyz = np.array([args.drop_x, args.drop_y, args.drop_z], dtype=float)
            drop_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # w,x,y,z

            try:
                set_freejoint_pose(model, data, clock_body, drop_xyz, drop_quat)
                mujoco.mj_forward(model, data)
                print(f"\nClock body '{clock_body}' dropped at {drop_xyz.tolist()}")
            except Exception as e:
                print("\n[ERROR] Failed to set clock pose:", e)
                list_bodies(model, limit=80)
                raise

    print("\nStarting viewer... (close window to stop)")
    dt = model.opt.timestep
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last = time.time()
        while viewer.is_running():
            mujoco.mj_step(model, data)

            # real-time-ish
            now = time.time()
            elapsed = now - last
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)
            last = time.time()

            viewer.sync()


if __name__ == "__main__":
    main()
