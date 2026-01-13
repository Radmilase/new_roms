# ============================================================
# TEST RUN: Shadow Dexterous Hand
# Drop RANDOM scanned object into hand (NO SCALE)
# ============================================================

import random
import time
from pathlib import Path

import mujoco
import mujoco.viewer


# -------------------------------
# PATHS (локалка)
# -------------------------------
ROOT_DIR = Path(r"C:\Users\rad\itmo\new_roms")
SCANNED_MODELS_DIR = ROOT_DIR / "project" / "models" / "mujoco_scanned_objects" / "models"

# Рука из gymnasium_robotics (как в твоём примере).
# Если у тебя другой файл руки — просто поменяй путь.
HAND_XML_PATH = Path(
    r"C:\Users\rad\anaconda3\envs\dexgrasp\Lib\site-packages\gymnasium_robotics\envs\assets\hand\hand.xml"
)

assert SCANNED_MODELS_DIR.exists(), f"Scanned models dir not found: {SCANNED_MODELS_DIR}"
assert HAND_XML_PATH.exists(), f"Hand xml not found: {HAND_XML_PATH}"


# -------------------------------
# PICK RANDOM OBJECT
# -------------------------------
def pick_random_obj(scanned_root: Path):
    """
    Ищем папки, где есть model.obj (или любой .obj).
    Возвращаем (dir, obj_path, texture_path_or_None)
    """
    candidates = []
    for d in scanned_root.iterdir():
        if not d.is_dir():
            continue

        obj_main = d / "model.obj"
        if obj_main.exists():
            obj_path = obj_main
        else:
            obj_path = next(d.glob("*.obj"), None)

        if obj_path is None:
            continue

        tex_path = d / "texture.png"
        if not tex_path.exists():
            tex_path = None

        candidates.append((d, obj_path, tex_path))

    if not candidates:
        raise FileNotFoundError(f"No .obj scanned objects found in {scanned_root}")

    return random.choice(candidates)


# -------------------------------
# BUILD SCENE XML (NO SCALE!)
# -------------------------------
def build_scene_xml(hand_xml_path: Path, obj_path: Path, tex_path: Path | None):
    obj_abs = obj_path.as_posix()

    texture_block = ""
    material_ref = ""
    if tex_path is not None:
        tex_abs = tex_path.as_posix()
        texture_block = f"""
        <texture name="obj_tex" type="2d" file="{tex_abs}"/>
        <material name="obj_mat" texture="obj_tex" texrepeat="1 1"
                  specular="0.2" shininess="0.2"/>
        """
        material_ref = 'material="obj_mat"'

    # Позиция падения: над ладонью (можно чуть менять, если падает мимо)
    drop_pos = "0 0 0.35"
    drop_quat = "1 0 0 0"

    xml = f"""
<mujoco model="shadowhand_random_drop_no_scale">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81" iterations="50" solver="Newton"/>
  <size njmax="2000" nconmax="400"/>

  <asset>
    {texture_block}
    <!-- ВАЖНО: scale НЕ задаём -->
    <mesh name="obj_mesh" file="{obj_abs}"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="1 1 1.5" dir="-1 -1 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Рука -->
    <include file="{hand_xml_path.as_posix()}"/>

    <!-- Падающий объект -->
    <body name="dropped_object" pos="{drop_pos}" quat="{drop_quat}">
      <freejoint/>
      <geom type="mesh" mesh="obj_mesh" {material_ref}
            density="800"
            friction="1.0 0.005 0.0001"
            solref="0.02 1"
            solimp="0.9 0.95 0.001"
            rgba="0.8 0.8 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    return xml


# -------------------------------
# ACTUATORS: fingers
# -------------------------------
def finger_actuators(model: mujoco.MjModel):
    """
    Пытаемся найти актуаторы пальцев по имени.
    Если не нашли — fallback: range(2, nu) как в твоём коде.
    """
    names = []
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        names.append(nm or "")

    ids = []
    for i, nm in enumerate(names):
        low = nm.lower()
        if any(k in low for k in ["ff", "mf", "rf", "lf", "th", "finger"]):
            ids.append(i)

    if len(ids) >= 5:
        return ids

    return list(range(2, model.nu))


# -------------------------------
# MAIN
# -------------------------------
def main():
    obj_dir, obj_path, tex_path = pick_random_obj(SCANNED_MODELS_DIR)
    print(f"[Picked] {obj_dir.name} -> {obj_path.name} (texture: {'yes' if tex_path else 'no'})")

    scene_xml = build_scene_xml(HAND_XML_PATH, obj_path, tex_path)

    print("Loading model...")
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    print("Model loaded")

    viewer = mujoco.viewer.launch_passive(model, data)

    FINGERS = finger_actuators(model)
    print("Finger actuators:", FINGERS, "nu=", model.nu)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(0.3)

    # OPEN
    for _ in range(60):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    # let it fall
    for _ in range(150):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)

    # CLOSE
    for _ in range(200):
        data.ctrl[:] = 0.0
        for i in FINGERS:
            data.ctrl[i] = 1.0
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)

    # HOLD
    print("Holding pose. Close the viewer window to exit.")
    while viewer.is_running():
        for i in FINGERS:
            data.ctrl[i] = 1.0
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
