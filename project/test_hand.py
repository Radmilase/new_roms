# ============================================================
# TEST RUN: Shadow Dexterous Hand
# hand_manipulate_clock0.xml
# ============================================================

import time
import mujoco
import mujoco.viewer

XML_PATH = r"C:\Users\rad\anaconda3\envs\dexgrasp\Lib\site-packages\gymnasium_robotics\envs\assets\hand\clock_only.xml"

# -------------------------------
# LOAD MODEL
# -------------------------------

print("Loading model...")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)
print("Model loaded")

# -------------------------------
# LAUNCH VIEWER
# -------------------------------

viewer = mujoco.viewer.launch_passive(model, data)

# -------------------------------
# ACTUATORS
# 0–1: wrist, 2–19: fingers
# -------------------------------

FINGER_ACTUATORS = range(2, model.nu)

# -------------------------------
# RESET
# -------------------------------

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
viewer.sync()

time.sleep(0.5)

# -------------------------------
# OPEN HAND
# -------------------------------

for _ in range(50):
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.01)

# -------------------------------
# CLOSE HAND
# -------------------------------

for _ in range(150):
    data.ctrl[:] = 0.0
    for i in FINGER_ACTUATORS:
        data.ctrl[i] = 1.0
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.01)

# -------------------------------
# HOLD
# -------------------------------

print("Holding pose. Close the viewer window to exit.")

while viewer.is_running():
    for i in FINGER_ACTUATORS:
        data.ctrl[i] = 1.0
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.02)
