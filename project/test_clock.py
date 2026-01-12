# ============================================================
# TEST RUN: CLOCK ONLY
# clock_only.xml
# ============================================================

import time
import mujoco
import mujoco.viewer

XML_PATH = r"C:\Users\rad\itmo\new_roms\project\models\clock_only.xml"

# -------------------------------
# LOAD MODEL
# -------------------------------

print("Loading clock-only model...")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)
print("Model loaded")

# -------------------------------
# LAUNCH VIEWER
# -------------------------------

viewer = mujoco.viewer.launch_passive(model, data)

# -------------------------------
# RESET SIMULATION
# -------------------------------

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
viewer.sync()

time.sleep(0.5)

# -------------------------------
# SIMULATION LOOP
# -------------------------------
# just physics: gravity, contacts, resting
# -------------------------------

print("Simulating clock physics. Close the viewer window to exit.")

while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.01)
