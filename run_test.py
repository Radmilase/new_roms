# run_tableware_set.py
import time
import mujoco
import mujoco.viewer

XML_PATH = r"C:\Users\rad\itmo\new_roms\project\models\mujoco_scanned_objects\models\TABLEWARE_SET\model.xml"

def main():
    print("Loading:", XML_PATH)
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print("Loaded OK. Viewer: ESC to close.")
    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            # realtime pacing
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

if __name__ == "__main__":
    main()
