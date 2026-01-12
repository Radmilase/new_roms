# ============================================================
# RUN: Shadow Hand + selectable object scene (MuJoCo)
# Usage examples:
#   python project/run_hand.py --list
#   python project/run_hand.py --scene manipulate_block
#   python project/run_hand.py --scene manipulate_egg_touch_sensors
#   python project/run_hand.py --xml project/models/hand_manipulate_clock.xml
# ============================================================

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer


def find_repo_root(start: Path) -> Path:
    """Find repo root by locating 'project' folder upward."""
    cur = start.resolve()
    for _ in range(10):
        if (cur / "project").is_dir():
            return cur
        cur = cur.parent
    return start.resolve()


def default_models_dir() -> Path:
    root = find_repo_root(Path(__file__).parent)
    return root / "project" / "models"


def list_scenes(models_dir: Path) -> list[str]:
    """
    Collect "scene names" from xml files in models_dir.
    Scene name = filename without extension.
    """
    if not models_dir.is_dir():
        return []
    xmls = sorted(models_dir.glob("*.xml"))
    names = []
    for p in xmls:
        # отсеем "shared*.xml", чтобы список был читабельнее
        stem = p.stem
        if stem.startswith("shared"):
            continue
        names.append(stem)
    return names


def resolve_scene_to_xml(models_dir: Path, scene_name: str) -> Path:
    """
    Convert scene name to a concrete xml path under models_dir.
    Example: scene_name='manipulate_block' -> models_dir/'manipulate_block.xml'
    """
    xml_path = models_dir / f"{scene_name}.xml"
    return xml_path


def main():
    parser = argparse.ArgumentParser(
        description="Run MuJoCo hand scene and choose object/scene from project/models."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(default_models_dir()),
        help="Path to project/models folder (default: auto-detected).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene name = xml filename without .xml (e.g., manipulate_block, hand_manipulate_clock).",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Explicit path to an XML file (overrides --scene).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scene xml files in models-dir and exit.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Viewer update rate target (default 60).",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()

    if args.list:
        scenes = list_scenes(models_dir)
        if not scenes:
            print(f"No XML files found in: {models_dir}")
            sys.exit(0)

        print(f"Available scenes in {models_dir}:")
        for s in scenes:
            print(f"  - {s}")
        sys.exit(0)

    # Choose XML
    if args.xml:
        xml_path = Path(args.xml).resolve()
    else:
        if not args.scene:
            # разумный дефолт: если есть какой-то "manipulate_block.xml" — возьмём его, иначе первый xml
            scenes = list_scenes(models_dir)
            if not scenes:
                raise FileNotFoundError(f"No XML files found in: {models_dir}")
            preferred = "manipulate_block"
            scene_name = preferred if preferred in scenes else scenes[0]
            print(f"[INFO] --scene not provided, using default: {scene_name}")
        else:
            scene_name = args.scene

        xml_path = resolve_scene_to_xml(models_dir, scene_name)

    if not xml_path.is_file():
        print(f"[ERROR] XML not found: {xml_path}")
        print("Tip: run with --list to see available scenes.")
        sys.exit(1)

    print(f"[INFO] Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Run viewer loop
    dt = model.opt.timestep
    target_dt = 1.0 / max(args.fps, 1e-6)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.perf_counter()
        last = t0
        while viewer.is_running():
            # simulate a bit
            mujoco.mj_step(model, data)

            # sync viewer
            viewer.sync()

            # soft rate limit so CPU doesn't melt
            now = time.perf_counter()
            elapsed = now - last
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
            last = now


if __name__ == "__main__":
    main()
