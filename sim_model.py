"""Convert the pinned vendor kinematics and mesh geometry to a MuJoCo scene."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from jaw_collision import add_jaw_collision
from sim_storage import add_bin

WRIST_CAMERA_NAME = "wrist_depth"
WRIST_CAMERA_POS = "0 .065 .075"
WRIST_CAMERA_FOV = 58
# Vendor EoAT nameplate 20 kgf cm; an upper bound, not continuous torque.
JAW_RATED_TORQUE_NM = 1.96133

VENDOR_COMMIT = "40dbd84b553695212fab713e8465f817ba95454d"


def add_wrist_envelopes(wrist, workspace, optical_geometry=False):
    padding = float(workspace["padding_mm"])
    items = workspace["wrist_envelopes"]
    if not 0 <= padding <= 30 or not items:
        raise ValueError("Measured wrist boxes require padding in [0,30] mm")
    measured_mass = ["mass_g" in item for item in items]
    if any(measured_mass) and not all(measured_mass):
        raise ValueError("Provide mass_g for every wrist box or for none")
    names = set()
    for item in items:
        name = item["name"]
        center, size = np.asarray(item["center_mm"], float), np.asarray(item["size_mm"], float)
        if (not isinstance(name, str) or not name or name in names or center.shape != (3,)
                or size.shape != (3,) or not np.isfinite([center, size]).all() or np.any(size <= 0)):
            raise ValueError("Invalid measured wrist envelope")
        names.add(name)
        mass = float(item.get("mass_g", 0))
        if "mass_g" in item and (not np.isfinite(mass) or mass <= 0):
            raise ValueError("Measured wrist box mass_g must be finite and positive")
        pos = " ".join(map(str, center / 1000))
        ET.SubElement(wrist, "geom", name=f"wrist_envelope_{name}", type="box", pos=pos,
                      size=" ".join(map(str, (size / 2 + padding) / 1000)),
                      group="4" if optical_geometry else "1", mass="0",
                      rgba="0 0 0 0" if optical_geometry else ".85 .43 .10 .35")
        if optical_geometry:
            ET.SubElement(wrist, "geom", name=f"wrist_surface_{name}", type="box", pos=pos,
                          size=" ".join(map(str, size / 2000)), group="1", mass=str(mass / 1000),
                          contype="0", conaffinity="0", rgba=".85 .43 .10 1")


def scene(vendor, block_xy, yaw, gripper="parallel-pads", camera_mode="wrist",
          block_size=(.046, .030, .030), block_mass=.020, jaw_limit_nm=JAW_RATED_TORQUE_NM,
          storage=False, ledge_height=None, clutter=False, ledge_x_shift=0., ledge_y_shift=0.,
          grip_friction=None, block_mesh=None, ledge_box=None, camera_transform=None, wrist_workspace=None):
    package = Path(vendor).resolve() / "src/roarm_main/roarm_description"
    source = package / "urdf/roarm_m3/roarm_m3.xacro"
    if not source.is_file():
        raise FileNotFoundError(f"Official RoArm-M3 xacro missing: {source}")
    urdf = ET.parse(source).getroot()
    root = ET.Element("mujoco", model="waveshare_roarm_m3")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    actuators = ET.SubElement(root, "actuator")
    exclusions = ET.SubElement(root, "contact")
    bodies = {"world": world}
    links = {link.attrib["name"]: link for link in urdf.findall("link")}
    ET.SubElement(world, "geom", name="table", type="box", pos="0.55 0 -0.085",
                  size="0.40 0.45 0.015", group="0", rgba=".6 .6 .6 1")
    ET.SubElement(world, "light", pos=".3 -.3 .7", dir="0 .3 -1", diffuse=".8 .8 .8")
    support_z = -.07 if ledge_height is None else -.07 + ledge_height
    if ledge_height is not None:
        x, y, length, width, thickness = ledge_box or (.296, 0., .142, .184, .024)
        ET.SubElement(world, "geom", name="rock_wall", type="box",
                      pos=f"{.405 + ledge_x_shift} {ledge_y_shift} .20", size=".04 .32 .27",
                      group="0", rgba=".32 .28 .24 1")
        ET.SubElement(world, "geom", name="rock_ledge", type="box",
                      pos=f"{x + ledge_x_shift} {y + ledge_y_shift} {support_z - thickness / 2}",
                      size=f"{length / 2} {width / 2} {thickness / 2}",
                      group="0", rgba=".44 .37 .31 1")
    half_size = np.asarray(block_size) / 2
    if storage:
        add_bin(world)
    block = ET.SubElement(world, "body", name="block",
                          pos=f"{block_xy[0]} {block_xy[1]} {support_z + half_size[2] + .0001}", euler=f"0 0 {yaw}")
    ET.SubElement(block, "freejoint")
    if block_mesh is None:
        geometry = dict(type="box", size=" ".join(map(str, half_size)))
    else:
        ET.SubElement(asset, "mesh", name="sample_mesh", file=str(block_mesh), scale=".001 .001 .001")
        geometry = dict(type="mesh", mesh="sample_mesh")
    ET.SubElement(block, "geom", name="block_geom", **geometry,
                  mass=str(block_mass), friction="1.2 .02 .001", group="0", rgba=".9 .45 .1 1")
    if clutter:
        positions = (((.296, -.058, -.35), (.286, .055, .55)) if ledge_height is None
                     else ((.268, -.150, -.35), (.250, .120, .55)))
        for index, (x, y, angle) in enumerate(positions):
            ET.SubElement(world, "geom", name=f"distractor_{index}", type="box",
                          pos=f"{x} {y} {-.07 + .013}", euler=f"0 0 {angle}",
                          size=".015 .012 .013", group="0", rgba=".35 .52 .66 1")
    if grip_friction is not None:
        if gripper != "vendor-split" or not 0 < grip_friction <= 2:
            raise ValueError("Grip friction requires vendor-split and a value in (0, 2]")
        for jaw in ("link5", "gripper_link"):
            for part in ("tip", "section_1"):
                ET.SubElement(exclusions, "pair", geom1="block_geom", geom2=f"{jaw}_{part}",
                              friction=f"{grip_friction} {grip_friction} .02 .001 .001")

    for joint in urdf.findall("joint"):
        name = joint.attrib["name"]
        if gripper == "parallel-pads" and name == "link5_to_gripper_link":
            continue
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = joint.find("origin")
        pos = origin.attrib["xyz"]
        if parent == "world":
            # Put base_link at the origin, matching the perception coordinate system.
            pos = "0 0 0"
        rpy = np.fromstring(origin.attrib["rpy"], sep=" ")
        xyzw = Rotation.from_euler("xyz", rpy).as_quat()
        quat = " ".join(map(str, xyzw[[3, 0, 1, 2]]))
        body = ET.SubElement(bodies[parent], "body", name=child, pos=pos, quat=quat)
        bodies[child] = body
        if joint.attrib["type"] == "revolute":
            limit = joint.find("limit").attrib
            bounds = f'{limit["lower"]} {limit["upper"]}'
            ET.SubElement(body, "joint", name=name, type="hinge",
                          axis=joint.find("axis").attrib["xyz"], range=bounds,
                          damping="0.4", armature="0.006")
            limit_nm = jaw_limit_nm if name == "link5_to_gripper_link" else 18
            ET.SubElement(actuators, "position", name=name, joint=name, ctrlrange=bounds,
                          kp="250", kv="12", forcerange=f"{-limit_nm} {limit_nm}")
        collision = links[child].find("collision/geometry/mesh")
        if collision is not None:
            filename = collision.attrib["filename"].split("/meshes/", 1)[1]
            path = package / "meshes" / filename
            ET.SubElement(asset, "mesh", name=child, file=path.as_posix(),
                          scale=collision.attrib["scale"])
            if gripper == "vendor-split" and child in ("link5", "gripper_link"):
                ET.SubElement(body, "geom", name=child, type="mesh", mesh=child, mass=".12",
                              group="2", contype="0", conaffinity="0", rgba=".35 .48 .65 1")
                add_jaw_collision(asset, body, path, child)
                if parent != "world":
                    ET.SubElement(exclusions, "exclude", body1=parent, body2=child)
                continue
            if child == "link5" and gripper == "parallel-pads":
                ET.SubElement(body, "geom", name=child, type="mesh", mesh=child,
                              mass=".12", group="1", contype="0", conaffinity="0")
                ET.SubElement(body, "geom", name="wrist_guard", type="box",
                              pos="-.005 0 .03", size=".030 .023 .031", group="1", mass="0")
                continue
            ET.SubElement(body, "geom", name=child, type="mesh", mesh=child,
                          mass="0.12", group="1", friction="1.3 .02 .001",
                          rgba=".35 .48 .65 1")
        if parent != "world":
            ET.SubElement(exclusions, "exclude", body1=parent, body2=child)
    if gripper == "parallel-pads":
        for side in (-1, 1):
            name = f"pad_{side}"
            pad = ET.SubElement(bodies["link5"], "body", name=name,
                                pos=f"{side * .007} 0 .109")
            ET.SubElement(pad, "joint", name=name, type="slide", axis=f"{side} 0 0",
                          range="0 .030", damping="4", armature=".01")
            ET.SubElement(pad, "geom", name=name, type="box", size=".004 .013 .013",
                          mass=".025", group="1", friction="1.3 .02 .001")
            ET.SubElement(actuators, "position", name=name, joint=name,
                          ctrlrange="0 .030", kp="650", kv="8", forcerange="-9 9")
            ET.SubElement(exclusions, "exclude", body1="link5", body2=name)
    if camera_mode == "wrist":
        wrist = bodies["link5"]
        ET.SubElement(wrist, "geom", name="camera_bracket", type="box",
                      pos="0 .033 .065", size=".012 .021 .006",
                      mass=".022", group="1", rgba=".28 .3 .34 1")
        ET.SubElement(wrist, "geom", name="camera_housing", type="box",
                      pos="0 .065 .065", size=".037 .013 .009",
                      mass=".075", group="1", rgba=".85 .43 .10 1")
        for x in (-.021, 0, .021):
            ET.SubElement(wrist, "geom", name=f"camera_window_{x}", type="cylinder",
                          pos=f"{x} .065 .0745", size=".005 .0005", mass="0",
                          contype="0", conaffinity="0", group="2", rgba=".12 .58 .75 1")
        if wrist_workspace is not None:
            use_measured_mass = all("mass_g" in item for item in wrist_workspace["wrist_envelopes"])
            for geom in wrist.findall("geom"):
                if geom.get("name", "").startswith("camera_"):
                    geom.attrib.update(group="2", contype="0", conaffinity="0", rgba="0 0 0 0")
                    if use_measured_mass:
                        geom.set("mass", "0")
            add_wrist_envelopes(wrist, wrist_workspace, optical_geometry=True)
        camera_pos, camera_axes = WRIST_CAMERA_POS, "1 0 0 0 -1 0"
        if camera_transform is not None:
            rotation, translation = camera_transform[:3, :3], camera_transform[:3, 3]
            camera_pos = " ".join(map(str, translation / 1000))
            camera_axes = " ".join(map(str, np.r_[rotation[:, 0], -rotation[:, 1]]))
        ET.SubElement(wrist, "camera", name=WRIST_CAMERA_NAME,
                      pos=camera_pos, xyaxes=camera_axes, fovy=str(WRIST_CAMERA_FOV))
    ET.SubElement(bodies["hand_tcp"], "site", name="tcp", size=".004")
    return ET.tostring(root, encoding="unicode")






