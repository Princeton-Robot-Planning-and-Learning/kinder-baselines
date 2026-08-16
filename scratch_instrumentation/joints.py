"""List every joint with its qpos/qvel address, to find the full gripper block."""

import os

import kinder
import numpy as np
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv  # noqa: F401

kinder.register_all_environments()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

env = kinder.make("kinder/Tossing3D-o1-v0", render_mode=None)
env.reset(seed=101)
sim = env.unwrapped._object_centric_env
re = env.unwrapped._object_centric_env._robot_env
m = re.sim.model
md = re.sim.data.mj_data
import mujoco

mm = m.mj_model
print("nq", mm.nq, "nv", mm.nv, "nu", mm.nu, "njnt", mm.njnt)
for j in range(mm.njnt):
    name = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_JOINT, j)
    print(
        f"jnt {j:3d} qposadr {mm.jnt_qposadr[j]:3d} qveladr {mm.jnt_dofadr[j]:3d} "
        f"type {mm.jnt_type[j]} name {name}"
    )
print("\nactuators")
for a in range(mm.nu):
    print(a, mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_ACTUATOR, a))
print("\nrobot_env gripper qpos indices:", np.asarray(re.qpos["gripper"]))
