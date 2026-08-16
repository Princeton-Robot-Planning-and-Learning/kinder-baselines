#!/usr/bin/env bash
# Scratch env parameterised by which kindergarden tree to use.
#   usage: ./envx.sh <kg_dir> [python args...]
# Run with no python args to print what it resolved.
set -euo pipefail
T=/home/josh/.claude/jobs/tossdiag
PY=/home/josh/miniconda3/envs/hitl-pmp/bin/python
KG="$1"; shift

export PYTHONPATH="$T:$T/$KG/src:$T/kb/kinder-models/src:$T/kb/kinder-bilevel-planning/src"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1

if [ "$#" -eq 0 ]; then
    echo "PYTHONPATH  $PYTHONPATH"
    exec "$PY" -c "
import kinder, kinder_models, kinder_bilevel_planning, bilevel_planning
from kinder.envs.dynamic3d.object_types import MujocoObjectTypeFeatures, MujocoTidyBotRobotObjectType
print('kinder       ', kinder.__file__)
print('kinder_models', kinder_models.__file__)
f = MujocoObjectTypeFeatures[MujocoTidyBotRobotObjectType]
print('robot features', len(f))
print('gripper feats ', [x for x in f if 'gripper' in x])
"
fi
exec "$PY" "$@"
