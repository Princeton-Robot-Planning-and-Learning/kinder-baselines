"""Exhaustive scratch instrumentation for the Tossing3D bilevel refinement/execution gap.

SCRATCH ONLY -- imported by probes, never committed alongside a fix.

Emits one flat JSONL record per event. Everything is dumped: full object-centric state
(every feature of every object), raw MuJoCo qpos/qvel/ctrl/qacc_warmstart/ncon/time,
the full action vector, every predicate evaluation with its intermediates, every sampler
call with its parameters and atom-set diff, every set_state with before/after, and the
goal-region bounds every single time they are read.

Records are tagged with a global PHASE ("planning" / "execution" / other) and the id() of
the env instance, so the refinement-simulated stream and the executed stream can be
aligned mechanically.
"""

import json
import os
import time
from typing import Any

import numpy as np

_FH = None
_SEQ = 0
PHASE = "init"
STEP_INDEX: dict[str, int] = {}
ENV_LABELS: dict[int, str] = {}


def open_log(path: str) -> None:
    global _FH  # noqa: PLW0603
    _FH = open(path, "w")  # noqa: SIM115


def close_log() -> None:
    if _FH is not None:
        _FH.flush()
        _FH.close()


def set_phase(p: str) -> None:
    global PHASE  # noqa: PLW0603
    PHASE = p


def label_env(env: Any, label: str) -> None:
    ENV_LABELS[id(env)] = label


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


def emit(kind: str, **kw: Any) -> None:
    global _SEQ  # noqa: PLW0603
    if _FH is None:
        return
    rec = {"seq": _SEQ, "t": time.time(), "kind": kind, "phase": PHASE}
    _SEQ += 1
    for k, v in kw.items():
        rec[k] = _jsonable(v)
    _FH.write(json.dumps(rec) + "\n")
    if _SEQ % 200 == 0:
        _FH.flush()


# --------------------------------------------------------------------------- dumps


def dump_state(state: Any) -> dict:
    """Every feature of every object in an ObjectCentricState."""
    from kinder.envs.dynamic3d.object_types import MujocoObjectTypeFeatures

    out: dict[str, Any] = {}
    try:
        objs = list(state)
    except Exception:  # noqa: BLE001
        objs = []
    if not objs:
        try:
            objs = state.data.keys()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            objs = []
    for obj in objs:
        feats = MujocoObjectTypeFeatures.get(obj.type, [])
        d = {}
        for f in feats:
            try:
                d[f] = float(state.get(obj, f))
            except Exception:  # noqa: BLE001, PERF203
                pass
        out[obj.name] = d
    return out


def dump_mj(env: Any) -> dict:
    """Raw MuJoCo state, the layer the object-centric state abstracts away."""
    out: dict[str, Any] = {}
    try:
        re = env._robot_env  # noqa: SLF001
        sim = re.sim
        md = sim.data.mj_data
        out["qpos"] = np.asarray(md.qpos).copy()
        out["qvel"] = np.asarray(md.qvel).copy()
        out["ctrl"] = np.asarray(md.ctrl).copy()
        out["time"] = float(md.time)
        out["ncon"] = int(md.ncon)
        try:
            out["qacc_warmstart"] = np.asarray(md.qacc_warmstart).copy()
        except Exception:  # noqa: BLE001
            pass
        try:
            out["qacc"] = np.asarray(md.qacc).copy()
        except Exception:  # noqa: BLE001
            pass
        try:
            out["act"] = np.asarray(md.act).copy()
        except Exception:  # noqa: BLE001
            pass
        try:
            out["qfrc_constraint"] = np.asarray(md.qfrc_constraint).copy()
        except Exception:  # noqa: BLE001
            pass
        for grp in ("base", "arm", "gripper"):
            try:
                out[f"qpos_{grp}"] = np.asarray(re.qpos[grp]).copy()
                out[f"qvel_{grp}"] = np.asarray(re.qvel[grp]).copy()
                out[f"ctrl_{grp}"] = np.asarray(re.ctrl[grp]).copy()
            except Exception:  # noqa: BLE001, PERF203
                pass
    except Exception as e:  # noqa: BLE001
        out["mj_error"] = repr(e)
    return out


# --------------------------------------------------------------------------- patches


def install(verbose_predicates: bool = True) -> None:
    """Monkeypatch everything. Idempotent-ish; call once."""
    from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv

    # ---- auto-label every env instance by creation order, so the refinement stream and
    # the executed stream are distinguishable without guessing.
    _orig_init = ObjectCentricTidyBot3DEnv.__init__
    made = {"n": 0}

    def init(self, *a, **kw):  # noqa: ANN001, ANN202
        _orig_init(self, *a, **kw)
        label = f"sim{made['n']}"
        made["n"] += 1
        ENV_LABELS[id(self)] = label
        emit("env_created", env=label, kwargs={k: repr(v) for k, v in kw.items()})

    ObjectCentricTidyBot3DEnv.__init__ = init  # type: ignore[assignment]

    # ---- env.step
    _orig_step = ObjectCentricTidyBot3DEnv.step

    def step(self, action):  # noqa: ANN001, ANN202
        label = ENV_LABELS.get(id(self), f"env{id(self) % 100000}")
        key = f"{PHASE}:{label}"
        i = STEP_INDEX.get(key, 0)
        STEP_INDEX[key] = i + 1
        emit(
            "step_pre",
            env=label,
            step=i,
            action=np.asarray(action),
            action_len=len(np.asarray(action).ravel()),
            action_dtype=str(np.asarray(action).dtype),
            mj=dump_mj(self),
        )
        out = _orig_step(self, action)
        obs = out[0]
        emit(
            "step_post",
            env=label,
            step=i,
            reward=out[1],
            terminated=bool(out[2]),
            truncated=bool(out[3]),
            state=dump_state(obs),
            mj=dump_mj(self),
        )
        return out

    ObjectCentricTidyBot3DEnv.step = step  # type: ignore[assignment]

    # ---- env.set_state
    _orig_set_state = ObjectCentricTidyBot3DEnv.set_state

    def set_state(self, state):  # noqa: ANN001, ANN202
        label = ENV_LABELS.get(id(self), f"env{id(self) % 100000}")
        before = dump_mj(self)
        out = _orig_set_state(self, state)
        after = dump_mj(self)
        changed = {}
        for k in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
            if k in before and k in after:
                a = np.asarray(before[k], dtype=float)
                b = np.asarray(after[k], dtype=float)
                if a.shape == b.shape:
                    changed[k] = float(np.max(np.abs(a - b))) if a.size else 0.0
        emit(
            "set_state",
            env=label,
            requested=dump_state(state),
            mj_before=before,
            mj_after=after,
            max_abs_change=changed,
        )
        return out

    ObjectCentricTidyBot3DEnv.set_state = set_state  # type: ignore[assignment]

    # ---- env.reset
    _orig_reset = ObjectCentricTidyBot3DEnv.reset

    def reset(self, *a, **kw):  # noqa: ANN001, ANN202
        label = ENV_LABELS.get(id(self), f"env{id(self) % 100000}")
        out = _orig_reset(self, *a, **kw)
        emit(
            "env_reset",
            env=label,
            args=list(a),
            kwargs={k: v for k, v in kw.items()},
            state=dump_state(out[0]),
            mj=dump_mj(self),
        )
        return out

    ObjectCentricTidyBot3DEnv.reset = reset  # type: ignore[assignment]

    # ---- _check_goals, with the region bounds it used
    _orig_check_goals = ObjectCentricTidyBot3DEnv._check_goals  # noqa: SLF001

    def check_goals(self):  # noqa: ANN001, ANN202
        label = ENV_LABELS.get(id(self), f"env{id(self) % 100000}")
        res = _orig_check_goals(self)
        st = self._get_current_state()  # noqa: SLF001
        emit(
            "check_goals",
            env=label,
            result=bool(res),
            goal_state=self.task_config.get("goal_state", []),
            regions=self.task_config.get("regions", {}),
            state=dump_state(st),
        )
        return res

    ObjectCentricTidyBot3DEnv._check_goals = check_goals  # type: ignore[assignment]  # noqa: SLF001

    # ---- check_in_region on every fixture/object class that defines it
    from kinder.envs.dynamic3d.objects import base as objbase
    from kinder.envs.dynamic3d.objects import fixtures as objfix

    for mod in (objbase, objfix):
        for name in dir(mod):
            cls = getattr(mod, name)
            if not isinstance(cls, type):
                continue
            if "check_in_region" not in cls.__dict__:
                continue
            _orig = cls.__dict__["check_in_region"]

            def make(_orig=_orig, _cls=cls):  # noqa: ANN202
                def wrapper(self, position, region_name, env=None, *a, **kw):  # noqa: ANN001, ANN202
                    try:
                        res = _orig(self, position, region_name, env, *a, **kw)
                    except TypeError:
                        res = _orig(self, position, region_name, *a, **kw)
                    info = {}
                    for attr in ("regions", "_regions", "region_bounds"):
                        if hasattr(self, attr):
                            try:
                                info[attr] = getattr(self, attr)
                            except Exception:  # noqa: BLE001, PERF203
                                pass
                    emit(
                        "check_in_region",
                        cls=_cls.__name__,
                        region_name=region_name,
                        position=np.asarray(position),
                        result=bool(res),
                        obj_info=info,
                    )
                    return res

                return wrapper

            setattr(cls, "check_in_region", make())

    # ---- the abstractor
    from kinder_models.dynamic3d.tossing import state_abstractions as sa

    A = sa.Tossing3DStateAbstractor
    _orig_abs = A.state_abstractor

    def state_abstractor(self, state):  # noqa: ANN001, ANN202
        out = _orig_abs(self, state)
        emit(
            "state_abstractor",
            atoms=sorted(str(x) for x in out.atoms),
            state=dump_state(state),
        )
        return out

    A.state_abstractor = state_abstractor  # type: ignore[assignment]

    if verbose_predicates:
        for meth in (
            "_check_gripper_open",
            "_check_on_ground",
            "_check_holding",
            "_check_is_down_x",
            "_check_in_goal_region",
        ):
            _o = getattr(A, meth)
            raw = _o.__func__ if isinstance(_o, staticmethod) else _o
            is_static = isinstance(A.__dict__.get(meth), staticmethod)

            def mk(_raw=raw, _meth=meth, _static=is_static):  # noqa: ANN202
                if _static:

                    def w(*a, **kw):  # noqa: ANN202
                        res = _raw(*a, **kw)
                        emit(
                            "predicate",
                            name=_meth,
                            result=bool(res),
                            args=[getattr(x, "name", None) for x in a],
                            nums=_pred_nums(_meth, a),
                        )
                        return res

                    return staticmethod(w)

                def w2(self, *a, **kw):  # noqa: ANN202
                    res = _raw(self, *a, **kw)
                    emit(
                        "predicate",
                        name=_meth,
                        result=bool(res),
                        args=[getattr(x, "name", None) for x in a],
                        nums=_pred_nums(_meth, a),
                    )
                    return res

                return w2

            setattr(A, meth, mk())

    # ---- the trajectory sampler
    from bilevel_planning.trajectory_samplers import parameterized_controller_sampler as pcs

    S = pcs.ParameterizedControllerTrajectorySampler
    _orig_call = S.__call__
    counter = {"n": 0}

    def sampler_call(self, x, s, a, ns, bpg, rng):  # noqa: ANN001, ANN202
        counter["n"] += 1
        n = counter["n"]
        rng_state_before = repr(rng.bit_generator.state)
        emit(
            "sampler_begin",
            call=n,
            abstract_action=str(a),
            s_atoms=sorted(str(y) for y in getattr(s, "atoms", [])),
            ns_atoms=sorted(str(y) for y in getattr(ns, "atoms", [])),
            x_state=dump_state(x),
            rng_state=rng_state_before,
        )
        t0 = time.time()
        # Re-implement the body so we can see the parameters and the rejection diff.
        controller = self._controller_generator(a)  # noqa: SLF001
        params = controller.sample_parameters(x, rng)
        emit("sampler_params", call=n, params=params, rng_state_after=repr(rng.bit_generator.state))
        try:
            res = _orig_call_with_params(self, x, ns, bpg, controller, params, n)
        except pcs.TrajectorySamplingFailure:
            emit("sampler_reject", call=n, wall_s=time.time() - t0)
            raise
        emit("sampler_accept", call=n, wall_s=time.time() - t0, n_actions=len(res[1]))
        return res

    def _orig_call_with_params(self, x, ns, bpg, controller, params, n):  # noqa: ANN001, ANN202
        from bilevel_planning.structs import TransitionFailure

        x_traj = [x]
        u_traj: list[Any] = []
        controller.reset(x, params)
        for step_i in range(self._max_trajectory_steps):  # noqa: SLF001
            if controller.terminated():
                emit("controller_terminated", call=n, step=step_i)
                break
            u = controller.step()
            try:
                nx = self._transition_function(x, u)  # noqa: SLF001
            except TransitionFailure as e:
                emit("transition_failure", call=n, step=step_i, err=repr(e))
                break
            controller.observe(nx)
            x_traj.append(nx)
            u_traj.append(u)
            bpg.add_state_node(nx)
            bpg.add_action_edge(x, u, nx)
            x = nx
        final_state = x_traj[-1]
        final_abstract_state = self._state_abstractor(final_state)  # noqa: SLF001
        bpg.add_abstract_state_node(final_abstract_state)
        bpg.add_state_abstractor_edge(final_state, final_abstract_state)
        fa = sorted(str(y) for y in final_abstract_state.atoms)
        na = sorted(str(y) for y in getattr(ns, "atoms", []))
        emit(
            "sampler_final",
            call=n,
            n_steps=len(u_traj),
            final_atoms=fa,
            target_atoms=na,
            only_in_final=sorted(set(fa) - set(na)),
            only_in_target=sorted(set(na) - set(fa)),
            equal=bool(final_abstract_state == ns),
            final_state=dump_state(final_state),
        )
        if final_abstract_state == ns:
            return x_traj, u_traj
        from bilevel_planning.trajectory_samplers.trajectory_sampler import (
            TrajectorySamplingFailure,
        )

        raise TrajectorySamplingFailure()

    S.__call__ = sampler_call  # type: ignore[assignment]

    # ---- the refiner
    from bilevel_planning.refiners import backtracking_refiner as br

    R = br.BacktrackingRefiner
    _orig_refine = R._refine_from_step  # noqa: SLF001

    def refine(self, index, x, s_plan, a_plan, remaining_time, bpg):  # noqa: ANN001, ANN202
        emit(
            "refine_enter",
            index=index,
            remaining_time=remaining_time,
            a_plan=[str(y) for y in a_plan],
            n_attempts_allowed=self._num_sampling_attempts_per_step,  # noqa: SLF001
            seed=self._seed,  # noqa: SLF001
        )
        out = _orig_refine(self, index, x, s_plan, a_plan, remaining_time, bpg)
        emit("refine_exit", index=index, success=bool(out[0]))
        return out

    R._refine_from_step = refine  # type: ignore[assignment]  # noqa: SLF001


def _pred_nums(meth: str, args: tuple) -> dict:
    """The raw numbers feeding a predicate check."""
    out: dict[str, Any] = {}
    try:
        state = args[0]
        for obj in args[1:]:
            if not hasattr(obj, "name"):
                continue
            from kinder.envs.dynamic3d.object_types import MujocoObjectTypeFeatures

            for f in MujocoObjectTypeFeatures.get(obj.type, []):
                try:
                    out[f"{obj.name}.{f}"] = float(state.get(obj, f))
                except Exception:  # noqa: BLE001, PERF203
                    pass
    except Exception as e:  # noqa: BLE001
        out["err"] = repr(e)
    return out
