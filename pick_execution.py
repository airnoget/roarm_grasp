"""Execute the grasp in stages and preserve raw grip feedback."""
from contextlib import nullcontext
import json
import math
import time
from storage_execution import plan_storage, release
from stow_execution import TransportStow


def execute_pick(arm, config, commands, log_path=None, store=False, observe_bin=None,
                 sample_id=None, target_class=None, base_lock_token=None, proof_only=False,
                 observe_proof=None):
    if proof_only and store:
        raise ValueError("Proof lift only cannot execute storage")
    above, descend, close, lift = commands
    tolerance = config["position_tolerance_mm"]
    rise = config.get("proof_lift_mm", 12.)
    full_rise = lift["z"] - descend["z"]
    if not math.isfinite(rise) or not 0 < 2 * tolerance < rise < full_rise:
        raise ValueError("Proof lift must exceed twice pose tolerance and stay below full lift")
    proof = dict(lift, z=descend["z"] + rise, spd=min(lift["spd"], .10))
    storage = plan_storage(config, lift) if store else None
    stow = TransportStow(config) if storage and config["storage"].get("stow") is not None else None
    verifier = None
    if storage and config["storage"].get("verify_depth", False):
        from bin_depth import BinDepth

        if observe_bin is None:
            raise ValueError("Bin depth verification requires wrist observations")
        verifier = BinDepth(config["storage"])
    start = time.perf_counter()
    connection = open(log_path, "x", encoding="utf-8") if log_path else nullcontext()
    with connection as log:
        def record(stage, sampled_at=None, **values):
            if log:
                timestamp = time.perf_counter() if sampled_at is None else sampled_at
                log.write(json.dumps(dict(stage=stage, elapsed_s=timestamp-start,
                                          **values), allow_nan=False) + "\n")
                log.flush()

        def move(stage, command):
            record(stage, command=command)
            if log:
                def sample(state, sampled_at):
                    record(stage, sampled_at, feedback=state, motion_sample=True)

                state = arm.move(command, config, on_feedback=sample)
                record(stage, motion_complete=True)
            else:
                state = arm.move(command, config)
            return state

        deposit = None
        arm_stowed = False
        stow_origin = "release"
        stage = "approach"
        try:
            for stage, command in (("approach", above), ("descend", descend)):
                state = move(stage, command)
                if not math.isfinite(state["g"]) or abs(state["g"] - config["open_rad"]) > .08:
                    raise RuntimeError(f"Gripper opening was not confirmed at {stage}")
            stage = "close"
            record(stage, command=close)
            arm.send(close)
            time.sleep(config["grip_wait_s"])
            closed = arm.check_grip(config, lambda state: record(stage, feedback=state))
            stage = "proof_lift"
            move(stage, proof)
            verified = arm.check_grip(config, lambda state: record(stage, feedback=state))
            if min(row["z"] for row in verified) - max(row["z"] for row in closed) < rise - tolerance:
                raise RuntimeError("Observed proof lift rise below required clearance")
            if proof_only:
                if observe_proof is not None:
                    stage = "proof_observation"
                    observe_proof()
                    record(stage, file="proof-lift.npz")
                    arm.check_grip(config, lambda state: record(stage, feedback=state))
                outcome = "proof_lift_telemetry_passed"
                record("finished", outcome=outcome, sample_id=sample_id, target_class=target_class,
                       arm_stowed=False, base_lock_token=base_lock_token)
                return outcome
            stage = "full_lift"
            move(stage, lift)
            arm.check_grip(config, lambda state: record(stage, feedback=state))
            if storage:
                waypoints, opening, minimum_z, view = storage
                stage = "transfer"
                for command in waypoints:
                    move(stage, command)
                    arm.check_grip(config, lambda state: record(stage, feedback=state))
                stage = "release"
                before = None
                if verifier:
                    if view:
                        stage = "bin_reference_view"
                        move(stage, view)
                        arm.check_grip(config, lambda state: record(stage, feedback=state))
                    stage = "bin_reference"
                    before = verifier.reference(observe_bin())
                    if view:
                        stage = "return_to_release"
                        move(stage, waypoints[-1])
                        arm.check_grip(config, lambda state: record(stage, feedback=state))
                stage = "release"
                command, state = release(arm, config, opening, minimum_z)
                record(stage, command=command, feedback=state)
                arm.send(command)
                time.sleep(config["grip_wait_s"])
                state = arm.feedback()
                record(stage, feedback=state)
                if abs(state["g"] - config["open_rad"]) > .08:
                    raise RuntimeError("Gripper opening was not confirmed")
                if before is not None:
                    if view:
                        stage = "bin_verification_view"
                        open_view = dict(view, g=config["open_rad"])
                        move(stage, open_view)
                        stow_origin = "view"
                    try:
                        after = [verifier.height_map(observe_bin()) for _ in range(3)]
                        deposit = verifier.compare(before, after)
                    except (ValueError, RuntimeError, OSError) as error:
                        deposit = dict(confirmed=False, reason=str(error))
                    record("bin_verification", **deposit)
                if stow:
                    stage = "stow"
                    for command in stow.routes[stow_origin]:
                        move(stage, command)
                    stow.check(arm, lambda state: record(stage, feedback=state))
                    arm_stowed = True
        except (ValueError, RuntimeError, TimeoutError, OSError) as error:
            record(stage, outcome="aborted", reason=str(error), arm_stowed=False)
            raise
        outcome = "released_unverified" if storage else "grip_telemetry_passed"
        if deposit and deposit["confirmed"]:
            outcome = "deposit_depth_confirmed"
        record("finished", outcome=outcome, sample_id=sample_id, target_class=target_class,
               arm_stowed=arm_stowed, base_lock_token=base_lock_token)
        return outcome






