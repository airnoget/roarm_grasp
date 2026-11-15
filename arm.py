import json
import math
import time

from planning import inside
from grip_calibration import held


class Arm:
    def __init__(self, port):
        import serial

        self.port = serial.Serial(port, 115200, timeout=0.2, write_timeout=1, exclusive=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.port.close()

    def send(self, command):
        self.port.write((json.dumps(command, allow_nan=False) + "\n").encode("ascii"))

    def feedback(self, timeout=2, required_fields=None):
        required_fields = tuple(required_fields) if required_fields is not None else ("x", "y", "z", "tit", "r", "g", "tG")
        self.port.reset_input_buffer()
        self.send({"T": 105})
        deadline = time.monotonic() + timeout
        retry_at=time.monotonic()+.5
        pending=bytearray(); lines=0; last_type=None; invalid=[]
        while time.monotonic() < deadline:
            if time.monotonic() >= retry_at:
                self.send({"T":105});retry_at=time.monotonic()+.5
            try:
                chunk=self.port.readline()
                pending.extend(chunk)
                if len(pending)>16384:pending.clear()
                if not chunk.endswith(b'\n'):continue
                raw=bytes(pending);pending.clear();lines+=1
                data = json.loads(raw)
                if time.monotonic() >= deadline:
                    break
                if not isinstance(data,dict):continue
                last_type=data.get('T')
                invalid=[key for key in required_fields if not isinstance(data.get(key),(int,float))
                         or isinstance(data.get(key),bool) or not math.isfinite(data[key])]
                if data.get("T") == 1051 and not invalid:
                    return data
            except (ValueError, KeyError, TypeError, UnicodeError):
                continue
        raise TimeoutError(f'RoArm T=1051 feedback unavailable: lines={lines}, last_T={last_type}, missing_or_invalid={invalid}. Check controller boot/power and selected port; no motion was requested by feedback polling.')

    def check_start(self, approach, config):
        initial = self.feedback()
        if not inside([initial[a] for a in "xyz"], config["workspace_mm"]):
            raise ValueError("Jog the arm to a verified starting pose inside the workspace")
        if initial["z"] < approach["z"] - config["position_tolerance_mm"]:
            raise ValueError("Starting pose must be at or above the approach height")

    def move(self, command, config, on_feedback=None):
        workspace = config.get("workspace_mm")
        if workspace is not None and not inside([command[k] for k in "xyz"], workspace):
            raise ValueError("Motion command is outside the configured workspace")
        self.send(command)
        deadline = time.monotonic() + config["motion_timeout_s"]
        previous = None
        while (remaining := deadline - time.monotonic()) > 0:
            before = time.perf_counter() if on_feedback is not None else None
            state = self.feedback(timeout=min(2, remaining))
            if on_feedback is not None:
                on_feedback(state, (before + time.perf_counter()) / 2)
            if time.monotonic() >= deadline:
                break
            if workspace is not None and not inside([state[k] for k in "xyz"], workspace):
                raise RuntimeError("Measured motion left the configured workspace")
            position = [state[k] for k in "xyz"]
            distance = math.dist(position, [command[k] for k in "xyz"])
            angle_errors = [abs(math.remainder(state[k] - command[j], 2*math.pi))
                            for k, j in (("tit", "t"), ("r", "r"))]
            if distance <= config["position_tolerance_mm"] and max(angle_errors) <= 0.10:
                angles = [state["tit"], state["r"]]
                if previous is not None:
                    last_position, last_angles = previous
                    drift = max(abs(math.remainder(current - last, 2*math.pi))
                                for current, last in zip(angles, last_angles))
                    if math.dist(position, last_position) <= 1.0 and drift <= .02:
                        return state
                previous = position, angles
            else:
                previous = None
            time.sleep(0.1)
        raise TimeoutError("Pose tolerance not reached; sequence aborted")

    def check_grip(self, config, on_feedback=None):
        samples = []
        for _ in range(3):
            state = self.feedback()
            samples.append(state)
            if on_feedback:
                on_feedback(state)
            if not held(state, config["held_angle_rad"], config["held_min_load"]):
                raise RuntimeError("Grip feedback outside measured held-object range")
            time.sleep(0.1)
        return samples



