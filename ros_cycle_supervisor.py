"""Coordinate exclusive serial ownership and verified legacy ROS grasp results."""
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from ros_stowed_heartbeat import last_motion_record, log_token


ROOT = Path(__file__).resolve().parent


class CycleSupervisor:
    def __init__(self, start, pick_args, config, log_dir, on_success, on_fault):
        self.start = start
        self.pick_args = pick_args
        self.config = config
        self.log_dir = log_dir
        self.on_success = on_success
        self.on_fault = on_fault
        self.phase = "idle"
        self.child = None
        self.deadline = 0.
        self.token = None
        self.mission_state = None
        self.pending_trigger_until = None
        self.trigger_consumed = False

    def bootstrap(self):
        if self.phase != "idle":
            raise ValueError("Transport bootstrap requires an idle supervisor")
        self.child = self.start(["ros_stowed_heartbeat.py", str(self.config), "--initial"])
        self.phase = "transport"

    def mission_update(self, state, now):
        previous = self.mission_state
        self.mission_state = state
        if self.phase in ("serial_handoff", "picking", "await_transport"):
            if state not in ("TARGET_PENDING", "GRASPING"):
                self.fault(f"Mission left grasp phase: {state}")
            return
        if state == "SEARCHING" and previous != state:
            self.trigger_consumed = False
        if state in ("TARGET_PENDING", "GRASPING") and self.pending_trigger_until is not None:
            deadline = self.pending_trigger_until
            self.pending_trigger_until = None
            if now <= deadline:
                self.trigger(now)
        elif state not in ("SEARCHING", "TARGET_PENDING", "GRASPING"):
            self.pending_trigger_until = None

    def trigger(self, now):
        if self.trigger_consumed or self.phase not in ("idle", "transport"):
            return False
        if self.mission_state == "SEARCHING":
            if self.pending_trigger_until is None or now > self.pending_trigger_until:
                self.pending_trigger_until = now + .5
            return False
        if self.mission_state not in ("TARGET_PENDING", "GRASPING"):
            return False
        self.trigger_consumed = True
        if self.phase not in ("idle", "transport"):
            return False
        self.sample = uuid.uuid4().hex
        self.log = self.log_dir / f"{self.sample}.jsonl"
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
        self.phase = "serial_handoff"
        self.deadline = now + 3
        return True

    def fault(self, reason):
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
        self.phase = "fault"
        self.pending_trigger_until = None
        self.on_fault(reason)

    def tick(self, now):
        if self.phase in ("idle", "fault"):
            return
        if self.phase != "transport" and now > self.deadline:
            self.fault(f"{self.phase} timeout")
            return
        status = self.child.poll() if self.child is not None else 0
        try:
            if self.phase == "serial_handoff" and status is not None:
                self.child = self.start(["run.py", *self.pick_args, "--execute", "--store",
                                         "--sample-id", self.sample, "--motion-log", str(self.log)])
                self.phase, self.deadline = "picking", now + 100
            elif self.phase == "picking" and status == 0:
                self.token = log_token(self.log)
                self.child = self.start(["ros_stowed_heartbeat.py", str(self.config),
                                         "--motion-log", str(self.log)])
                self.phase, self.deadline = "await_transport", now + 8
            elif self.phase != "serial_handoff" and status is not None:
                reason = f"{self.phase} worker exited with code {status}"
                if self.phase == "picking":
                    try:
                        record = last_motion_record(self.log)
                        if record.get("outcome") == "aborted" and isinstance(record.get("reason"), str):
                            reason += f" ({record.get('stage')}): {record['reason'][:400]}"
                    except (OSError, ValueError):
                        pass
                self.fault(reason)
        except (ValueError, OSError, RuntimeError) as error:
            self.fault(str(error))

    def ready(self, payload, now):
        if (self.phase != "await_transport" or self.child.poll() is not None
                or self.mission_state != "GRASPING" or now > self.deadline):
            return
        try:
            valid = payload["token"] == self.token and 0 <= now-float(payload["monotonic_s"]) <= .25
        except (TypeError, ValueError, KeyError):
            valid = False
        if valid:
            self.phase = "transport"
            self.on_success()


def main():
    import argparse
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Empty, String
    from target_semantics import COMPETITION_TARGET

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--camera", choices=["orbbec", "realsense"], required=True)
    parser.add_argument("--target-weights", type=Path, required=True)
    parser.add_argument("--target-class", choices=[COMPETITION_TARGET], default=COMPETITION_TARGET)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--trusted-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if not (args.upstream.is_dir() and args.trusted_checkpoint.is_file() and args.target_weights.is_file()):
        parser.error("Provide an existing GG-CNN source directory and both local weight files")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if (config.get("require_base_interlock") is not True
            or config.get("camera_mount") != "wrist"
            or not (config.get("storage") or {}).get("verify_depth")
            or not (config.get("storage") or {}).get("stow")):
        parser.error("Coordinator requires wrist mode, base interlock, depth storage and stow")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    pick_args = ["--config", str(args.config.resolve()), "--camera", args.camera,
                 "--target-weights", str(args.target_weights.resolve()), "--target-class", args.target_class,
                 "--upstream", str(args.upstream.resolve()),
                 "--trusted-checkpoint", str(args.trusted_checkpoint.resolve())]

    class Coordinator(Node):
        def __init__(self):
            super().__init__("m3s_cycle_supervisor")
            self.results = self.create_publisher(String, "/grasp/result", 10)
            self.status = self.create_publisher(String, "/m3s/cycle_status", 10)
            self.cycle = CycleSupervisor(
                lambda command: subprocess.Popen([sys.executable, str(ROOT / command[0]), *command[1:]]),
                pick_args, args.config.resolve(), args.log_dir.resolve(),
                lambda: self.results.publish(String(data="SUCCESS")),
                lambda reason: self.status.publish(String(data=f"FAULT: {reason}")),
            )
            self.create_subscription(String, "/mission/state", self.state, 10)
            self.create_subscription(Empty, "/grasp/trigger", self.trigger, 10)
            self.create_subscription(String, "/grasp/command", self.command, 10)
            self.create_subscription(String, "/m3s/transport_ready", self.ready, 10)
            self.create_timer(.05, lambda: self.cycle.tick(time.monotonic()))
            self.cycle.bootstrap()

        def state(self, msg):
            self.cycle.mission_update(msg.data, time.monotonic())

        def trigger(self, _):
            self.cycle.trigger(time.monotonic())


        def command(self, msg):
            command = msg.data.strip().lower()
            if command in ("abort", "stop", "cancel", "unload"):
                self.cycle.fault(f"Mission command {command}; no verified M3-S unload controller")

        def ready(self, msg):
            try:
                payload = json.loads(msg.data)
            except ValueError:
                return
            self.cycle.ready(payload, time.monotonic())

    rclpy.init(args=[])
    node = None
    try:
        node = Coordinator()
        rclpy.spin(node)
    finally:
        if node:
            if node.cycle.child is not None and node.cycle.child.poll() is None:
                node.cycle.child.terminate()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

