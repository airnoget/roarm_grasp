"""Explicit onboard serial handoff after a confirmed deposit and stow."""
import argparse
import json
from pathlib import Path
import time

from arm import Arm
from stow_execution import JOINTS, TransportStow


def last_motion_record(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    record = json.loads(lines[-1]) if lines else {}
    if not isinstance(record, dict):
        raise ValueError("Invalid motion log record")
    return record


def log_token(path):
    record = last_motion_record(path)
    token = record.get("base_lock_token")
    if (record.get("stage") != "finished" or record.get("outcome") != "deposit_depth_confirmed"
            or record.get("arm_stowed") is not True or not isinstance(token, str) or not token):
        raise ValueError("Handoff needs a completed depth-confirmed deposit, stow and lock token")
    return token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--motion-log", type=Path)
    mode.add_argument("--initial", action="store_true", help="Publish feedback without unlocking")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    token = log_token(args.motion_log) if args.motion_log else None
    TransportStow(config)

    import rclpy
    from std_msgs.msg import String

    rclpy.init(args=[])
    node = rclpy.create_node("m3s_transport_feedback_owner")
    try:
        feedback = node.create_publisher(String, "/m3s/transport_feedback", 1)
        request = node.create_publisher(String, "/m3s/base_unlock_request", 1)
        unlocked = args.initial

        def acknowledge(message):
            nonlocal unlocked
            try:
                reply = json.loads(message.data)
                if (reply["token"] == token and reply["unlocked"] is True
                        and 0 <= time.monotonic()-float(reply["monotonic_s"]) <= .25):
                    unlocked = True
            except (ValueError, KeyError, TypeError):
                pass

        node.create_subscription(String, "/m3s/base_unlock_ack", acknowledge, 1)
        with Arm(config["port"]) as arm:
            deadline = time.monotonic() + 4
            while rclpy.ok():
                now = time.monotonic()
                state = arm.feedback(timeout=.15)
                timestamp = time.monotonic()
                feedback.publish(String(data=json.dumps(dict(
                    monotonic_s=timestamp, busy=False, **{key: state[key] for key in JOINTS}))))
                if not unlocked:
                    if now > deadline:
                        raise RuntimeError("No verified transport unlock acknowledgement")
                    request.publish(String(data=json.dumps(dict(token=token, monotonic_s=timestamp))))
                rclpy.spin_once(node, timeout_sec=.02)
                time.sleep(max(0, .07 - (time.monotonic()-now)))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

