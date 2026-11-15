"""ROS 2 gate between the team's command mux and motion_solver."""
import json
from pathlib import Path
import time

from stow_execution import TransportStow
from transport_gate import TransportGate


def main():
    import rclpy
    from lunar_rover_msgs.msg import ManualControl
    from rclpy.node import Node
    from std_msgs.msg import String

    class Interlock(Node):
        def __init__(self):
            super().__init__("m3s_transport_interlock")
            path = self.declare_parameter("config", "config.json").value
            config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            self.gate = TransportGate(TransportStow(config).bounds)
            incoming = self.declare_parameter("input_topic", "/manual_control_ungated").value
            outgoing = self.declare_parameter("output_topic", "/manual_control").value
            if self.resolve_topic_name(incoming) == self.resolve_topic_name(outgoing):
                raise ValueError("Interlock input and output topics must differ")
            self.publisher = self.create_publisher(ManualControl, outgoing, 1)
            self.create_subscription(ManualControl, incoming, self.command, 1)
            self.create_subscription(String, "/m3s/transport_feedback", self.feedback, 1)
            self.ack = self.create_publisher(String, "/m3s/base_lock_ack", 1)
            self.create_subscription(String, "/m3s/base_lock_request", self.lock, 1)
            self.ready = self.create_publisher(String, "/m3s/transport_ready", 1)
            self.release_ack = self.create_publisher(String, "/m3s/base_unlock_ack", 1)
            self.create_subscription(String, "/m3s/base_unlock_request", self.unlock, 1)
            self.create_timer(.05, self.publish)

        def lock(self, msg):
            try:
                request = json.loads(msg.data)
                if not isinstance(request, dict):
                    return
            except ValueError:
                return
            now = time.monotonic()
            if self.gate.lock(request, now):
                self.publisher.publish(ManualControl())
                self.ack.publish(String(data=json.dumps(dict(
                    token=request["token"], locked=True, monotonic_s=now))))

        def unlock(self, msg):
            try:
                request = json.loads(msg.data)
                if not isinstance(request, dict):
                    return
            except ValueError:
                return
            now = time.monotonic()
            if self.gate.unlock(request, now):
                self.release_ack.publish(String(data=json.dumps(dict(
                    token=request["token"], unlocked=True, monotonic_s=now))))

        def command(self, msg):
            if msg.relax:
                self.gate.reset()
                stop = ManualControl()
                stop.relax = True
                self.publisher.publish(stop)
                return
            self.gate.request(msg, time.monotonic())

        def feedback(self, msg):
            try:
                payload = json.loads(msg.data)
                if not isinstance(payload, dict):
                    raise ValueError("Invalid transport feedback")
            except ValueError:
                self.gate.reset()
            else:
                self.gate.feedback(payload, time.monotonic())
            if not self.gate.permitted(time.monotonic()):
                self.publisher.publish(ManualControl())

        def publish(self):
            now = time.monotonic()
            message = self.gate.output(now)
            self.publisher.publish(ManualControl() if message is None else message)
            if self.gate.permitted(now):
                self.ready.publish(String(data=json.dumps(dict(
                    token=self.gate.last_unlocked_token, monotonic_s=now))))

    rclpy.init()
    node = None
    try:
        node = Interlock()
        rclpy.spin(node)
    finally:
        if node:
            node.publisher.publish(ManualControl())
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()




