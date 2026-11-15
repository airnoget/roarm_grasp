"""Check fresh odometry and a matching base-lock acknowledgement before arm motion."""
from collections import deque
import json
import math
import time
import uuid


class StoppedEvidence:
    def __init__(self):
        self.samples = deque()
        self.last_stamp = -math.inf

    def observe(self, stamp, ros_now, now, linear, angular):
        if not math.isfinite(stamp) or stamp <= self.last_stamp:
            self.samples.clear()
            return
        self.last_stamp = stamp
        if (not all(math.isfinite(v) for v in (stamp, ros_now, now, linear, angular))
                or not 0 <= ros_now-stamp <= .25 or abs(linear) > .015 or abs(angular) > .05):
            self.samples.clear()
            return
        if self.samples and (now-self.samples[-1][0] > .25 or stamp-self.samples[-1][1] > .25):
            self.samples.clear()
        self.samples.append((now, stamp))
        while len(self.samples) > 1 and now-self.samples[0][0] > .65:
            self.samples.popleft()

    def ready(self, now):
        return (len(self.samples) >= 3 and 0 <= now-self.samples[-1][0] <= .25
                and self.samples[-1][0]-self.samples[0][0] >= .30
                and self.samples[-1][1]-self.samples[0][1] >= .30)


def lock_base(config):
    import rclpy
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String

    token = uuid.uuid4().hex
    evidence = StoppedEvidence()
    ack_at = None
    rclpy.init(args=[])
    node = rclpy.create_node("m3s_arm_start_handshake")
    try:
        request = node.create_publisher(String, "/m3s/base_lock_request", 1)

        def ack(msg):
            nonlocal ack_at
            try:
                result = json.loads(msg.data)
                now = time.monotonic()
                if (result["token"] == token and result["locked"] is True
                        and 0 <= now-float(result["monotonic_s"]) <= .25):
                    ack_at = now
            except (ValueError, TypeError, KeyError):
                pass

        def odom(msg):
            if ack_at is None:
                return
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
            twist = msg.twist.twist
            linear = math.hypot(twist.linear.x, twist.linear.y, twist.linear.z)
            angular = math.hypot(twist.angular.x, twist.angular.y, twist.angular.z)
            evidence.observe(stamp, node.get_clock().now().nanoseconds*1e-9, time.monotonic(),
                             linear, angular)

        node.create_subscription(String, "/m3s/base_lock_ack", ack, 1)
        from rclpy.qos import qos_profile_sensor_data

        node.create_subscription(Odometry, config.get("odom_topic", "/localization/lio_odometry"),
                                 odom, qos_profile_sensor_data)
        deadline = time.monotonic() + 4
        next_request = 0.
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_request:
                request.publish(String(data=json.dumps(dict(token=token, monotonic_s=now))))
                next_request = now + .10
            rclpy.spin_once(node, timeout_sec=.05)
            now = time.monotonic()
            if ack_at is not None and now-ack_at <= .25 and evidence.ready(now):
                return token
        raise RuntimeError("Base lock or stationary odometry not confirmed; arm motion not started")
    finally:
        node.destroy_node()
        rclpy.shutdown()

