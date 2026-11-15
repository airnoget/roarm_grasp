"""Fail-closed lease for chassis motion based on fresh, stationary arm feedback."""
from collections import deque
import math

import numpy as np

from stow_execution import JOINTS


class TransportGate:
    def __init__(self, bounds):
        self.bounds = np.asarray(bounds, float)
        if (self.bounds.shape != (2, 6) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[0] >= self.bounds[1])
                or np.any(self.bounds[1] - self.bounds[0] > .25)):
            raise ValueError("Invalid commissioned transport bounds")
        self.samples = deque(maxlen=3)
        self.last_source = -math.inf
        self.lock_token = None
        self.locked_at = -math.inf
        self.last_unlocked_token = None
        self.command = None

    def reset(self):
        self.samples.clear()
        self.command = None

    def lock(self, request, now):
        try:
            token = request["token"]
            stamp = float(request["monotonic_s"])
        except (KeyError, TypeError, ValueError):
            return False
        if not isinstance(token, str) or not token or not 0 <= now-stamp <= .25:
            return False
        if self.lock_token is not None:
            return self.lock_token == token
        if token == self.last_unlocked_token:
            return False
        self.lock_token = token
        self.locked_at = now
        self.reset()
        return True

    def unlock(self, request, now):
        try:
            token = request["token"]
            stamp = float(request["monotonic_s"])
        except (KeyError, TypeError, ValueError):
            return False
        if not isinstance(token, str) or not token or not 0 <= now-stamp <= .25:
            return False
        if self.lock_token is None:
            return token == self.last_unlocked_token
        if token != self.lock_token or not self.stationary(now):
            return False
        self.last_unlocked_token = token
        self.lock_token = None
        self.reset()
        return True

    def feedback(self, message, now):
        try:
            source = float(message["monotonic_s"])
            joints = np.asarray([message[key] for key in JOINTS], float)
            valid = (message.get("busy") is False and 0 <= now-source <= .25
                     and source > max(self.last_source, self.locked_at) and np.isfinite(joints).all()
                     and np.all(joints >= self.bounds[0]) and np.all(joints <= self.bounds[1]))
        except (KeyError, ValueError, TypeError, OverflowError):
            self.reset()
            return
        if not valid:
            self.reset()
            return
        self.last_source = source
        if self.samples and source-self.samples[-1][0] > .25:
            self.reset()
        self.samples.append((source, joints))
        if len(self.samples) == 3 and np.max(np.ptp([q for _, q in self.samples], axis=0)) > .015:
            self.reset()

    def stationary(self, now):
        if self.samples and not 0 <= now-self.samples[-1][0] <= .25:
            self.reset()
        valid = len(self.samples) == 3 and self.samples[-1][0] - self.samples[0][0] >= .10
        if not valid:
            self.command = None
        return valid

    def permitted(self, now):
        return self.stationary(now) and self.lock_token is None

    def request(self, command, now):
        self.command = (now, command) if self.permitted(now) else None

    def output(self, now):
        if not self.permitted(now) or self.command is None:
            return None
        timestamp, payload = self.command
        return payload if 0 <= now-timestamp <= .20 else None



