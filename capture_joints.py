"""Record joint feedback while the arm is operated independently."""
import argparse
import csv
from pathlib import Path
import time

from arm import Arm


FIELDS = ("time_s", "b", "s", "e", "t", "r", "g", "x", "y", "z")


def record(arm, stream, seconds):
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        before = time.monotonic()
        feedback = arm.feedback(timeout=.15)
        after = time.monotonic()
        writer.writerow(dict(time_s=(before + after)/2-start,
                             **{key: feedback[key] for key in FIELDS[1:]}))
        stream.flush()
        time.sleep(max(0., .05 - (after-before)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=float, default=10.)
    args = parser.parse_args()
    if not 0 < args.seconds <= 120:
        parser.error("--seconds must be in (0,120]")
    with args.output.open("x", encoding="utf-8", newline="") as output, Arm(args.port) as arm:
        record(arm, output, args.seconds)


if __name__ == "__main__":
    main()

