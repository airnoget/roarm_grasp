"""Audit XH-202605 score evidence without sending ROS messages or robot commands."""
import argparse
import json
import math
from pathlib import Path


LIMIT_S = 600.


def score(events, allowed, zone):
    previous = -1.
    phase = "start"
    stored, unloaded, retained = set(), set(), set()
    points = {"departure": 0, "collection": 0, "return": 0, "unload": 0}
    rejected = []
    x0, y0, x1, y1 = zone
    if not all(math.isfinite(x) for x in zone) or not math.isclose(x1-x0, .5) or not math.isclose(y1-y0, .5):
        raise ValueError("The unloading zone must be a measured 0.5 m × 0.5 m rectangle")
    for index, event in enumerate(events):
        now = float(event["time_s"])
        if not math.isfinite(now) or now < 0 or now < previous:
            raise ValueError("Event times must be finite, nonnegative and ordered")
        previous = now
        kind = event["event"]
        if now > LIMIT_S:
            rejected.append(dict(index=index, reason="outside_600_s"))
            continue
        reason = None
        sample = event.get("sample_id")
        if kind == "departed":
            if phase == "start" and event.get("base_clear") is True:
                points["departure"] = 10
                phase = "collect"
            else:
                reason = "departure_not_verified_or_repeated"
        elif kind == "stored":
            if phase != "collect" or not sample or sample in stored:
                reason = "invalid_or_duplicate_sample_event"
            elif event.get("target_class") not in allowed or event.get("deposit_depth_confirmed") is not True:
                reason = "unconfirmed_target_or_deposit"
            else:
                stored.add(sample)
                points["collection"] += 10
        elif kind == "returned":
            if phase != "collect" or event.get("base_inside") is not True or event.get("stopped") is not True:
                reason = "base_return_unverified"
            else:
                retained = set(event.get("retained_samples", []))
                if not retained <= stored:
                    raise ValueError("Return evidence names samples absent from verified storage")
                points["return"] = 10 if retained else 0
                phase = "unload"
        elif kind == "unloaded":
            if phase != "unload" or sample not in retained or sample in unloaded:
                reason = "unload_out_of_order_or_duplicate"
            elif event.get("settled") is not True or "footprint_m" not in event:
                reason = "no_settled_landing_evidence"
            else:
                left, near, right, far = event["footprint_m"]
                if not (x0 <= left < right <= x1 and y0 <= near < far <= y1):
                    reason = "sample_footprint_outside_zone"
                else:
                    unloaded.add(sample)
                    points["unload"] += 20
        else:
            reason = "unsupported_event"
        if reason:
            rejected.append(dict(index=index, event=kind, reason=reason))
    return dict(points=points, total=sum(points.values()), stored=sorted(stored),
                retained=sorted(retained), unloaded=sorted(unloaded), rejected=rejected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Path, help="JSON with allowed_classes and unload_zone_m")
    parser.add_argument("attempts", type=Path, nargs=2, help="Two independent JSONL attempt logs")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    scenario = json.loads(args.scenario.read_text(encoding="utf-8-sig"))
    if args.attempts[0].resolve() == args.attempts[1].resolve():
        parser.error("Provide two independent attempts")
    allowed = set(scenario["allowed_classes"])
    if not allowed or not allowed <= {"wood_block", "artificial_object"}:
        parser.error("Allowed classes must be drawn from the designated target classes")
    attempts = []
    for file in args.attempts:
        events = [json.loads(line) for line in file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        attempts.append(score(events, allowed, scenario["unload_zone_m"]))
    result = json.dumps(dict(scope="recorded_evidence", total=sum(x["total"] for x in attempts),
                             attempts=attempts), indent=2)
    print(result)
    if args.report:
        args.report.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
