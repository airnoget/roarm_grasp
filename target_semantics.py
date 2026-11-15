"""Confirm the requested semantic class before offering a depth component."""
import numpy as np

COMPETITION_TARGET = "artificial_object"


class TargetDetector:
    def __init__(self, checkpoint, allowed, device="cpu", *, prompts=None):
        from ultralytics import YOLO, YOLOWorld

        self.model = YOLOWorld(checkpoint) if prompts else YOLO(checkpoint, task="detect")
        if prompts:
            self.model.set_classes(list(prompts))
        self.allowed = set(allowed)
        self.device = device
        names = self.model.names
        if not self.allowed or not self.allowed <= set(names.values() if isinstance(names, dict) else names):
            raise ValueError("Requested target classes are absent from detector weights")

    def __call__(self, frame):
        if (frame.bgr is None or frame.bgr.dtype != np.uint8
                or frame.bgr.shape != (*frame.depth_m.shape, 3)):
            raise ValueError("Semantic gating requires color registered to the depth raster")
        from rgbd_projection import supports_native, depth_to_rgb, boxes_to_depth
        native = supports_native(frame)
        self.image_source = 'native_rgb' if native else 'depth_aligned_rgb'
        self.last_depth_to_rgb = None
        self.last_native_detections = []
        image = frame.raw_bgr if native else frame.bgr
        result = self.model.predict(image, device=self.device, conf=.35, verbose=False)[0]
        boxes = result.boxes
        detections = [(result.names[int(cls)], box) for cls, box in
                      zip(boxes.cls.cpu().numpy(), boxes.xyxy.cpu().numpy())]
        if native:
            self.last_native_detections = detections
            self.last_depth_to_rgb = depth_to_rgb(frame)
            return boxes_to_depth(detections, self.last_depth_to_rgb)
        return detections


def support_pixels(valid, detections):
    """Exclude detected objects from support fitting in the aligned raster."""
    support = valid.copy()
    height, width = support.shape
    for _, box in detections:
        if not np.isfinite(box).all():
            raise ValueError("Detector returned nonfinite bounds")
        x0, y0 = np.clip(np.floor(box[:2]), 0, [width, height]).astype(int)
        x1, y1 = np.clip(np.ceil(box[2:]), 0, [width, height]).astype(int)
        if x0 < x1 and y0 < y1:
            support[y0:y1, x0:x1] = False
    return support


def confirmed_components(components, detections, allowed):
    components = list(components)
    pixels = [np.nonzero(component) for component in components]
    positive = [set() for _ in components]
    localized = set()
    veto = set()
    for name, (x0, y0, x1, y1) in detections:
        matches = []
        for index, (v, u) in enumerate(pixels):
            coverage = np.mean((u >= x0) & (u < x1) & (v >= y0) & (v < y1))
            if name in allowed and coverage >= .75:
                positive[index].add(name)
                matches.append(index)
            if name not in allowed and coverage >= .35:
                veto.add(index)
        # A box spanning multiple components cannot establish which one it labels.
        if len(matches) == 1:
            localized.add(matches[0])
    for index, component in enumerate(components):
        if index in localized and index not in veto and len(positive[index]) == 1:
            yield component, next(iter(positive[index]))

