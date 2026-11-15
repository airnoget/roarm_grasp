import cv2
import numpy as np

from perception import Candidate, projected_span


class Predictor:
    def __init__(self, upstream, checkpoint, device):
        import sys
        from pathlib import Path

        import torch

        upstream = Path(upstream).resolve()
        if not (upstream / "models" / "ggcnn.py").is_file():
            raise ValueError("Provide the official dougsm/ggcnn checkout")
        sys.path.insert(0, str(upstream))
        try:
            # Official release files serialize the model object. Only load trusted files.
            model = torch.load(checkpoint, map_location=device, weights_only=False)
        finally:
            sys.path.pop(0)
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Expected a full upstream GG-CNN model checkpoint")
        self.model = model.to(device).eval()
        self.torch = torch
        self.device = device

    def predict(self, depth, roi, components, min_score, surface_z_mm=None):
        x, y, size = roi
        crop = depth[y:y+size, x:x+size].astype(np.float32).copy()
        bad = ~np.isfinite(crop) | (crop <= 0)
        if bad.mean() > 0.25:
            raise ValueError("Too many depth holes inside GG-CNN ROI")
        crop[bad] = 0
        crop = cv2.inpaint(crop, bad.astype(np.uint8), 2, cv2.INPAINT_NS)
        image = cv2.resize(crop, (300, 300), interpolation=cv2.INTER_LINEAR)
        image = np.clip(image - image.mean(), -1, 1)
        tensor = self.torch.from_numpy(image)[None, None].to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(tensor)
        q, c, s, w = [output.detach().cpu().numpy()[0, 0] for output in outputs]
        if any(m.shape != (300, 300) or not np.isfinite(m).all() for m in (q, c, s, w)):
            raise ValueError("Unexpected GG-CNN output")
        q, c, s = [cv2.GaussianBlur(m, (0, 0), 2) for m in (q, c, s)]
        w = cv2.GaussianBlur(w, (0, 0), 1)
        scale = size / 300
        u_map = x + np.floor((np.arange(300) + 0.5) * scale).astype(int)
        v_map = y + np.floor((np.arange(300) + 0.5) * scale).astype(int)
        peak_window = np.ones((11, 11), np.uint8)
        for mask in components:
            centers = mask
            if surface_z_mm is not None:
                # Keep side-wall pixels for the opening estimate but not for the center.
                top_z = np.percentile(surface_z_mm[mask], 85)
                centers = mask & (surface_z_mm >= top_z - 5.)
            eligible = centers[np.ix_(v_map, u_map)] & (q >= min_score) & (w > 0)
            # A stronger response on another object must not suppress this target.
            masked_quality = np.where(eligible, q, np.finfo(q.dtype).min)
            peaks = eligible & (q == cv2.dilate(masked_quality, peak_window))
            rows, cols = np.nonzero(peaks)
            seen = set()
            for i in np.argsort(-q[rows, cols], kind="stable"):
                row, col = rows[i], cols[i]
                pixel = int(u_map[col]), int(v_map[row])
                if pixel in seen:
                    continue
                seen.add(pixel)
                angle = -0.5 * np.arctan2(s[row, col], c[row, col])
                yield Candidate(*pixel, angle,
                                projected_span(mask, angle), float(q[row, col]),
                                model_width_px=float(w[row, col] * 150 * scale))
                if len(seen) == 10:
                    break
