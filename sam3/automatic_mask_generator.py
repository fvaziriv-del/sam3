# automatic_mask_generator_sam3.py
# Converted automatic mask generator (point-based) for SAM3
# - Uses: build_sam3_image_model(), Sam3Processor
# - Produces: same-style outputs as original SAM2 AutomaticMaskGenerator
# Note: If your local sam3 API uses slightly different method names for prediction,
# the code will try several common alternatives; otherwise adjust the small wrapper.

from typing import Any, Dict, List, Optional, Tuple, Sequence
from collections import deque

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

# SAM3 imports
try:
    # these module paths assume repo structure like facebookresearch/sam3
    from sam3.model_builder import build_sam3_image_model
    from sam3.image_processor.sam3_image_processor import Sam3Processor
except Exception:
    # If your repo paths differ, adjust imports accordingly.
    raise ImportError(
        "Couldn't import SAM3 model/processor. Make sure sam3 is installed and import paths are correct."
    )

# --- RLE / COCO helpers (uses pycocotools if available) ---
try:
    from pycocotools import mask as coco_mask  # type: ignore

    def coco_encode_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
        return coco_mask.encode(np.asfortranarray(binary_mask.astype("uint8")))

    def area_from_rle(rle: Dict[str, Any]) -> int:
        return int(coco_mask.area(rle))

    def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
        return coco_mask.decode(rle)

except Exception:
    # Fallback lightweight RLE-like behavior (dense)
    def coco_encode_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
        h, w = binary_mask.shape
        return {"size": [h, w], "counts": None, "dense_mask": binary_mask.astype(np.uint8)}

    def area_from_rle(rle: Dict[str, Any]) -> int:
        if "dense_mask" in rle:
            return int(rle["dense_mask"].sum())
        return 0

    def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
        return rle.get("dense_mask", np.zeros(tuple(rle.get("size", (0, 0))), dtype=np.uint8))


# --- Small utility helpers (compact versions) ---
def build_all_layer_point_grids(n_per_side: int, n_layers: int = 1, scale_per_layer: int = 1) -> List[np.ndarray]:
    grids: List[np.ndarray] = []
    for layer in range(max(1, n_layers + 1)):
        scale = scale_per_layer ** layer
        n = max(1, int(n_per_side / (scale if scale > 0 else 1)))
        xs = (np.arange(n) + 0.5) / n
        ys = (np.arange(n) + 0.5) / n
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        grids.append(pts)
    return grids


def generate_crop_boxes(im_size: Tuple[int, int], n_layers: int, overlap_ratio: float = 512 / 1500):
    h, w = im_size
    crop_boxes = []
    layer_idxs = []
    # If n_layers == 0 -> just full image
    if n_layers <= 0:
        return [[0, 0, w, h]], [0]
    for layer in range(n_layers):
        # layer_scale: how many splits (2**layer)
        n_cells = 2 ** layer
        crop_h = max(1, int(np.ceil(h / n_cells)))
        crop_w = max(1, int(np.ceil(w / n_cells)))
        # overlap in pixels
        step_h = max(1, int(crop_h * (1 - overlap_ratio)))
        step_w = max(1, int(crop_w * (1 - overlap_ratio)))
        for y0 in range(0, max(1, h - crop_h + 1), step_h):
            for x0 in range(0, max(1, w - crop_w + 1), step_w):
                x1 = min(w, x0 + crop_w)
                y1 = min(h, y0 + crop_h)
                crop_boxes.append([x0, y0, x1, y1])
                layer_idxs.append(layer)
    if len(crop_boxes) == 0:
        crop_boxes = [[0, 0, w, h]]
        layer_idxs = [0]
    return crop_boxes, layer_idxs


def uncrop_masks(masks: torch.Tensor, crop_box: Sequence[int], orig_h: int, orig_w: int) -> torch.Tensor:
    # masks: (N, Hc, Wc)
    x0, y0, x1, y1 = crop_box
    ch, cw = masks.shape[-2], masks.shape[-1]
    out = torch.zeros((masks.shape[0], orig_h, orig_w), device=masks.device, dtype=masks.dtype)
    out[:, y0 : y0 + ch, x0 : x0 + cw] = masks
    return out


def uncrop_points(points: torch.Tensor, crop_box: Sequence[int]) -> torch.Tensor:
    # points: (N, 2) in absolute coords
    x0, y0, _, _ = crop_box
    return points + torch.tensor([x0, y0], device=points.device, dtype=points.dtype)


def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: Sequence[int]) -> torch.Tensor:
    x0, y0, _, _ = crop_box
    offset = torch.tensor([x0, y0, x0, y0], device=boxes.device, dtype=boxes.dtype)
    return boxes + offset


def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    n = masks.shape[0]
    boxes = []
    for i in range(n):
        m = masks[i].bool()
        if m.any():
            ys = torch.where(m.any(dim=1))[0]
            xs = torch.where(m.any(dim=0))[0]
            if len(xs) == 0 or len(ys) == 0:
                boxes.append(torch.tensor([0, 0, 0, 0], device=masks.device))
            else:
                x0, x1 = int(xs[0]), int(xs[-1]) + 1
                y0, y1 = int(ys[0]), int(ys[-1]) + 1
                boxes.append(torch.tensor([x0, y0, x1, y1], device=masks.device))
        else:
            boxes.append(torch.tensor([0, 0, 0, 0], device=masks.device))
    return torch.stack(boxes, dim=0)


def remove_small_regions(mask: np.ndarray, area_thresh: int = 20, mode: str = "holes") -> Tuple[np.ndarray, bool]:
    # naive flood-fill component filter (slow but dependency-free)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    out = mask.copy()
    changed = False
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not visited[i, j]:
                q = deque([(i, j)])
                comp = []
                visited[i, j] = 1
                while q:
                    y, x = q.popleft()
                    comp.append((y, x))
                    for dy, dx in dirs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = 1
                            q.append((ny, nx))
                if len(comp) < area_thresh:
                    changed = True
                    for (y, x) in comp:
                        out[y, x] = 0
    return out, changed


# --- MaskData: lightweight container (similar semantics to original) ---
class MaskData:
    def __init__(self):
        # store lists then convert to tensors as needed
        self.rles: List[Dict[str, Any]] = []
        self.boxes: List[torch.Tensor] = []
        self.iou_preds: List[torch.Tensor] = []
        self.points: List[torch.Tensor] = []
        self.stability_score: List[torch.Tensor] = []
        self.crop_boxes: List[List[int]] = []
        self.low_res_masks: List[torch.Tensor] = []

    def cat(self, other: "MaskData"):
        self.rles.extend(other.rles)
        self.boxes.extend(other.boxes)
        self.iou_preds.extend(other.iou_preds)
        self.points.extend(other.points)
        self.stability_score.extend(other.stability_score)
        self.crop_boxes.extend(other.crop_boxes)
        self.low_res_masks.extend(other.low_res_masks)

    def to_tensors(self, device: Optional[torch.device] = None):
        device = device or torch.device("cpu")
        if len(self.boxes) > 0:
            self.boxes = torch.stack(self.boxes).to(device)
        else:
            self.boxes = torch.empty((0, 4), device=device)
        if len(self.iou_preds) > 0:
            self.iou_preds = torch.stack(self.iou_preds).to(device)
        else:
            self.iou_preds = torch.empty((0,), device=device)
        if len(self.points) > 0:
            self.points = torch.stack(self.points).to(device)
        else:
            self.points = torch.empty((0, 2), device=device)
        if len(self.stability_score) > 0:
            self.stability_score = torch.stack(self.stability_score).to(device)
        else:
            self.stability_score = torch.empty((0,), device=device)
        if len(self.low_res_masks) > 0:
            self.low_res_masks = torch.stack(self.low_res_masks).to(device)
        else:
            self.low_res_masks = torch.empty((0, 1, 1), device=device)

    def filter(self, keep_idx: Sequence[int]):
        keep_idx = list(keep_idx)
        self.rles = [self.rles[i] for i in keep_idx]
        if isinstance(self.boxes, torch.Tensor):
            self.boxes = self.boxes[keep_idx]
        else:
            self.boxes = [self.boxes[i] for i in keep_idx]
        if isinstance(self.iou_preds, torch.Tensor):
            self.iou_preds = self.iou_preds[keep_idx]
        else:
            self.iou_preds = [self.iou_preds[i] for i in keep_idx]
        self.points = [self.points[i] for i in keep_idx]
        self.stability_score = [self.stability_score[i] for i in keep_idx]
        self.crop_boxes = [self.crop_boxes[i] for i in keep_idx]
        if isinstance(self.low_res_masks, torch.Tensor):
            self.low_res_masks = self.low_res_masks[keep_idx]
        else:
            self.low_res_masks = [self.low_res_masks[i] for i in keep_idx]

    def __len__(self):
        return len(self.rles)


# ------------------------------
# SAM3 Automatic Mask Generator
# ------------------------------
class SAM3AutomaticMaskGenerator:
    def __init__(
        self,
        model: Optional[Any] = None,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        multimask_output: bool = True,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        """
        SAM3 Automatic Mask Generator (point-based).
        If model is None, build_sam3_image_model() is called with kwargs.
        """

        if model is None:
            model = build_sam3_image_model(**kwargs)
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Sam3Processor expects model (and handles transforms)
        self.processor = Sam3Processor(self.model).to(self.device) if hasattr(Sam3Processor, "to") else Sam3Processor(self.model)
        # point grids
        assert (points_per_side is None) != (point_grids is None), "Provide exactly one of points_per_side or point_grids"
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(points_per_side, crop_n_layers, crop_n_points_downscale_factor)
        else:
            self.point_grids = point_grids

        assert output_mode in ["binary_mask", "uncompressed_rle", "coco_rle"]
        if output_mode == "coco_rle":
            try:
                import pycocotools  # noqa: F401
            except Exception as e:
                raise ImportError("Please install pycocotools for coco_rle output") from e

        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.multimask_output = multimask_output

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM3AutomaticMaskGenerator":
        # A simple loader: if you have a HF model or local path, adapt this to load weights
        model = build_sam3_image_model(model_id=model_id, **kwargs)
        return cls(model=model, **kwargs)

    # --------------------------
    # Prediction wrapper: tries several likely method names on Sam3Processor
    # --------------------------
    def _processor_predict(
        self,
        state: Any,
        points: torch.Tensor,
        labels: torch.Tensor,
        multimask_output: bool = True,
        return_logits: bool = True,
        mask_input: Optional[torch.Tensor] = None,
    ):
        """
        Try multiple processor API names and return (masks, iou_preds, low_res_masks)
        masks: Tensor (B, M, Hc, Wc) or (B*M,Hc,Wc) depending on implementation
        iou_preds: Tensor (B, M)
        low_res_masks: Tensor (B, M, Hlow, Wlow) or None
        """
        # Prepare possible method signatures to try
        candidates = [
            "predict",  # common name
            "predict_points",
            "predict_from_points",
            "predict_point_prompt",
            "predict_for_points",
        ]
        args = {
            "state": state,
            "point_coords": points,
            "point_labels": labels,
            "multimask_output": multimask_output,
            "return_logits": return_logits,
            "mask_input": mask_input,
        }
        # try attribute names and flexible kwargs (some implementations accept different names)
        for name in candidates:
            fn = getattr(self.processor, name, None)
            if fn is None:
                continue
            try:
                # try calling with the kwargs; some functions may ignore unknown kwargs
                out = fn(**args)
                # Expect out as dict-like or tuple
                if isinstance(out, dict):
                    masks = out.get("masks", out.get("pred_masks", None))
                    iou_preds = out.get("iou_predictions", out.get("iou_preds", out.get("scores", None)))
                    low_res = out.get("low_res_masks", None)
                elif isinstance(out, (list, tuple)):
                    # typical tuple: (masks, iou_preds, low_res)
                    if len(out) >= 2:
                        masks, iou_preds = out[0], out[1]
                        low_res = out[2] if len(out) >= 3 else None
                    else:
                        continue
                else:
                    continue
                return masks, iou_preds, low_res
            except TypeError:
                # try calling with positional instead
                try:
                    out = fn(points, labels, multimask_output, return_logits, mask_input)
                    if isinstance(out, dict):
                        masks = out.get("masks", None)
                        iou_preds = out.get("iou_preds", None)
                        low_res = out.get("low_res_masks", None)
                    elif isinstance(out, (list, tuple)):
                        masks, iou_preds = out[0], out[1]
                        low_res = out[2] if len(out) >= 3 else None
                    else:
                        continue
                    return masks, iou_preds, low_res
                except Exception:
                    continue
            except Exception:
                # method exists but failed for other reason -> rethrow for debugging
                raise
        # if we reach here, no method worked
        raise RuntimeError(
            "Couldn't find a usable prediction method on Sam3Processor. "
            "Check processor API and adjust _processor_predict."
        )

    # --------------------------
    # High-level generate interface (matches SAM2 generate)
    # --------------------------
    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Public API: input HWC uint8 image -> list of mask records.
        """
        mask_data = self._generate_masks(image)

        # encode according to output_mode
        if self.output_mode == "coco_rle":
            segs = [coco_encode_rle(rle) for rle in mask_data.rles]
        elif self.output_mode == "binary_mask":
            segs = [rle_to_mask(rle) for rle in mask_data.rles]
        else:
            segs = mask_data.rles

        anns: List[Dict[str, Any]] = []
        for i in range(len(segs)):
            ann = {
                "segmentation": segs[i],
                "area": area_from_rle(mask_data.rles[i]),
                "bbox": (mask_data.boxes[i].tolist() if isinstance(mask_data.boxes, torch.Tensor) else mask_data.boxes[i]).copy(),
                "predicted_iou": float(mask_data.iou_preds[i]) if isinstance(mask_data.iou_preds, (torch.Tensor, np.ndarray)) else float(mask_data.iou_preds[i]),
                "point_coords": (mask_data.points[i].cpu().numpy().tolist() if torch.is_tensor(mask_data.points[i]) else mask_data.points[i]),
                "stability_score": float(mask_data.stability_score[i]) if isinstance(mask_data.stability_score[i], (torch.Tensor, np.floating, float)) else float(mask_data.stability_score[i]),
                "crop_box": mask_data.crop_boxes[i].copy(),
            }
            anns.append(ann)
        return anns

    # --------------------------
    # Core mask generation pipeline
    # --------------------------
    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]  # H, W
        crop_boxes, layer_idxs = generate_crop_boxes(orig_size, self.crop_n_layers, self.crop_overlap_ratio)
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # merge between crops with NMS
        if len(crop_boxes) > 1 and len(data) > 0:
            # prefer masks from smaller crops -> higher score
            boxes_tensor = data.boxes if isinstance(data.boxes, torch.Tensor) else torch.stack(data.boxes)
            scores = 1.0 / (box_area(boxes_tensor) + 1e-6)
            scores = scores.to(boxes_tensor.device)
            keep = batched_nms(boxes_tensor.float(), scores, torch.zeros_like(boxes_tensor[:, 0]), iou_threshold=self.crop_nms_thresh)
            data.filter(list(keep.cpu().numpy()))
        return data

    def _process_crop(self, image: np.ndarray, crop_box: Sequence[int], crop_layer_idx: int, orig_size: Tuple[int, int]) -> MaskData:
        x0, y0, x1, y1 = map(int, crop_box)
        cropped = image[y0:y1, x0:x1, :]
        ch, cw = cropped.shape[:2]
        # set image on processor (returns a state if SAM3 processor uses stateful API)
        # many implementations: state = processor.set_image(pil_or_np)
        state = None
        if hasattr(self.processor, "set_image"):
            # accept either numpy or PIL; Sam3Processor should handle conversion
            state = self.processor.set_image(cropped)
        else:
            # Some processors might require passing image in predict; we just keep state=None
            state = None

        # point grids (normalized -> absolute)
        points_scale = np.array([cw, ch])[None, :]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale  # Nx2 (x,y)
        # Batch iteration over points
        data = MaskData()
        pts_np = points_for_image.astype(np.float32)
        # chunk
        for i0 in range(0, pts_np.shape[0], self.points_per_batch):
            chunk = pts_np[i0 : i0 + self.points_per_batch]
            # convert to torch absolute coordinates
            pts_torch = torch.as_tensor(chunk, dtype=torch.float32, device=self.device)
            # The processor usually expects point coords in absolute pixel coordinates or normalized - try both:
            # We'll try to transform via processor if transform helper exists
            try:
                if hasattr(self.processor, "transform_coords"):
                    in_points = self.processor.transform_coords(pts_torch, normalize=False, orig_hw=(ch, cw))
                else:
                    in_points = pts_torch
            except Exception:
                in_points = pts_torch

            labels = torch.ones((in_points.shape[0],), dtype=torch.int, device=in_points.device)
            # call prediction wrapper; supports optional mask_input for m2m (not used here)
            masks_out, iou_preds_out, low_res_out = self._processor_predict(
                state=state,
                points=in_points,
                labels=labels,
                multimask_output=self.multimask_output,
                return_logits=True,
                mask_input=None,
            )

            # Normalize masks_out to expected shapes:
            # Expect masks_out: (B, M, Hc, Wc) or (B*M, Hc, Wc)
            if isinstance(masks_out, torch.Tensor):
                m = masks_out
                if m.ndim == 4:
                    # (B, M, Hc, Wc) -> flatten first two dims
                    b, mcount, Hc, Wc = m.shape
                    masks_flat = m.reshape(b * mcount, Hc, Wc)
                elif m.ndim == 3:
                    masks_flat = m
                else:
                    raise RuntimeError("Unexpected masks tensor shape from processor")
            else:
                # if numpy array
                mnp = np.asarray(masks_out)
                masks_flat = torch.as_tensor(mnp, device=self.device)

            # iou preds flatten
            if isinstance(iou_preds_out, torch.Tensor):
                iou_flat = iou_preds_out.reshape(-1)
            else:
                iou_flat = torch.as_tensor(np.asarray(iou_preds_out)).reshape(-1).to(self.device)

            # low_res masks stack if present
            if isinstance(low_res_out, torch.Tensor):
                lrm = low_res_out
                if lrm.ndim == 4:
                    lrm_flat = lrm.reshape(-1, lrm.shape[-2], lrm.shape[-1])
                else:
                    lrm_flat = lrm
            else:
                lrm_flat = None

            # thresholding logits -> binary masks using mask_threshold (note: return_logits=True yields logits)
            # If values look like probs (0..1), it's fine; user may need to adjust mask_threshold
            if masks_flat.dtype.is_floating_point:
                binary_masks = masks_flat > self.mask_threshold
            else:
                binary_masks = masks_flat.bool()

            # uncrop masks to original image coordinates
            uncropped_masks = uncrop_masks(binary_masks, crop_box, orig_size[0], orig_size[1])

            # compute boxes
            boxes = batched_mask_to_box(uncropped_masks)

            # fill MaskData entries
            for idx in range(uncropped_masks.shape[0]):
                data.rles.append(coco_encode_rle(uncropped_masks[idx].cpu().numpy()))
                data.boxes.append(boxes[idx].cpu())
                data.iou_preds.append(iou_flat[idx].cpu() if torch.is_tensor(iou_flat) else torch.tensor(float(iou_flat[idx])))
                # convert point to absolute on original image (uncrop)
                pt_abs = torch.tensor([chunk[idx % chunk.shape[0], 0] + x0, chunk[idx % chunk.shape[0], 1] + y0], device=self.device)
                data.points.append(pt_abs.cpu())
                data.stability_score.append(torch.tensor(1.0))  # placeholder: SAM3 may not expose stability; set 1.0
                data.crop_boxes.append([int(x0), int(y0), int(x1), int(y1)])
                if lrm_flat is not None:
                    data.low_res_masks.append(lrm_flat[idx].cpu())
                else:
                    data.low_res_masks.append(torch.zeros((1, 1), dtype=torch.uint8))

        # After processing batches within crop, apply intra-crop NMS to remove duplicates
        if len(data.boxes) > 0:
            # stack boxes and scores
            boxes_t = torch.stack(data.boxes).to(self.device)
            scores_t = torch.stack([torch.as_tensor(float(x)) for x in data.iou_preds]).to(self.device)
            keep = batched_nms(boxes_t.float(), scores_t, torch.zeros_like(boxes_t[:, 0], dtype=torch.int), iou_threshold=self.box_nms_thresh)
            data.filter(list(keep.cpu().numpy()))
        return data

    # --------------------------
    # Optional small region postprocessing (same contract as SAM2)
    # --------------------------
    @staticmethod
    def postprocess_small_regions(mask_data: MaskData, min_area: int, nms_thresh: float) -> MaskData:
        if len(mask_data.rles) == 0:
            return mask_data
        # Reconstruct dense masks, filter small components, recompute boxes, run NMS preferring unchanged masks
        new_masks = []
        scores = []
        for rle in mask_data.rles:
            mask = rle_to_mask(rle)
            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed2 = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed2
            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            scores.append(1.0 if unchanged else 0.0)
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep = batched_nms(boxes.float(), torch.as_tensor(scores), torch.zeros_like(boxes[:, 0]), iou_threshold=nms_thresh)
        # update rles for changed masks that survived
        for i in keep:
            i = int(i)
            if scores[i] == 0.0:
                mask_data.rles[i] = coco_encode_rle(masks[i].cpu().numpy())
                mask_data.boxes[i] = boxes[i]
        mask_data.filter(list(keep.cpu().numpy()))
        return mask_data

