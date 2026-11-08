"""Synthetic image pairs with known transformations. **Not lunar data, always labelled as such.**

A crater height field is rendered with a Lambertian surface and ray-marched cast shadows,
so that changing the Sun really moves shadows and reverses shading instead of merely
changing contrast. The source is then resampled through a known transformation, which
gives exact ground truth for every pixel, something the official sample does not provide.

These results show that the machinery recovers known geometry and whether refinement
truly reduces error. They do not predict accuracy on real scenes, whose terrain, sensors
and distortions are not modelled here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from pydantic import BaseModel

from app.geometry.estimation import apply_transform
from app.ingestion.raster import RasterData, RasterInfo, compute_valid_mask
from app.metadata.overlap import OverlapEstimate

SYNTHETIC_LABEL = "SYNTHETIC: rendered terrain with a known transformation, not lunar data"


@dataclass
class SyntheticCase:
    name: str
    group: str
    description: str
    source: RasterData
    reference: RasterData
    truth: np.ndarray  # 3x3, source pixel -> reference pixel
    prior: OverlapEstimate | None  # imitates metadata: right orientation, wrong position
    parameters: dict = field(default_factory=dict)


class SyntheticOutcome(BaseModel):
    label: str = SYNTHETIC_LABEL
    case: str
    group: str
    matcher: str
    status: str
    model: str | None = None
    inliers: int | None = None
    candidates: int | None = None
    truth_rmse_px: float | None = None
    truth_median_px: float | None = None
    truth_max_px: float | None = None
    tie_points: int | None = None
    tie_error_initial_median_px: float | None = None
    tie_error_refined_median_px: float | None = None
    tie_error_initial_rmse_px: float | None = None
    tie_error_refined_rmse_px: float | None = None
    refined_points: int | None = None
    seconds: float | None = None
    failure: str | None = None


def crater_terrain(size: tuple[int, int], seed: int = 0) -> np.ndarray:
    """Height field (float32, in pixels of height) with rolling ground and fresh craters."""
    rng = np.random.default_rng(seed)
    h, w = size
    height = np.zeros((h, w), np.float32)
    for sigma, amplitude in ((60, 45.0), (20, 14.0), (6, 3.5), (2, 0.8)):
        layer = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
        height += amplitude * layer / (layer.std() + 1e-6)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    count = int(h * w / 3500)
    radii = np.clip(rng.pareto(2.2, count) * 5.0 + 4.0, 4.0, 70.0)
    for radius in radii:
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        x0, x1 = int(max(0, cx - 2.2 * radius)), int(min(w, cx + 2.2 * radius + 1))
        y0, y1 = int(max(0, cy - 2.2 * radius)), int(min(h, cy + 2.2 * radius + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        d = np.hypot(xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy) / radius
        depth = 0.42 * radius
        # Parabolic bowl with a raised rim that decays with distance.
        profile = np.where(d < 1.0, depth * (d**2 - 1.0) + 0.1 * radius, 0.1 * radius * np.exp(-3.0 * (d - 1.0)))
        height[y0:y1, x0:x1] += profile.astype(np.float32)
    return height


def render(height: np.ndarray, sun_azimuth_deg: float, sun_elevation_deg: float) -> np.ndarray:
    """Lambertian shading with cast shadows, float32 in [0, 1]. Azimuth is measured in image axes."""
    az, el = np.radians(sun_azimuth_deg), np.radians(sun_elevation_deg)
    light = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)], dtype=np.float32)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    norm = np.sqrt(gx**2 + gy**2 + 1.0)
    shade = np.clip((-gx * light[0] - gy * light[1] + light[2]) / norm, 0.0, 1.0)

    # Cast shadows: march towards the Sun in a frame where it lies along +x.
    h, w = height.shape
    centre = ((w - 1) / 2.0, (h - 1) / 2.0)
    side = int(np.ceil(np.hypot(w, h))) + 4
    rotation = cv2.getRotationMatrix2D(centre, np.degrees(az), 1.0)
    rotation[:, 2] += np.array([(side - w) / 2.0, (side - h) / 2.0])
    turned = cv2.warpAffine(height, rotation, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    slope = float(np.tan(el))
    ramp = np.arange(side, dtype=np.float32)[None, :] * slope
    # A point at x is shadowed if some x' further towards the Sun rises above the ray through it:
    # h(x') > h(x) + (x' - x) * tan(el), that is, if h - x * tan(el) is larger anywhere beyond x.
    level = turned - ramp
    horizon = np.maximum.accumulate(level[:, ::-1], axis=1)[:, ::-1]
    shadowed = level < (horizon - 0.25)
    inverse = cv2.invertAffineTransform(rotation)
    lit = cv2.warpAffine((~shadowed).astype(np.float32), inverse, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return (shade * lit).astype(np.float32)


def _raster(data: np.ndarray, label: str, nodata: float | None) -> RasterData:
    h, w = data.shape
    info = RasterInfo(
        path=f"synthetic://{label}",
        width=w,
        height=h,
        band_count=1,
        dtype=str(data.dtype),
        nodata=nodata,
        crs=None,
        crs_is_geographic=None,
        transform=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        has_geotransform=False,
        pixel_size=None,
        bounds=None,
        gcps=[],
        gcp_crs=None,
        crs_label_consistent=None,
        tags={"SYNTHETIC": SYNTHETIC_LABEL},
    )
    return RasterData(info=info, data=data, valid_mask=compute_valid_mask(data, nodata))


def _to_uint8(image: np.ndarray) -> np.ndarray:
    # Offset by one grey level so that 0 stays reserved for "no data".
    return np.clip(np.rint(image * 254.0) + 1.0, 1, 255).astype(np.uint8)


def _similarity(angle_deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = scale * np.cos(a), scale * np.sin(a)
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def _prior(
    truth: np.ndarray, source_size: tuple[int, int], reference_size: tuple[int, int], rng: np.random.Generator, offset_px: float, search_px: float
) -> OverlapEstimate:
    """What metadata would offer: nearly the right frame, wrong by ``offset_px`` in position."""
    angle = np.radians(rng.uniform(-1.5, 1.5))
    scale = rng.uniform(0.98, 1.02)
    perturb = np.array([[scale * np.cos(angle), -scale * np.sin(angle), 0.0], [scale * np.sin(angle), scale * np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    w, h = source_size
    centre = np.array([[w / 2.0, h / 2.0]])
    affine = truth.copy() / truth[2, 2]
    affine[2] = [0.0, 0.0, 1.0]
    prior = perturb @ affine
    shift = apply_transform(truth, centre)[0] - apply_transform(prior, centre)[0]
    direction = rng.uniform(0.0, 2.0 * np.pi)
    shift += offset_px * np.array([np.cos(direction), np.sin(direction)])
    prior[:2, 2] += shift
    corners = apply_transform(prior, np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
    return OverlapEstimate(
        mode="metadata",
        reason="synthetic prior imitating metadata",
        prior_affine=prior[:2].tolist(),
        prior_mirrored=bool(np.linalg.det(prior[:2, :2]) < 0),
        predicted_corners_px=corners.tolist(),
        search_radius_px=search_px,
        search_bbox_px=[0, 0, reference_size[0], reference_size[1]],
    )


def build_cases(seed: int = 20260166, source_size: tuple[int, int] = (420, 1000), reference_size: tuple[int, int] = (1500, 1500)) -> list[SyntheticCase]:
    """One case per property the brief asks to test; each differs from the baseline in that property."""
    rng = np.random.default_rng(seed)
    rw, rh = reference_size
    w, h = source_size
    terrain = crater_terrain((rh, rw), seed)
    reference_image = render(terrain, 40.0, 25.0)
    cx, cy = (rw - 1) / 2.0, (rh - 1) / 2.0

    def placed(angle: float = 8.0, scale: float = 1.0, at: tuple[float, float] = (cx, cy)) -> np.ndarray:
        matrix = _similarity(angle, scale, 0.0, 0.0)
        centre = matrix[:2, :2] @ np.array([(w - 1) / 2.0, (h - 1) / 2.0])
        matrix[:2, 2] = np.array(at) - centre
        return matrix

    def make(
        name: str,
        group: str,
        description: str,
        truth: np.ndarray,
        sun: tuple[float, float] = (40.0, 25.0),
        noise: float = 0.0,
        blur: float = 0.0,
        with_prior: bool = True,
        **parameters,
    ) -> SyntheticCase:
        seen = reference_image if sun == (40.0, 25.0) else render(terrain, *sun)
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        m = truth / truth[2, 2]
        if np.allclose(m[2], [0.0, 0.0, 1.0]):
            warped = cv2.warpAffine(seen, m[:2], (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
        else:
            warped = cv2.warpPerspective(seen, m, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
        outside = warped < -0.5
        image = np.clip(warped, 0.0, 1.0)
        if blur > 0:
            image = cv2.GaussianBlur(image, (0, 0), blur)
        if noise > 0:
            image = image + rng.normal(0.0, noise, image.shape).astype(np.float32)
        data = _to_uint8(np.clip(image, 0.0, 1.0))
        data[outside] = 0
        prior = _prior(truth, (w, h), (rw, rh), rng, offset_px=220.0, search_px=420.0) if with_prior else None
        return SyntheticCase(
            name, group, description,
            _raster(data, f"{name}/source", 0),
            _raster(_to_uint8(reference_image), f"{name}/reference", 0),
            truth, prior, {"sun_azimuth_deg": sun[0], "sun_elevation_deg": sun[1], "noise_sigma": noise, "blur_sigma": blur, **parameters},
        )  # fmt: skip

    return [
        make("baseline", "baseline", "Same Sun, small rotation, no degradation", placed()),
        make("translation", "translation", "Far from the reference centre", placed(at=(cx - 280.0, cy + 240.0)), offset=(-280, 240)),
        make("rotation_25", "rotation", "25 degree rotation, known to the prior", placed(25.0), angle_deg=25),
        make("rotation_110", "rotation", "110 degree rotation, known to the prior", placed(110.0), angle_deg=110),
        make("scale_0.8", "scale", "Source is 1.25x finer than the reference", placed(scale=0.8), scale=0.8),
        make("scale_1.25", "scale", "Source is 0.8x as fine as the reference", placed(scale=1.25), scale=1.25),
        make(
            "perspective",
            "perspective",
            "Mild perspective",
            placed() @ np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [4e-5, 2e-5, 1.0]]),
            perspective=[4e-5, 2e-5],
        ),
        make("noise_0.05", "noise", "Gaussian noise, sigma 5% of full scale", placed(), noise=0.05),
        make("noise_0.12", "noise", "Gaussian noise, sigma 12% of full scale", placed(), noise=0.12),
        make("blur_1.5", "blur", "Gaussian blur, sigma 1.5 px", placed(), blur=1.5),
        make("blur_3", "blur", "Gaussian blur, sigma 3 px", placed(), blur=3.0),
        make("sun_azimuth_90", "illumination", "Sun azimuth moved by 90 degrees", placed(), sun=(130.0, 25.0)),
        make("sun_azimuth_180", "illumination", "Sun from the opposite side: shading reverses", placed(), sun=(220.0, 25.0)),
        make("sun_low_5", "illumination", "Sun lowered from 25 to 5 degrees: long shadows", placed(), sun=(40.0, 5.0)),
        make("sun_low_opposite", "illumination", "Opposite azimuth and 5 degree Sun, the hardest case here", placed(), sun=(220.0, 5.0)),
        make(
            "partial_overlap",
            "partial_overlap",
            "About 40% of the source lies outside the reference",
            placed(at=(cx, rh - 0.1 * h)),
            overlap="partial",
        ),
        make("no_metadata", "no_prior", "No prior at all: orientation and position must be found", placed(30.0), with_prior=False),
    ]


def truth_errors(apply_model, truth: np.ndarray, source_valid: np.ndarray, reference_size: tuple[int, int], step: int = 16) -> dict:
    """Error of a recovered mapping against the known one, over valid source pixels inside the reference."""
    h, w = source_valid.shape
    xx, yy = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    points = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
    points = points[source_valid[points[:, 1].astype(int), points[:, 0].astype(int)]]
    target = apply_transform(truth, points)
    inside = (target[:, 0] >= 0) & (target[:, 1] >= 0) & (target[:, 0] < reference_size[0]) & (target[:, 1] < reference_size[1])
    if not inside.any():
        return {"rmse": None, "median": None, "max": None}
    errors = np.linalg.norm(apply_model(points[inside]) - target[inside], axis=1)
    return {"rmse": float(np.sqrt(np.mean(errors**2))), "median": float(np.median(errors)), "max": float(errors.max())}
