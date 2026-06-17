import argparse, os, sys, math
import numpy as np
from typing import Tuple
from PIL import Image, ImageSequence, ImageDraw

# ---------- I/O ----------
def imread_gif(path: str):
    im = Image.open(path)
    frames_rgb = [frame.convert("RGB") for frame in ImageSequence.Iterator(im)]
    frames_L   = [frame.convert("L")   for frame in ImageSequence.Iterator(im)]
    durations = []
    for frame in ImageSequence.Iterator(Image.open(path)):
        durations.append(frame.info.get("duration", None))
    if all(d is None for d in durations):
        default = im.info.get("duration", 100)
        durations = [default] * len(frames_rgb)
    else:
        last_known = im.info.get("duration", 100)
        for i, d in enumerate(durations):
            if d is None:
                durations[i] = last_known
            else:
                last_known = d
    arrs = np.stack([np.array(f, dtype=np.float32) for f in frames_L], axis=0)
    return frames_rgb, arrs, durations

# ---------- preprocessing ----------
def clip_frames(arrs: np.ndarray, vmin: float=None, vmax: float=None) -> np.ndarray:
    out = arrs.copy()
    if vmin is not None:
        out = np.maximum(out, float(vmin))
    if vmax is not None:
        out = np.minimum(out, float(vmax))
    return out

def box_blur(img: np.ndarray, k: int=3) -> np.ndarray:
    if k <= 1 or k % 2 == 0:
        return img
    H, W = img.shape
    pad = k // 2
    integ = np.pad(img, ((1,0),(1,0)), mode='constant', constant_values=0).cumsum(0).cumsum(1)
    y0 = np.clip(np.arange(H) - pad, 0, H)
    y1 = np.clip(np.arange(H) + pad + 1, 0, H)
    x0 = np.clip(np.arange(W) - pad, 0, W)
    x1 = np.clip(np.arange(W) + pad + 1, 0, W)
    out = np.empty_like(img, dtype=np.float32)
    for i in range(H):
        top, bot = y0[i], y1[i]
        A = integ[top  :top + 1, x0]
        B = integ[top  :top + 1, x1]
        C = integ[bot  :bot + 1, x0]
        D = integ[bot  :bot + 1, x1]
        win_sum = (D - B - C + A).astype(np.float32)
        out[i, :] = win_sum / float((bot - top) * (pad*2+1))
    return out

def preprocess_frames(arrs: np.ndarray, clip_vmin: float=None, clip_vmax: float=None,
                      smooth: bool=True, k: int=3) -> np.ndarray:
    out = clip_frames(arrs, clip_vmin, clip_vmax)
    if smooth and k and k>1:
        out = np.stack([box_blur(f, k=k) for f in out], axis=0)
    return out

# ---------- mask ----------
def _dilate_bool(mask: np.ndarray, iters: int=1) -> np.ndarray:
    if iters <= 0:
        return mask
    out = mask.astype(bool)
    for _ in range(iters):
        m = out.astype(np.uint8)
        p = np.pad(m, ((1,1),(1,1)), mode='edge')
        neigh = (
            (p[:-2,:-2]) | (p[:-2,1:-1]) | (p[:-2,2:]) |
            (p[1:-1,:-2]) | (p[1:-1,1:-1]) | (p[1:-1,2:]) |
            (p[2:,:-2]) | (p[2:,1:-1]) | (p[2:,2:])
        )
        out = neigh.astype(bool)
    return out

def build_auto_mask(arrs: np.ndarray, edge: int=4, clip_vmin: float=None, clip_vmax: float=None,
                    sat_frac: float=0.10, mad_k: float=8.0, dilate_iters: int=1) -> np.ndarray:
    N, H, W = arrs.shape
    good = np.ones((H, W), dtype=bool)
    if edge > 0:
        good[:edge, :] = False; good[-edge:, :] = False
        good[:, :edge] = False; good[:, -edge:] = False
    if clip_vmax is not None:
        hits = (arrs >= (float(clip_vmax) - 1e-6)).mean(axis=0)
        good &= (hits < float(sat_frac))
    med = np.median(arrs, axis=0)
    mad = np.median(np.abs(arrs - med), axis=0)
    gmad = np.median(mad)
    if gmad <= 0: gmad = np.mean(mad) + 1e-6
    good &= (mad <= float(mad_k) * gmad)
    bad = _dilate_bool(~good, iters=int(dilate_iters))
    return ~bad

def load_mask_file(path: str, expect_shape: Tuple[int,int]) -> np.ndarray:
    if path is None or not os.path.isfile(path):
        return None
    H, W = expect_shape
    try:
        m = Image.open(path).convert("L")
        if (m.size[0], m.size[1]) != (W, H):
            return None
        arr = np.array(m, dtype=np.uint8)
        return arr > 0
    except Exception:
        return None

# ---------- shift/stack ----------
def shift_int_pad(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    H, W = img.shape
    iy, fy = int(np.floor(dy)), float(dy - np.floor(dy))
    ix, fx = int(np.floor(dx)), float(dx - np.floor(dx))
    base = np.zeros_like(img)
    if iy >= 0:
        y_dst_start, y_dst_end = iy, H
        y_src_start, y_src_end = 0, H - iy
    else:
        y_dst_start, y_dst_end = 0, H + iy
        y_src_start, y_src_end = -iy, H
    if ix >= 0:
        x_dst_start, x_dst_end = ix, W
        x_src_start, x_src_end = 0, W - ix
    else:
        x_dst_start, x_dst_end = 0, W + ix
        x_src_start, x_src_end = -ix, W
    if y_src_end > y_src_start and x_src_end > x_src_start:
        base[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = img[y_src_start:y_src_end, x_src_start:x_src_end]
    if abs(fx) < 1e-6 and abs(fy) < 1e-6:
        return base
    def shift_frac_x(a, fx):
        if abs(fx) < 1e-6: return a
        x1 = np.roll(a, -1, axis=1); x1[:, -1] = 0.0
        return (1.0 - fx) * a + fx * x1
    def shift_frac_y(a, fy):
        if abs(fy) < 1e-6: return a
        y1 = np.roll(a, -1, axis=0); y1[-1, :] = 0.0
        return (1.0 - fy) * a + fy * y1
    tmp = shift_frac_x(base, fx if fx>=0 else 1+fx) if fx<0 else shift_frac_x(base, fx)
    out = shift_frac_y(tmp, fy if fy>=0 else 1+fy) if fy<0 else shift_frac_y(tmp, fy)
    return out

def stack_for_velocity(R: np.ndarray, vy: float, vx: float, ref_index: int=None) -> np.ndarray:
    N, H, W = R.shape
    if ref_index is None:
        ref_index = N // 2
    acc = np.zeros((H, W), dtype=np.float32)
    for i in range(N):
        di = (i - ref_index)
        acc += shift_int_pad(R[i], -di * vy, -di * vx)
    return acc

# ---------- priors ----------
def circ_ang_diff_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)

def prior_weight(vx, vy, args):
    w = 1.0
    s = math.hypot(vx, vy)
    if args.prior_angle_deg is not None and args.prior_angle_sigma is not None and s > 1e-9:
        ang = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0
        dth = min(
        circ_ang_diff_deg(ang, args.prior_angle_deg),
        circ_ang_diff_deg((ang + 180.0) % 360.0, args.prior_angle_deg)
    )
        w *= math.exp(-0.5 * (dth / max(args.prior_angle_sigma, 1e-6))**2)
    if args.prior_speed is not None and args.prior_speed_sigma is not None:
        ds = (s - args.prior_speed)
        w *= math.exp(-0.5 * (ds / max(args.prior_speed_sigma, 1e-6))**2)
    if args.prior_vx is not None and args.prior_vy is not None and args.prior_sigma_pix is not None:
        dvx = vx - args.prior_vx; dvy = vy - args.prior_vy
        w *= math.exp(-0.5 * (dvx*dvx + dvy*dvy) / (max(args.prior_sigma_pix, 1e-6)**2))
    return float(w)

# ---------- search & refine ----------
def search_best_velocity(R: np.ndarray, args, step: float=1.0):
    best = None; best_map = None
    H, W = R.shape[1:]
    vx_min = -args.vmax if args.vx_min is None else max(-args.vmax, args.vx_min)
    vx_max =  args.vmax if args.vx_max is None else min( args.vmax, args.vx_max)
    vy_min = -args.vmax if args.vy_min is None else max(-args.vmax, args.vy_min)
    vy_max =  args.vmax if args.vy_max is None else min( args.vmax, args.vy_max)
    grid_vx = np.arange(-args.vmax, args.vmax + 1e-9, step, dtype=float)
    grid_vy = np.arange(-args.vmax, args.vmax + 1e-9, step, dtype=float)
    for vy in grid_vy:
        if vy < vy_min or vy > vy_max: continue
        for vx in grid_vx:
            if vx < vx_min or vx > vx_max: continue
            s = math.hypot(vx, vy)
            vmin_eff = max(0.0, min(args.vmin, args.vmax))
            if s < vmin_eff or s > args.vmax: continue
            acc = stack_for_velocity(R, vy, vx)
            idx = int(np.argmax(acc)); pk = float(acc.ravel()[idx]); y, x = divmod(idx, W)
            w = prior_weight(vx, vy, args)
            score = pk * w
            if (best is None) or (score > best[0]):
                best = (score, pk, w, vx, vy, (x, y)); best_map = acc
    if best is None:
        acc = stack_for_velocity(R, 0.0, 0.0)
        idx = int(np.argmax(acc)); pk = float(acc.ravel()[idx]); y, x = divmod(idx, R.shape[2])
        w = prior_weight(0.0, 0.0, args)
        best = (pk*w, pk, w, 0.0, 0.0, (x, y)); best_map = acc
    score, pk, w, vx, vy, (x0, y0) = best
    return vx, vy, (x0, y0), pk, w, best_map

def refine_velocity_local(R: np.ndarray, vx0: float, vy0: float, halfwin: float=2.0, step: float=0.25, args=None):
    best = None; best_map = None
    H, W = R.shape[1:]
    vx_list = np.arange(vx0 - halfwin, vx0 + halfwin + 1e-9, step)
    vy_list = np.arange(vy0 - halfwin, vy0 + halfwin + 1e-9, step)
    for vy in vy_list:
        for vx in vx_list:
            acc = stack_for_velocity(R, vy, vx)
            idx = int(np.argmax(acc)); pk = float(acc.ravel()[idx]); y, x = divmod(idx, W)
            w = prior_weight(vx, vy, args) if args is not None else 1.0
            score = pk * w
            if (best is None) or (score > best[0]):
                best = (score, pk, w, vx, vy, (x, y)); best_map = acc
    score, pk, w, vx, vy, (x0, y0) = best
    return vx, vy, (x0, y0), pk, w, best_map

# ---------- SNR ----------
def robust_snr(acc: np.ndarray, x0: int, y0: int, exclude_r: int=15):
    H, W = acc.shape
    Y, X = np.ogrid[:H, :W]
    mask = (X - x0)**2 + (Y - y0)**2 > exclude_r**2
    vals = acc[mask]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    sigma = 1.4826 * mad if mad>0 else (np.std(vals) + 1e-9)
    snr = (acc[y0, x0] - med) / (sigma + 1e-9)
    return snr, med, sigma

def _pill_mask(shape, cx, cy, L, w, theta_rad):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    x =  (xx - cx)*c + (yy - cy)*s
    y = -(xx - cx)*s + (yy - cy)*c
    halfL = 0.5 * float(L)
    halfW = 0.5 * float(w)
    rect = (np.abs(x) <= halfL) & (np.abs(y) <= halfW)
    left_cap  = (x < -halfL) & ((x + halfL)**2 + y**2 <= halfW**2)
    right_cap = (x >  halfL) & ((x - halfL)**2 + y**2 <= halfW**2)
    return rect | left_cap | right_cap

def _ring_mask(shape, cx, cy, rin, rout):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(xx - cx, yy - cy)
    return (rr >= rin) & (rr <= rout)

def pill_photometry_SNR(im, M, cx, cy, L, w, theta_rad, rin, rout):
    pill = _pill_mask(im.shape, cx, cy, L, w, theta_rad) & M
    ring = _ring_mask(im.shape, cx, cy, rin, rout) & (~pill) & M
    if pill.sum() < 30 or ring.sum() < 50:
        return float("nan")
    bg_vals = im[ring]
    bg_med = np.median(bg_vals)
    mad = np.median(np.abs(bg_vals - bg_med))
    sigma = 1.4826 * mad if mad>0 else (np.std(bg_vals) + 1e-9)
    F = float(np.sum(im[pill] - bg_med))
    N = int(pill.sum())
    var = max(F, 0.0) + N*(sigma**2)
    return F / math.sqrt(var) if var>0 else float("nan")

# ---------- XY snap & velocity refit helpers (do NOT affect detection logic) ----------
def snap_xy_local(im, M, x_pred, y_pred, L, w, theta, rin, rout,
                  r=3.0, coarse_step=0.25, refine=True):
    """Search a small window around (x_pred, y_pred) to maximize pill SNR.
    Returns (x_best, y_best, snr_best)."""
    xs = np.arange(x_pred - r, x_pred + r + 1e-9, float(coarse_step), dtype=float)
    ys = np.arange(y_pred - r, y_pred + r + 1e-9, float(coarse_step), dtype=float)
    best_s = -np.inf; bx, by = float(x_pred), float(y_pred)
    for yy in ys:
        for xx in xs:
            s = pill_photometry_SNR(im, M, float(xx), float(yy), float(L), float(w), float(theta),
                                    float(rin), float(rout))
            if np.isfinite(s) and s > best_s:
                best_s = float(s); bx, by = float(xx), float(yy)
    if (not refine) or (not np.isfinite(best_s)):
        return bx, by, float(best_s)
    # refine around the coarse peak
    fine_r = max(float(coarse_step), float(r)*0.6)
    fine_step = max(0.05, float(coarse_step)/4.0)
    xs2 = np.arange(bx - fine_r, bx + fine_r + 1e-9, fine_step, dtype=float)
    ys2 = np.arange(by - fine_r, by + fine_r + 1e-9, fine_step, dtype=float)
    best2_s = best_s; bx2, by2 = bx, by
    for yy in ys2:
        for xx in xs2:
            s = pill_photometry_SNR(im, M, float(xx), float(yy), float(L), float(w), float(theta),
                                    float(rin), float(rout))
            if np.isfinite(s) and s > best2_s:
                best2_s = float(s); bx2, by2 = float(xx), float(yy)
    return float(bx2), float(by2), float(best2_s)


def refit_velocity_from_positions(positions, snr_list, i0=0, snr_min=0.5):
    """Weighted linear fit: x(i), y(i) = a + b*(i-i0), weights ~ SNR^2.
    Returns dict(a_x, b_x, a_y, b_y, vx, vy, used)."""
    N = len(positions)
    t = np.arange(N, dtype=float) - float(i0)
    xs = np.array([p[0] for p in positions], dtype=float)
    ys = np.array([p[1] for p in positions], dtype=float)
    w = np.array([float(s) if np.isfinite(s) else 0.0 for s in snr_list], dtype=float)
    w = np.maximum(w, float(snr_min))**2
    good = np.isfinite(xs) & np.isfinite(ys) & (w > 0)
    idx = np.where(good)[0]
    if idx.size < 2:
        return dict(a_x=np.nan, b_x=np.nan, a_y=np.nan, b_y=np.nan, vx=np.nan, vy=np.nan, used=0)
    X = np.vstack([np.ones(idx.size), t[idx]]).T
    W = np.diag(w[idx])
    XtW = X.T @ W
    a_x, b_x = np.linalg.lstsq(XtW @ X, XtW @ xs[idx], rcond=None)[0]
    a_y, b_y = np.linalg.lstsq(XtW @ X, XtW @ ys[idx], rcond=None)[0]
    return dict(a_x=float(a_x), b_x=float(b_x), a_y=float(a_y), b_y=float(b_y),
                vx=float(b_x), vy=float(b_y), used=int(idx.size))

# ---------- empirical test ----------
def empirical_pill_test(im, M, target_snr, cx_t, cy_t, L, w, theta_rad,
                        rin, rout, n=100, nsigma=2.0, ptail=None,
                        exclude_radius=20.0, min_coverage=0.98, max_tries=200000, seed=42):
    rng = np.random.default_rng(seed)
    H, W = im.shape
    yy, xx = np.mgrid[0:H, 0:W]
    rr_ex = None
    if exclude_radius and exclude_radius > 0:
        rr_ex = (np.hypot(xx - cx_t, yy - cy_t) <= float(exclude_radius))
    halfL = 0.5 * float(L); halfW = 0.5 * float(w)
    x_min = int(np.ceil(0 + halfL + 2)); x_max = int(np.floor(W - halfL - 3))
    y_min = int(np.ceil(0 + halfW + 2)); y_max = int(np.floor(H - halfW - 3))
    if x_min >= x_max or y_min >= y_max:
        return False, np.nan, np.nan, np.nan, np.nan, np.array([])
    samples = []; tries = 0
    while len(samples) < n and tries < max_tries:
        tries += 1
        cx = rng.integers(x_min, x_max+1)
        cy = rng.integers(y_min, y_max+1)
        if rr_ex is not None and rr_ex[int(cy), int(cx)]:  # exclude center
            continue
        pill = _pill_mask(im.shape, cx, cy, L, w, theta_rad)
        good = pill & M
        tot = int(pill.sum())
        if tot < 30:
            continue
        cov = good.sum() / max(1, tot)
        if cov < min_coverage:
            continue
        ring = _ring_mask(im.shape, cx, cy, rin, rout) & (~pill) & M
        if ring.sum() < 50:
            continue
        bg_vals = im[ring]
        bg_med = np.median(bg_vals)
        mad = np.median(np.abs(bg_vals - bg_med))
        sigma = 1.4826 * mad if mad>0 else (np.std(bg_vals) + 1e-9)
        F = float(np.sum(im[good] - bg_med))
        Np = int(good.sum())
        var = max(F, 0.0) + Np*(sigma**2)
        snr_i = F / math.sqrt(var) if var>0 else np.nan
        if np.isfinite(snr_i):
            samples.append(float(snr_i))
    samples = np.array(samples, dtype=float)
    if samples.size < max(50, 0.2*n):
        return False, np.nan, np.nan, np.nan, np.nan, samples
    mu = np.median(samples)
    mad_s = np.median(np.abs(samples - mu))
    std = 1.4826 * mad_s if mad_s>0 else (np.std(samples) + 1e-9)
    if ptail is not None and 0 < ptail < 1:
        thresh = float(np.quantile(samples, 1.0 - ptail))
        accept = (target_snr >= thresh)
        z = (target_snr - mu) / max(std, 1e-9)
        return accept, z, mu, std, thresh, samples
    thresh = mu + nsigma * std
    z = (target_snr - mu) / max(std, 1e-9)
    return (z >= nsigma), z, mu, std, thresh, samples

# ---------- pill fit ----------
def build_pill_mask(H: int, W: int, x0: float, y0: float, L: float, w: float, theta: float):
    Y, X = np.mgrid[0:H, 0:W]
    xr = X - x0; yr = Y - y0
    c, s = math.cos(theta), math.sin(theta)
    xp =  xr * c + yr * s
    yp = -xr * s + yr * c
    halfL = max(L, 0.0) * 0.5
    halfW = max(w, 0.0) * 0.5
    rect = (np.abs(xp) <= halfL) & (np.abs(yp) <= halfW)
    left_cap  = ((xp + halfL)**2 + (yp)**2) <= (halfW**2)
    right_cap = ((xp - halfL)**2 + (yp)**2) <= (halfW**2)
    return rect | left_cap | right_cap

def circular_masks(H: int, W: int, x0: float, y0: float, r_ap: float, r_in: float, r_out: float):
    Y, X = np.mgrid[0:H, 0:W]
    rr2 = (X - x0)**2 + (Y - y0)**2
    ap = rr2 <= r_ap**2
    bg = (rr2 >= r_in**2) & (rr2 <= r_out**2)
    return ap, bg

def pill_snr(stack: np.ndarray, mask_pill: np.ndarray, x0: float, y0: float, fwhm_px: float):
    H, W = stack.shape
    r_circ = 1.5 * fwhm_px
    r_in, r_out = 2.5 * fwhm_px, 3.5 * fwhm_px
    ap_circ, bg_ring = circular_masks(H, W, x0, y0, r_circ, r_in, r_out)
    bg_vals = stack[bg_ring]
    bg_med = float(np.median(bg_vals)) if bg_vals.size>0 else 0.0
    bg_mad = float(np.median(np.abs(bg_vals - bg_med))) if bg_vals.size>0 else 0.0
    bg_sigma = 1.4826 * bg_mad if bg_mad>0 else (float(np.std(bg_vals)) if bg_vals.size>0 else 1.0)
    F = float(np.sum(stack[mask_pill] - bg_med))
    N = int(np.sum(mask_pill))
    var = max(F, 0.0) + N * (bg_sigma**2)
    snr = F / math.sqrt(var + 1e-9) if var>0 else 0.0
    return snr, F, N, bg_sigma

def fit_pill_params(stack: np.ndarray, x0: float, y0: float, L_guess: float, fwhm_px: float):
    H, W = stack.shape
    best = None
    alphas = np.linspace(0.6, 1.6, 11)
    wfs    = np.linspace(1.0, 2.5, 16)
    for a in alphas:
        L = float(max(1.0, a * L_guess))
        for wf in wfs:
            w = float(max(1.0, wf * fwhm_px))
            mask = build_pill_mask(H, W, x0, y0, L=L, w=w, theta=0.0)
            snr, F, N, bg_sigma = pill_snr(stack, mask, x0, y0, fwhm_px=fwhm_px)
            if (best is None) or (snr > best[0]):
                best = (snr, a, wf, L, w, F, N, bg_sigma)
    snr, a, wf, L, w, F, N, bg_sigma = best
    return {"alpha": a, "width_factor": wf, "L_px": L, "w_px": w, "snr": snr, "flux": F, "npix": N, "bg_sigma": bg_sigma}

# ---------- drawing ----------
def draw_capsule_on_frame(draw: ImageDraw.ImageDraw, x0: float, y0: float, L: float, w: float, theta: float, color=(255,0,0), width=2):
    halfL = L * 0.5; halfW = w * 0.5
    c, s = math.cos(theta), math.sin(theta)
    ux, uy = c, s
    vx, vy = -s, c
    p1 = (x0 + ux*halfL + vx*halfW, y0 + uy*halfL + vy*halfW)
    p2 = (x0 + ux*halfL - vx*halfW, y0 + uy*halfL - vy*halfW)
    p3 = (x0 - ux*halfL - vx*halfW, y0 - uy*halfL - vy*halfW)
    p4 = (x0 - ux*halfL + vx*halfW, y0 - uy*halfL + vy*halfW)
    try:
        draw.polygon([p1, p2, p3, p4], outline=color, width=width)
    except TypeError:
        draw.polygon([p1, p2, p3, p4], outline=color)
    r = halfW
    e1 = (x0 + ux*halfL, y0 + uy*halfL)
    e2 = (x0 - ux*halfL, y0 - uy*halfL)
    draw.ellipse([e1[0]-r, e1[1]-r, e1[0]+r, e1[1]+r], outline=color, width=width)
    draw.ellipse([e2[0]-r, e2[1]-r, e2[0]+r, e2[1]+r], outline=color, width=width)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Mover detector with pill SNR gating (snr_peak uses pill SNR).")
    ap.add_argument("gif")
    ap.add_argument("--vmax", type=float, default=8.0)
    ap.add_argument("--vmin", type=float, default=0.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--min-snr", type=float, default=3.0)
    ap.add_argument("--min-speed", type=float, default=0.3)
    ap.add_argument("--fwhm-px", type=float, default=3.0)
    ap.add_argument("--texp", type=float, default=None)
    ap.add_argument("--subframes", type=int, default=1)
    ap.add_argument("--trail-filter", action="store_true")
    # preproc & mask
    ap.add_argument("--no-preproc", action="store_true")
    ap.add_argument("--clip-vmin", type=float, default=None)
    ap.add_argument("--clip-vmax", type=float, default=None)
    ap.add_argument("--smooth-k", type=int, default=3)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--mask-edge", type=int, default=4)
    ap.add_argument("--mask-sat-frac", type=float, default=0.10)
    ap.add_argument("--mask-mad-k", type=float, default=8.0)
    ap.add_argument("--mask-dilate", type=int, default=1)
    ap.add_argument("--mask-file", type=str, default=None)
    # priors & bounds
    ap.add_argument("--prior-angle-deg", type=float, default=None)
    ap.add_argument("--prior-angle-sigma", type=float, default=None)
    ap.add_argument("--prior-speed", type=float, default=None)
    ap.add_argument("--prior-speed-sigma", type=float, default=None)
    ap.add_argument("--prior-vx", type=float, default=None)
    ap.add_argument("--prior-vy", type=float, default=None)
    ap.add_argument("--prior-sigma-pix", type=float, default=None)
    ap.add_argument("--vx-min", type=float, default=None); ap.add_argument("--vx-max", type=float, default=None)
    ap.add_argument("--vy-min", type=float, default=None); ap.add_argument("--vy-max", type=float, default=None)
    # outputs
    ap.add_argument("--linewidth", type=int, default=2)
    ap.add_argument("--outdir", type=str, default=None)

    # XY snap & refined stacking tools (DO NOT affect detection logic)
    ap.add_argument("--snap", dest="snap", action="store_true", default=True,
                    help="Per-frame local XY snap using pill-SNR (for outputs/forced photometry). Default: ON.")
    ap.add_argument("--no-snap", dest="snap", action="store_false",
                    help="Disable XY snap.")
    ap.add_argument("--snap-radius", type=float, default=None,
                    help="Snap search radius in pixels (default auto: max(2.5, 0.3*L_fit, 0.7*w_fit)).")
    ap.add_argument("--snap-step", type=float, default=0.25,
                    help="Snap coarse step in pixels (default 0.25).")
    ap.add_argument("--make-refined-stack", action="store_true",default=True,
                    help="Build an extra refined stack using velocity fit from snapped positions (detection unchanged).")
    # empirical
    ap.add_argument("--empirical-p", action="store_true",default = True)
    ap.add_argument("--emp-n", type=int, default=1000)
    ap.add_argument("--emp-nsigma", type=float, default=3.0)
    ap.add_argument("--emp-ptail", type=float, default=None)
    ap.add_argument("--emp-exclude-radius", type=float, default=20.0)
    ap.add_argument("--emp-min-coverage", type=float, default=0.98)
    ap.add_argument("--emp-max-tries", type=int, default=200000)
    ap.add_argument("--emp-seed", type=int, default=42)
    args = ap.parse_args()

    in_path = args.gif
    outdir = args.outdir or os.path.dirname(os.path.abspath(in_path)) or "."
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(in_path))[0]
    if not os.path.isfile(in_path):
        print("status,ERROR"); print(f"message,File not found: {in_path}", file=sys.stderr); return 1

    # read
    frames_rgb, arrs_raw, durations = imread_gif(in_path)
    N, H, W = arrs_raw.shape
    dt_frame = float(np.median(np.array(durations, dtype=float)) / 1000.0)
    texp = float(args.texp) if args.texp is not None else dt_frame

    # preproc
    use_preproc = (not args.no_preproc)
    arrs_pre = preprocess_frames(arrs_raw, args.clip_vmin, args.clip_vmax, smooth=(args.smooth_k and args.smooth_k>1), k=max(1,args.smooth_k)) if use_preproc else arrs_raw

    # background removal
    median = np.median(arrs_pre, axis=0)
    R = arrs_pre - median

    # mask
    use_mask = (not args.no_mask)
    if use_mask:
        M = build_auto_mask(arrs_pre, edge=args.mask_edge, clip_vmin=args.clip_vmin, clip_vmax=args.clip_vmax,
                            sat_frac=args.mask_sat_frac, mad_k=args.mask_mad_k, dilate_iters=args.mask_dilate)
        if args.mask_file:
            ext = load_mask_file(args.mask_file, (H, W))
            if ext is not None:
                M &= ext
        R = np.stack([R[i] * M for i in range(N)], axis=0)
        # try:
        #     Image.fromarray((M.astype(np.uint8)*255), mode="L").save(os.path.join(outdir, f"{stem}_mask.png"))
        # except Exception:
        #     pass
    else:
        M = np.ones((H, W), dtype=bool)

    # search
    vx0, vy0, (x0, y0), pk0, w0, acc0 = search_best_velocity(R, args, step=args.step)
    # optional de-trail & refine
    R_det = R.copy()
    if args.trail_filter and args.subframes > 1:
        R_tmp = np.zeros_like(R_det)
        for i in range(N):
            # simple de-trailing via intra-frame shifts
            K = int(args.subframes)
            out = np.zeros_like(R_det[i])
            taus = np.linspace(-0.5, 0.5, K)
            scale = (texp / dt_frame)
            for tau in taus:
                dy = -tau * scale * vy0
                dx = -tau * scale * vx0
                out += shift_int_pad(R_det[i], dy, dx)
            R_tmp[i] = out / float(K)
        vx1, vy1, (x1, y1), pk1, w1, acc1 = search_best_velocity(R_tmp, args, step=args.step)
        R_det = R_tmp; vx_init, vy_init, x_init, y_init = vx1, vy1, x1, y1
    else:
        vx_init, vy_init, x_init, y_init = vx0, vy0, x0, y0

    vx, vy, (x_star, y_star), pk, w_prior, acc = refine_velocity_local(R_det, vx_init, vy_init, halfwin=2.0, step=0.25, args=args)
    aligned = stack_for_velocity(R_det, vy, vx)

    # pixel-peak snr (diagnostic only)
    snr_pixel_peak, _, _ = robust_snr(aligned, int(x_star), int(y_star))

    # geometry
    speed = math.hypot(vx, vy)
    L_guess = max(1.0, speed * (texp / max(dt_frame, 1e-9)))
    w_guess = max(1.0, args.fwhm_px)
    theta = math.atan2(vy, vx)
    rin, rout = 1.5*args.fwhm_px, 3.5*args.fwhm_px

    # pill SNRs
    snr_pill_guess = pill_photometry_SNR(aligned, M, int(x_star), int(y_star), L_guess, w_guess, theta, rin, rout)
    fit = fit_pill_params(aligned, x_star, y_star, L_guess=L_guess, fwhm_px=args.fwhm_px)
    L_fit, w_fit = float(fit["L_px"]), float(fit["w_px"])
    snr_pill_fit = pill_photometry_SNR(aligned, M, int(x_star), int(y_star), L_fit, w_fit, theta, rin, rout)

    # empirical tests (DO NOT set status here; we decide at the end)
    ok_emp1 = ok_emp2 = True
    if args.empirical_p:
        ok_emp1, _, _, _, _, _ = empirical_pill_test(
            aligned, M.astype(bool), snr_pill_guess, int(x_star), int(y_star),
            L_guess, w_guess, theta, rin, rout,
            n=int(args.emp_n), nsigma=float(args.emp_nsigma), ptail=args.emp_ptail,
            exclude_radius=float(args.emp_exclude_radius), min_coverage=float(args.emp_min_coverage),
            max_tries=int(args.emp_max_tries), seed=int(args.emp_seed)
        )
        ok_emp2, _, _, _, _, _ = empirical_pill_test(
            aligned, M.astype(bool), snr_pill_fit, int(x_star), int(y_star),
            L_fit, w_fit, theta, rin, rout,
            n=int(args.emp_n), nsigma=float(args.emp_nsigma), ptail=args.emp_ptail,
            exclude_radius=float(args.emp_exclude_radius), min_coverage=float(args.emp_min_coverage),
            max_tries=int(args.emp_max_tries), seed=int(args.emp_seed)
        )

    # FINAL decision at the end (snr_peak uses pill SNR)
    snr_gate = snr_pill_fit if np.isfinite(snr_pill_fit) else snr_pill_guess
    detected_basic = (snr_gate >= args.min_snr) and (speed >= args.min_speed)
    if args.empirical_p:
        status = "DETECTED" if (detected_basic and ok_emp2) else "NO_DETECTION"
    else:
        status = "DETECTED" if detected_basic else "NO_DETECTION"

    # outputs (stackmap always; annotated GIF only if DETECTED)
    stack_png = os.path.join(outdir, f"{stem}_stackmap.png")
    try:
        acc_norm = aligned - np.min(aligned)
        acc_norm = (255.0 * acc_norm / (np.max(acc_norm) + 1e-9)).astype(np.uint8)
        Image.fromarray(acc_norm).save(stack_png)
    except Exception:
        stack_png = "NA"
    annotated_gif = "NA"; track_csv = "NA"
    if status == "DETECTED":
        i0 = N // 2
        # Predicted per-frame coordinates from (vx, vy) — original behavior (kept for detection outputs)
        positions = [(float(x_star + (i - i0) * vx), float(y_star + (i - i0) * vy)) for i in range(N)]
        # Optional per-frame XY snap (for annotation/forced photometry only; detection unchanged)
        positions_for_annot = positions
        track_snap_csv = "NA"
        if args.snap:
            base_r = max(2.5, 0.3*float(L_fit), 0.7*float(w_fit))
            snap_r = float(args.snap_radius) if (args.snap_radius is not None and args.snap_radius > 0) else base_r
            positions_snap = []
            snr_snap_list = []
            for i, (xp, yp) in enumerate(positions):
                xs, ys, ss = snap_xy_local(
                    R_det[i], M, float(xp), float(yp),
                    L=float(L_fit), w=float(w_fit), theta=float(theta),
                    rin=float(rin), rout=float(rout),
                    r=snap_r, coarse_step=float(args.snap_step), refine=True
                )
                if not np.isfinite(ss):
                    xs, ys = xp, yp
                positions_snap.append((float(xs), float(ys)))
                snr_snap_list.append(float(ss) if np.isfinite(ss) else float("nan"))
            positions_for_annot = positions_snap
            # Extra CSV for snapped coordinates (keep original track.csv unchanged)
            track_snap_csv = os.path.join(outdir, f"{stem}_track_snap.csv")
            try:
                with open(track_snap_csv, "w") as f:
                    f.write("frame,x_pred,y_pred,x_snap,y_snap,snr_snap\n")
                    for i, ((xp, yp), (xs, ys), ss) in enumerate(zip(positions, positions_snap, snr_snap_list)):
                        f.write(f"{i},{xp:.3f},{yp:.3f},{xs:.3f},{ys:.3f},{(ss if np.isfinite(ss) else float('nan')):.3f}\n")
            except Exception:
                track_snap_csv = "NA"
            # Optional: refined stack using velocity fit from snapped coords (for visualization only)
            if args.make_refined_stack:
                rf = refit_velocity_from_positions(positions_snap, snr_snap_list, i0=i0, snr_min=0.5)
                vx_ref, vy_ref = float(rf["vx"]), float(rf["vy"])
                aligned_ref = stack_for_velocity(R_det, vy_ref, vx_ref)  # same function as original stacking
                stack_ref_png = os.path.join(outdir, f"{stem}_stackmap_refined.png")
                try:
                    acc_ref = aligned_ref - np.min(aligned_ref)
                    acc_ref = (255.0 * acc_ref / (np.max(acc_ref) + 1e-9)).astype(np.uint8)
                    Image.fromarray(acc_ref).save(stack_ref_png)
                except Exception:
                    pass
        annot = []
        for i, (frm, (x, y)) in enumerate(zip(frames_rgb, positions_for_annot)):
            frm = frm.copy()
            dr = ImageDraw.Draw(frm)
            draw_capsule_on_frame(dr, x, y, L=L_fit, w=w_fit, theta=theta, color=(255,0,0), width=args.linewidth)
            try: dr.text((x + w_fit*0.6, y + w_fit*0.6), str(i), fill=(255,0,0))
            except Exception: pass
            annot.append(frm)
        annotated_gif = os.path.join(outdir, f"{stem}_annotated.gif")
        dur = durations if len(durations) == len(annot) else [durations[0]] * len(annot)
        try:
            annot[0].save(annotated_gif, save_all=True, append_images=annot[1:], loop=0, duration=dur, disposal=2)
        except Exception:
            annotated_gif = "NA"
        track_csv = os.path.join(outdir, f"{stem}_track.csv")
        try:
            with open(track_csv, "w") as f:
                f.write("frame,x_pix,y_pix\n")
                for i, (x, y) in enumerate(positions):
                    f.write(f"{i},{x:.3f},{y:.3f}\n")
        except Exception:
            track_csv = "NA"

    # print summary (snr_peak == pill SNR used for gating)
    ang_deg = (math.degrees(theta) + 360.0) % 360.0
    print(f"Detection,{status == 'DETECTED'}")
    # print(f"n_frames,{N}")
    # print(f"ref_index_center,{N//2}")
    # print(f"image_size,{W}x{H}")
    # print(f"dt_frame_s,{dt_frame:.6g}")
    # print(f"texp_s,{texp:.6g}")
    # print(f"vx_pix_per_frame,{vx:.6g}")
    # print(f"vy_pix_per_frame,{vy:.6g}")
    print(f"speed_pix_per_frame,{speed:.6g}")
    print(f"angle_deg,{ang_deg:.3f}")
    print(f"snr_peak,{snr_gate:.3f}")           # <= pill SNR
    # print(f"snr_pill_guess,{snr_pill_guess:.3f}")
    # print(f"snr_pill_fit,{snr_pill_fit:.3f}")
    # print(f"snr_pixel_peak,{snr_pixel_peak:.3f}")  # diagnostic
    # print(f"stack_map_png,{stack_png}")
    # print(f"L_guess_px,{L_guess:.3f}")
    print(f"L_fit_px,{L_fit:.3f}")
    print(f"w_fit_px,{w_fit:.3f}")
    # print(f"alpha_length_scale,{fit['alpha']:.3f}")
    # print(f"width_factor,{fit['width_factor']:.3f}")
    print(f"annotated_gif,{annotated_gif}")
    print(f"track_csv,{track_csv}")
    if args.empirical_p:
    #     print(f"empirical_enabled,True")
    #     print(f"emp_n,{int(args.emp_n)}")
    #     print(f"emp_nsigma,{float(args.emp_nsigma)}")
        print(f"emp_pass1,{ok_emp1}")
        print(f"emp_pass2,{ok_emp2}")
    # else:
    #     print(f"empirical_enabled,False")

    # # mask/preproc info
    # print(f"preprocess_enabled,{use_preproc}")
    # print(f"clip_vmin,{args.clip_vmin if args.clip_vmin is not None else 'NA'}")
    # print(f"clip_vmax,{args.clip_vmax if args.clip_vmax is not None else 'NA'}")
    # print(f"mask_enabled,{use_mask}")
    # print(f"mask_edge_px,{args.mask_edge}")
    # print(f"mask_sat_frac,{args.mask_sat_frac}")
    # print(f"mask_mad_k,{args.mask_mad_k}")
    # print(f"mask_dilate,{args.mask_dilate}")
    # print(f"mask_png,{os.path.join(outdir, f'{stem}_mask.png') if use_mask else 'NA'}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
