#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily thumbnails workflow with object+site grouping and time-window splitting.
Generates one GIF per subgroup with robust I/O handling.
"""
import os
import os.path as osp
import sys
import argparse
import subprocess
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np


def compute_size_from_unc(unc_series, fallback_px, multiplier=3.0, min_px=150, max_px=400, 
                          pixscale_arcsec_per_px=1.26, angle_px_deg_mean=None, trail_px_est_mean=None):
    """
    Compute cutout size (in pixels) for a subgroup from its `unc` values.

    Rules:
      size_px = clamp(ceil(unc * multiplier), min_px, max_px)
    - If `pixscale_arcsec_per_px` is provided (>0), `unc` is treated as arcseconds
      and converted to pixels: size_px = ceil((unc_arcsec * multiplier) / pixscale).
    - If not provided, `unc` is treated as already in pixels.
    - If the series is empty/invalid, fall back to `fallback_px`.
    - Additionally, if `angle_px_deg_mean` and `trail_px_est_mean` are provided,
      the size is also estimated from trail projection and the maximum of both methods is used.
      The trail-based size uses the maximum projection width (x or y) multiplied by 4 (for 4-frame GIF).

    Parameters
    ----------
    unc_series : pandas.Series
        The `unc` column for the subgroup.
    fallback_px : int
        Default size to fall back to when `unc` is missing or invalid.
    multiplier : float
        Multiplier applied to `unc` (default 3.0).
    min_px : int
        Lower bound for size (default 200).
    max_px : int
        Upper bound for size (default 1000).
    pixscale_arcsec_per_px : float or None
        Pixel scale (arcsec/px). If provided, `unc` is assumed arcsec.
    angle_px_deg_mean : float or None
        Mean trail angle in degrees.
    trail_px_est_mean : float or None
        Mean trail length in pixels.

    Returns
    -------
    int
        The pixel size to use for this subgroup.
    """
    try:
        size_from_unc = None
        
        # Method 1: Compute size from unc series
        if unc_series is not None:
            unc_vals = pd.to_numeric(unc_series, errors="coerce").dropna()
            if len(unc_vals) > 0:
                unc = float(unc_vals.max())
                if pixscale_arcsec_per_px and pixscale_arcsec_per_px > 0:
                    size_est = np.ceil((unc * multiplier) / float(pixscale_arcsec_per_px))
                else:
                    size_est = np.ceil(unc * multiplier)
                size_from_unc = int(np.clip(size_est, min_px, max_px))
        
        # Method 2: Compute size from trail projection (for 4-frame GIF)
        size_from_trail = None
        if (trail_px_est_mean is not None and angle_px_deg_mean is not None 
            and np.isfinite(trail_px_est_mean) and np.isfinite(angle_px_deg_mean)):
            # Convert angle to radians
            angle_rad = np.deg2rad(angle_px_deg_mean)
            # Calculate projections in x and y directions
            proj_x = abs(trail_px_est_mean * np.cos(angle_rad))
            proj_y = abs(trail_px_est_mean * np.sin(angle_rad))
            # Use maximum projection as base width, multiply by 4 for 4-frame GIF
            max_projection = max(proj_x, proj_y)
            size_est_trail = np.ceil(max_projection * 4)
            size_from_trail = int(np.clip(size_est_trail, min_px, max_px))
        
        # Use the maximum of both methods, or fallback
        if size_from_unc is not None and size_from_trail is not None:
            return max(size_from_unc, size_from_trail)
        elif size_from_unc is not None:
            return size_from_unc
        elif size_from_trail is not None:
            return size_from_trail
        else:
            return int(fallback_px)
            
    except Exception:
        return int(fallback_px)


def melbourne_today_str():
    tz = ZoneInfo("Australia/Melbourne")
    now_local = datetime.now(tz)
    return now_local.strftime("%Y-%m-%d")


def run_cmd(cmd, cwd=None, env=None, verbose=False):
    if verbose:
        print("[CMD]", cmd if isinstance(cmd, str) else " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        shell=isinstance(cmd, str),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if verbose and proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if verbose or True:
            print(proc.stderr, file=sys.stderr, end="")
    return proc.returncode, proc.stdout, proc.stderr


def find_daily_csv(data_dir, date_str):
    if not osp.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    candidates = []
    for fn in os.listdir(data_dir):
        low = fn.lower()
        if low.endswith(".csv") and low.startswith(date_str.lower() + "_mpc"):
            candidates.append(osp.join(data_dir, fn))
    if not candidates:
        raise FileNotFoundError(f"No CSV found in {data_dir!r} starting with '{date_str}_MPC'")
    candidates.sort(key=lambda p: osp.getmtime(p), reverse=True)
    return candidates[0]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def parse_ra_dec(ra_val, dec_val):
    try:
        ra = float(ra_val)
        dec = float(dec_val)
        return ra, dec
    except Exception:
        pass
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(str(ra_val), str(dec_val), unit=(u.hourangle, u.deg), frame="icrs")
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:
        try:
            from astropy.coordinates import Angle
            import astropy.units as u
            ra = Angle(str(ra_val), unit=u.deg).degree
            dec = Angle(str(dec_val), unit=u.deg).degree
            return float(ra), float(dec)
        except Exception as e:
            raise ValueError(f"Failed to parse RA/DEC: {ra_val!r}, {dec_val!r} ({e})")


def spherical_mean_radec(ra_deg_list: List[float], dec_deg_list: List[float]) -> Tuple[float, float]:
    ra = np.deg2rad(np.asarray(ra_deg_list, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg_list, dtype=float))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    X, Y, Z = x.mean(), y.mean(), z.mean()
    hyp = np.hypot(X, Y)
    ra_mean = np.arctan2(Y, X) % (2*np.pi)
    dec_mean = np.arctan2(Z, hyp)
    return float(np.rad2deg(ra_mean)), float(np.rad2deg(dec_mean))


def sanitize(s: str) -> str:
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", str(s).strip())
    return s.strip("_") or "NA"


def autodetect_column(df: pd.DataFrame, choices: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for key in choices:
        if key.lower() in cols:
            return cols[key.lower()]
    return None


def site_from_path(path: str) -> str:
    p = str(path).lower()
    if any(tok in p for tok in ["la_palma", "lapalma", "lp", "r13", "north"]):
        return "LaPalma"
    if any(tok in p for tok in ["siding", "spring", "ss", "e55", "south"]):
        return "SidingSpring"
    return "UnknownSite"


def detect_time_utc(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    dt_candidates = [
        "date_mid", "datetime", "time_utc", "time", "dateobs", "date_obs", "date-obs",
        "DATE_OBS", "DATE-OBS", "timestamp", "time_mid", "obs_time",
    ]
    for name in dt_candidates:
        col = autodetect_column(df, [name])
        if col and col in df.columns:
            ser = pd.to_datetime(df[col], utc=True, errors="coerce")
            if ser.notna().sum() >= max(1, int(0.5 * len(df))):
                return col, ser
    num_candidates = [("mjd", "mjd"), ("jd", "jd")]
    for key, kind in num_candidates:
        col = autodetect_column(df, [key])
        if col and col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().sum() == 0:
                continue
            try:
                from astropy.time import Time
                t = Time(vals.values, format=kind, scale="utc")
                py_dt = pd.to_datetime(t.to_datetime())
                ser = py_dt.dt.tz_localize("UTC")
                return col, ser
            except Exception:
                if kind == "mjd":
                    epoch = pd.Timestamp("1858-11-17T00:00:00Z")
                    ser = epoch + pd.to_timedelta(vals * 86400, unit="s")
                    return col, ser
                if kind == "jd":
                    epoch = pd.Timestamp("1858-11-17T00:00:00Z") - pd.to_timedelta(2400000.5 * 86400, unit="s")
                    ser = epoch + pd.to_timedelta(vals * 86400, unit="s")
                    return col, ser
    raise RuntimeError("Could not detect a usable time column. Provide one or include date_mid/DATE-OBS/mjd/jd in CSV.")


def build_run_client_cmd(run_client_path, server, ra, dec, size, image_paths: List[str],
                         local_temp, gif_out, keep_png=True, loose_wcs=False, verbose=False):
    py = sys.executable
    args = [
        py, run_client_path,
        "--server", server,
        "--ra", f"{ra:.8f}",
        "--dec", f"{dec:.8f}",
        "--size", str(int(size)),
        "--local_temp", local_temp,
        "--gif-out", gif_out,
    ]
    args += ["--images", *image_paths]
    if keep_png:
        args.append("--keep-png")
    if loose_wcs:
        args.append("--loose-wcs")
    if verbose:
        args.append("--verbose")
    return args


def _fix_gif_metadata_inplace(gif_path: str, duration_s: float = 0.2, loop: int = 0, verbose: bool = False) -> bool:
    """
    Ensure the GIF has non-zero per-frame durations and proper loop/disposal settings.
    Returns True if the file was rewritten (fixed), False otherwise.
    """
    if not (gif_path and osp.isfile(gif_path)):
        return False
    try:
        from PIL import Image
    except Exception:
        return False
    try:
        im = Image.open(gif_path)
    except Exception:
        return False
    n = getattr(im, 'n_frames', 1)
    durs = []
    need_fix = False
    for i in range(n):
        try:
            im.seek(i)
        except Exception:
            break
        d = im.info.get('duration', 0)
        try:
            di = int(d) if d is not None else 0
        except Exception:
            di = 0
        if di <= 0:
            need_fix = True
            durs.append(None)
        else:
            durs.append(di)
    if not need_fix:
        return False
    dur_ms = int(round(float(duration_s) * 1000.0)) if duration_s and duration_s > 0 else 100
    if dur_ms <= 0:
        dur_ms = 100
    durs = [(dur_ms if (d is None or d <= 0) else d) for d in durs]
    try:
        frames = []
        im.seek(0)
        for i in range(n):
            im.seek(i)
            frames.append(im.convert('P', palette=Image.ADAPTIVE, colors=256))
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=durs,
            loop=int(loop),
            disposal=2,
            optimize=False,
        )
        if verbose:
            print(f"[FIX] Rewrote GIF metadata: {gif_path} (frames={n}, dur_ms={dur_ms}, loop={loop})")
        return True
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to rewrite GIF metadata for {gif_path}: {e}")
        return False


def _parse_detector_stdout(stdout: str) -> dict:
    """
    Parse detect_mover_gif.py output in 'key,value' format to a dictionary.
    Attempts type conversion for True/False and numeric values.
    """
    out = {}
    for raw in (stdout or "").splitlines():
        if "," not in raw:
            continue
        k, v = raw.split(",", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
        else:
            try:
                if v.lower() in ("na", "none", ""):
                    out[k] = None
                else:
                    fv = float(v)
                    out[k] = int(fv) if fv.is_integer() else fv
            except Exception:
                out[k] = v
    return out


def _run_detector_on_gif(gif_path: str, detector_path: str,
                         vmax: float, vmin: float, angle_deg: float,
                         verbose: bool = False) -> dict:
    """
    Run detect_mover_gif.py with specified parameters:
      --vmax, --vmin, --step 0.5, --prior-angle-deg, --prior-angle-sigma 15,
      --trail-filter, --fwhm-px 3.0, --empirical-p
    Returns parsed detection information.
    """
    if not (os.path.isfile(detector_path) and os.path.isfile(gif_path)):
        return {"detected": False, "detector_rc": None, "detector_err": "detector or gif not found"}

    cmd = [
        sys.executable, detector_path, gif_path,
        "--vmax", f"{float(vmax):.1f}",
        "--vmin", f"{float(vmin):.1f}",
        "--step", "0.5",
        "--prior-angle-deg", f"{float(angle_deg):.1f}",
        "--prior-angle-sigma", "15",
        "--subframes", "11",
        "--trail-filter",
        "--fwhm-px", "3.0",
        "--empirical-p",
    ]
    rc, out, err = run_cmd(cmd, verbose=verbose)
    info = _parse_detector_stdout(out)
    detected = bool(info.get("Detection", False) or info.get("detected", False))
    return {
        "detected": detected,
        "detector_rc": rc,
        "detector_err": (err or "").strip(),
        "detector_snr_peak": info.get("snr_peak"),
        "detector_speed_pix_per_frame": info.get("speed_pix_per_frame"),
        "detector_angle_deg": info.get("angle_deg"),
        "detector_L_fit_px": info.get("L_fit_px"),
        "detector_w_fit_px": info.get("w_fit_px"),
        "detector_annotated_gif": info.get("annotated_gif"),
        "detector_track_csv": info.get("track_csv"),
    }


def main():
    ap = argparse.ArgumentParser(description="Daily workflow: object+site grouping, split by time gaps, one GIF per subgroup.")
    ap.add_argument("--csv", default=None, help="Path to a specific CSV (skip finder). If empty, find YYYY-MM-DD_MPC*.csv in --data-dir.")
    ap.add_argument("--date", default=None, help="Date label like YYYY-MM-DD (default: today in Australia/Melbourne)")
    ap.add_argument("--data-dir", default="./data", help="Directory where autochecker writes CSV (default: ./data)")
    ap.add_argument("--autochecker-path", default="./autochecker_MPClist_pipeline.py",
                    help="Path to autochecker_MPClist_pipeline.py")
    ap.add_argument("--skip-autochecker", action="store_true",
                    help="Do not run autochecker; just use the existing CSV for the given date/path")
    ap.add_argument("--run-client-path", default="./run_client_batch.py",
                    help="Path to run_client_batch.py")
    ap.add_argument("--server", default="warwick", choices=["monash", "warwick"], help='Server alias (default: "warwick")')
    ap.add_argument("--size", type=int, default=150, help="Cutout size in pixels (default: 150)")
    ap.add_argument("--thumbs-root", default="./thumbs", help="Where to save group GIFs/PNGs (default: ./thumbs)")
    ap.add_argument("--path-col", default="filepath", help="CSV column for remote FITS path (default: filepath)")
    ap.add_argument("--ra-col", default="RA_pre", help="CSV column for RA (deg or sexagesimal) (default: RA_pre)")
    ap.add_argument("--dec-col", default="DEC_pre", help="CSV column for DEC (deg or sexagesimal) (default: DEC_pre)")
    ap.add_argument("--obj-col", default=None, help="Column name for object ID (auto-detect if not given)")
    ap.add_argument("--site-col", default=None, help="Column name for site/station code (auto-detect if not given)")
    ap.add_argument("--gap-minutes", type=float, default=30.0, help="Max time gap within a subgroup in minutes (default: 30)")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N subgroups (debug)")
    ap.add_argument("--concurrency", type=int, default=2, help="Number of parallel downloads (default: 2)")
    ap.add_argument("--loose-wcs", action="store_true", help="Pass --loose-wcs to run_client_batch")
    ap.add_argument("--gif-fix-duration", type=float, default=0.2, help="Fallback per-frame duration (seconds) when GIF has 0/None durations (default: 0.2)")
    ap.add_argument("--gif-fix-loop", type=int, default=0, help="Loop count to enforce when rewriting GIF metadata (default: 0 = infinite)")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--detector-path", default="./detect_mover_gif.py",
                    help="Path to detect_mover_gif.py. If exists, run detection on each generated GIF.")
    args = ap.parse_args()

    date_str = args.date or melbourne_today_str()
    thumbs_root = ensure_dir(osp.abspath(args.thumbs_root))
    day_out = ensure_dir(osp.join(thumbs_root, f"{date_str}_MPC"))
    print(f"[INFO] Output dir: {day_out}")

    # # Step 1: run autochecker (optional) unless user supplied explicit CSV
    if not args.csv and not args.skip_autochecker:
        csv_data_dir = osp.abspath(args.data_dir)
        print(f"[STEP] Running autochecker to build CSV for {date_str} ... (data_dir={csv_data_dir})")
        if not osp.isfile(args.autochecker_path):
            print(f"[WARN] autochecker not found at {args.autochecker_path!r}; continuing (assuming CSV already exists).")
        else:
            rc, out, err = run_cmd([sys.executable, args.autochecker_path], verbose=args.verbose)
            if rc != 0:
                print(f"[WARN] autochecker exited with {rc}. Attempting to continue if CSV exists...")

    # Step 2: get CSV path
    csv_path = osp.abspath(args.csv) if args.csv else find_daily_csv(osp.abspath(args.data_dir), date_str)
    df = pd.read_csv(csv_path)
    print(f"[INFO] CSV: {csv_path}")

    # Step 3: load rows, clean columns
    df = pd.read_csv(csv_path)
    missing = [c for c in (args.path_col, args.ra_col, args.dec_col) if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}. Available: {list(df.columns)}")
    base = df[[args.path_col, args.ra_col, args.dec_col]].copy()
    base = base.dropna(subset=[args.path_col, args.ra_col, args.dec_col])
    base[args.path_col] = base[args.path_col].astype(str).str.strip()
    if 'unc' in df.columns:
        try:
            base['unc'] = pd.to_numeric(df['unc'], errors='coerce')
        except Exception:
            base['unc'] = pd.to_numeric(df['unc'], errors='ignore')

    obj_col = args.obj_col or autodetect_column(df, ["mpc", "mpc_name", "mpc_desig", "designation", "desig",
                                                      "object", "obj", "target", "name", "prov_desig",
                                                      "provisional", "mpc_id"])
    site_col = args.site_col or autodetect_column(df, ["site", "site_code", "obs_code", "station", "obs_site",
                                                        "goto_site", "site_name", "station_code"])
    if obj_col and obj_col in df.columns:
        base["__obj__"] = df[obj_col].astype(str).fillna("NA").str.strip()
    else:
        base["__obj__"] = "UnknownObject"
    if site_col and site_col in df.columns:
        base["__site__"] = df[site_col].astype(str).fillna("NA").str.strip()
    else:
        base["__site__"] = base[args.path_col].map(site_from_path)

    ra_deg = []
    dec_deg = []
    for ra_v, dec_v in zip(base[args.ra_col].values, base[args.dec_col].values):
        ra, dec = parse_ra_dec(ra_v, dec_v)
        ra_deg.append(ra)
        dec_deg.append(dec)
    base["__ra_deg__"] = ra_deg
    base["__dec_deg__"] = dec_deg

    time_col, t_utc = detect_time_utc(df)
    base["__t_utc__"] = t_utc.reindex(base.index)

    missing_time = base["__t_utc__"].isna().sum()
    if missing_time > 0:
        print(f"[WARN] Dropping {missing_time} rows with missing/invalid time in column {time_col!r}.")
    base = base.dropna(subset=["__t_utc__"]).copy()

    groups = []

    # Compute motion angle and trail length estimate per row
    # Pixel-angle convention: left=0°, down=90°, right=180°, up=270°
    plate_scale_arcsec_per_px = 1.26
    texp_s = 45.0

    def _autocol(df, names):
        col = autodetect_column(df, list(names))
        return col if (col in df.columns) else None

    ra_rate_col = _autocol(df, ["RA_rate_pre", "ra_rate_pre", "ra_rate", "ra_rate_as", "ra_rate_arcsec_s"])
    dec_rate_col = _autocol(df, ["DEC_rate_pre", "dec_rate_pre", "dec_rate", "dec_rate_as", "dec_rate_arcsec_s"])
    dtotal_col = _autocol(df, ["d_total", "rate_total", "total_rate", "sky_rate", "motion_total"])

    base["__angle_px_deg__"] = float("nan")
    base["__trail_px_est__"] = float("nan")

    if ra_rate_col and dec_rate_col:
        rr = np.asarray(pd.to_numeric(df[ra_rate_col], errors="coerce"))
        dd = np.asarray(pd.to_numeric(df[dec_rate_col], errors="coerce"))
        vx = rr
        vy = -dd
        ang = np.degrees(np.arctan2(vy, vx)) % 360.0
        base.loc[:, "__angle_px_deg__"] = pd.Series(ang, index=df.index).reindex(base.index).values
    else:
        print("[WARN] RA/DEC rate columns not found; skipping angle computation.")

    if dtotal_col:
        dtot = pd.to_numeric(df[dtotal_col], errors="coerce") / 3600
        trail_px = (dtot * texp_s) / plate_scale_arcsec_per_px
        base.loc[:, "__trail_px_est__"] = trail_px.reindex(base.index).values
    else:
        print("[WARN] dtotal column not found; skipping trail length estimate.")

    def circular_mean_deg(vals: np.ndarray) -> float:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan")
        rad = np.radians(vals % 360.0)
        C = np.cos(rad).mean()
        S = np.sin(rad).mean()
        if C == 0 and S == 0:
            return float("nan")
        return (float(np.degrees(np.arctan2(S, C))) + 360.0) % 360.0

    # Initialize __sub__ column
    base["__sub__"] = 0
    gap_sec = float(args.gap_minutes) * 60.0
    for site, g_site in base.groupby("__site__", dropna=False, sort=True):
        for obj, g in g_site.groupby("__obj__", dropna=False, sort=True):
            g_sorted = g.sort_values("__t_utc__").copy()
            
            # First-level grouping: by time interval
            diffs = g_sorted["__t_utc__"].diff().dt.total_seconds()
            newgrp = (diffs.isna()) | (diffs >= 110.0)
            g_sorted["__sub__"] = newgrp.cumsum()
            
            # Extract UT number
            g_sorted["__ut_num__"] = g_sorted[args.path_col].str.extract(r'_ut(\d+)\.fits')[0]
            
            # Second-level grouping: within each __sub__, split by UT number
            # Generate a unique ID for each (site, obj, __sub__, ut_num) combination
            g_sorted["__sub__"] = g_sorted.groupby(["__sub__", "__ut_num__"]).ngroup() + 1
            # Write __sub__ back to base so worker can access it
            base.loc[g_sorted.index, "__sub__"] = g_sorted["__sub__"].values

            for sub_id, sg in g_sorted.groupby("__sub__"):
                paths = sg[args.path_col].dropna().astype(str).tolist()
                if not paths:
                    print("[DEBUG] EMPTY paths subgroup, sg index:", sg.index.tolist())
                    continue
                ra_mean, dec_mean = spherical_mean_radec(sg["__ra_deg__"].tolist(), sg["__dec_deg__"].tolist())
                t0 = sg["__t_utc__"].min()
                t1 = sg["__t_utc__"].max()
                span_min = float((t1 - t0).total_seconds() / 60.0)

                ang_mean = circular_mean_deg(sg["__angle_px_deg__"].values) if "__angle_px_deg__" in sg.columns else float("nan")
                trail_mean = float(sg["__trail_px_est__"].mean()) if "__trail_px_est__" in sg.columns else float("nan")

                try:
                    size_px = compute_size_from_unc(
                        sg.get('unc', pd.Series(dtype=float)),
                        fallback_px=args.size,
                        multiplier=3.0,
                        min_px=150,
                        max_px=300,
                        angle_px_deg_mean=ang_mean,
                        trail_px_est_mean=trail_mean,
                    )
                except Exception:
                    size_px = int(args.size)
                    
                groups.append({
                    "object": obj,
                    "site": site,
                    "sub_id": int(sub_id),
                    "n": len(paths),
                    "ra_deg": ra_mean,
                    "dec_deg": dec_mean,
                    "angle_px_deg_mean": ang_mean,
                    "trail_px_est_mean": trail_mean,
                    "size_px": int(size_px),
                    "t_start": t0.isoformat(),
                    "t_end": t1.isoformat(),
                    "span_min": span_min,
                    "paths": paths,
                })

    if not groups:
        print("[WARN] No subgroups to process.")
        return 0

    if args.limit is not None:
        groups = groups[:int(args.limit)]

    grp_manifest = osp.join(day_out, "group_manifest.csv")
    ensure_dir(osp.dirname(grp_manifest))
    pd.DataFrame([{k: (v if k != "paths" else ";".join(v)) for k, v in d.items()} for d in groups]).to_csv(grp_manifest, index=False)
    print(f"[INFO] Group manifest: {grp_manifest} (subgroups={len(groups)})")

    def worker(gidx, gdict):
        obj    = gdict["object"]
        site   = gdict["site"]
        sub_id = gdict["sub_id"]
        ra     = gdict["ra_deg"]
        dec    = gdict["dec_deg"]
    
        # Directly use the packed 'paths' list from gdict (keep the original order)
        paths = [str(p) for p in gdict.get("paths", [])]
        if not paths:
            raise ValueError(f"paths is empty for obj={obj}, site={site}, sub_id={sub_id}")
    
        subdir   = ensure_dir(osp.join(day_out, f"{sanitize(site)}__{sanitize(obj)}__sub{sub_id:02d}"))
        gif_name = f"{sanitize(obj)}__{sanitize(site)}__sub{sub_id:02d}.gif"
        gif_path = osp.join(subdir, gif_name)
        
        # Before calling run_client, clean old PNG/GIF files to avoid mixing with leftover frames
        for fn in os.listdir(subdir):
            if fn.lower().endswith((".png", ".gif")):
                try:
                    os.remove(osp.join(subdir, fn))
                except Exception:
                    pass
        
        # Use size_px precomputed for this subgroup (already stored in gdict)
        size_px = int(gdict.get("size_px", args.size))
        
        cmd = build_run_client_cmd(
            run_client_path=osp.abspath(args.run_client_path),
            server=args.server, ra=ra, dec=dec, size=size_px,
            image_paths=paths, local_temp=subdir, gif_out=gif_path,
            keep_png=True, loose_wcs=args.loose_wcs, verbose=args.verbose,
        )
            
        rc, out, err = run_cmd(cmd, verbose=args.verbose)
        status = "OK" if rc == 0 else f"ERR({rc})"
        
        det = {}
        try:
            _fix_gif_metadata_inplace(gif_path, duration_s=args.gif_fix_duration, loop=args.gif_fix_loop, verbose=args.verbose)
            
            try:
                vmax = float(gdict.get("trail_px_est_mean", float("nan"))) * 1.5
                vmin = float(gdict.get("trail_px_est_mean", float("nan"))) * 0.5
                angle = float(gdict.get("angle_px_deg_mean", float("nan")))
            except Exception:
                vmax = vmin = angle = float("nan")

            if (status == "OK" and os.path.isfile(gif_path)
                    and os.path.isfile(args.detector_path)
                    and np.isfinite(vmax) and np.isfinite(vmin) and np.isfinite(angle)):
                det = _run_detector_on_gif(
                    gif_path=gif_path, detector_path=args.detector_path,
                    vmax=vmax, vmin=vmin, angle_deg=angle, verbose=args.verbose
                )
        except Exception:
            pass
            
        pngs = [osp.join(subdir, fn) for fn in os.listdir(subdir) if fn.lower().endswith(".png")]

        row = {
            "group_index": gidx, "object": obj, "site": site, "sub_id": sub_id, "n": len(paths),
            "ra_deg": ra, "dec_deg": dec, "size_px": int(gdict.get("size_px", args.size)), "png_count": len(pngs),
            "gif_path": gif_path if osp.isfile(gif_path) else "", "status": status,
            "stderr_tail": (err or "").splitlines()[-1] if err else "",
            "t_start": gdict["t_start"], "t_end": gdict["t_end"], "span_min": gdict["span_min"],
        }
        row.update(det)
        return row

    print(f"[STEP] Fetching thumbnails for {len(groups)} subgroups with concurrency={args.concurrency} ...")
    manifest_rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as ex:
        futures = [ex.submit(worker, i, g) for i, g in enumerate(groups)]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                manifest_rows.append(res)
                print(f"[{res['status']}] obj={res['object']} site={res['site']} sub={res['sub_id']} n={res['n']} span={res['span_min']:.1f}m gif={'OK' if res['gif_path'] else 'NA'}")
            except Exception as e:
                print(f"[ERROR] Worker failed: {e}", file=sys.stderr)
                raise

    manifest_path = osp.join(day_out, "manifest_groups.csv")
    ensure_dir(osp.dirname(manifest_path))
    pd.DataFrame(manifest_rows).sort_values(["object", "site", "sub_id"]).to_csv(manifest_path, index=False)
    print(f"[DONE] Saved manifest: {manifest_path}")
    
    try:
        df_groups = pd.DataFrame([
            {k: (v if k != "paths" else ";".join(v)) for k, v in d.items()}
            for d in groups
        ])
        df_manifest = pd.DataFrame(manifest_rows)
        det_cols = [
            "detected", "detector_snr_peak", "detector_speed_pix_per_frame",
            "detector_angle_deg", "detector_L_fit_px", "detector_w_fit_px"]
        keep_cols = ["object", "site", "sub_id"] + [c for c in det_cols if c in df_manifest.columns]
        df_out = df_groups.merge(df_manifest[keep_cols], on=["object", "site", "sub_id"], how="left")
        df_out.to_csv(grp_manifest, index=False)
        print(f"[INFO] Group manifest updated with detection results: {grp_manifest}")
    except Exception as e:
        print(f"[WARN] Failed to update group_manifest with detection results: {e}")

    print(f"[DONE] Thumbnails directory: {day_out}")
    
    # try:
    #     ppt_path_local = osp.join(osp.dirname(__file__), "build_daily_ppt.py")
    #     ppt_script = ppt_path_local if osp.isfile(ppt_path_local) else "build_daily_ppt.py"
    #     if osp.isfile(ppt_script):
    #         print("[STEP] Running build_daily_ppt.py to generate PPT ...")
    #         rc, out, err = run_cmd([sys.executable, ppt_script],
    #                                cwd=osp.dirname(osp.abspath(__file__)),
    #                                verbose=args.verbose)
    #         if rc != 0:
    #             print(f"[WARN] build_daily_ppt.py exited with code {rc}")
    #     else:
    #         print("[WARN] build_daily_ppt.py not found; skipping PPT generation.")
    # except Exception as e:
    #     print(f"[WARN] Failed to run build_daily_ppt.py: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
