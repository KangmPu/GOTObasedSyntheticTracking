#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['PGPASSFILE'] = os.path.expanduser('~/.pgpass')

from pathlib import Path
import re
import io
import ast
import math
import time
import argparse
import warnings
import contextlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- DB ----------
import gotodb as gdb


def goto_silent():
    """Silent wrapper for gdb.goto() to suppress startup prints."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return gdb.goto()


def query_df(db, sql, params=None):
    return db.query(sql, params or {})

# ---------- Astronomy ----------
from astroquery.mpc import MPC
import astropy.units as u
from astropy.coordinates import SkyCoord

# ---------- Interp / modeling ----------
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, RationalQuadratic
from sklearn.exceptions import ConvergenceWarning
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend

# ---------- Geometry ----------
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from shapely.strtree import STRtree
try:
    from shapely import points as shapely_points
    from shapely import bounds as shapely_bounds
    _SHAPELY_V2 = True
except Exception:
    _SHAPELY_V2 = False

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

# ========================= Config =========================
SITE_PARAMS = {
    "La Palma":      dict(lat=28.76012,  lon=-17.87929, alt=2348),
    "Siding Spring": dict(lat=-31.2734,  lon=149.06411, alt=1137),
}
ALIASES = {
    "Side Spring": "Siding Spring",
    "SideSpring": "Siding Spring",
    "SidingSpring": "Siding Spring",
    "GOTO-N": "La Palma", "GOTON": "La Palma", "LP": "La Palma",
    "GOTO-S": "Siding Spring", "GOTOS": "Siding Spring", "SSO": "Siding Spring",
}

SITE_CHOICES = ["BOTH", "LP", "SSO"]

# ========================= Utilities =========================

def _tprefix(filepath: str):
    """Extract tN (t1..t4) from filepath."""
    if not isinstance(filepath, str):
        return None
    mobj = re.search(r"/(t[1-4])[_\.]", filepath.lower())
    return mobj.group(1) if mobj else None


def safe_parse_footprint(v):
    """Parse polygon from text/list; return shapely Polygon or None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        pts = ast.literal_eval(v) if isinstance(v, str) else v
        if pts is None or len(pts) < 3:
            return None
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except Exception:
        return None


def ensure_list_length(seq, L, fill=np.nan):
    """Pad or truncate a sequence to length L."""
    if seq is None:
        return [fill] * L
    if isinstance(seq, (pd.Series, np.ndarray)):
        seq = seq.tolist()
    elif not isinstance(seq, (list, tuple)):
        seq = [seq]
    n = len(seq)
    if n >= L:
        return list(seq[:L])
    return list(seq) + [fill] * (L - n)


# ========================= Observations from image.single =========================

def load_observations_from_image_single(lookback_days=3):
    """
    Fetch observations from image.single between [now - lookback_days, now].

    Returns columns: site, obs_date, astrom_footprint, filepath, image_type, obs_date_utc, telescope
    """
    db = goto_silent()

    tmax = pd.Timestamp.utcnow()
    tmin = tmax - timedelta(days=lookback_days)
    print(f"[OBS] Fetch window: {tmin} -> {tmax}")

    sql_recent = """
        SELECT
          s.id                             AS id,
          r.site                           AS site,
          s.date_mid                       AS obs_date,
          CAST(s.astrom_footprint AS TEXT) AS astrom_footprint,
          s.filepath                       AS filepath,
          s.image_type                     AS image_type,
          s.astrom_n_match,
          s.astrom_uncert_med,
          s.astrom_ra_centre,
          s.astrom_dec_centre,
          s.astrom_pixel_scale
        FROM "image"."single" AS s
        LEFT JOIN "raw"."science" AS r
          ON r.id = s.raw_id
        WHERE s.date_mid BETWEEN %(tmin)s AND %(tmax)s
        ORDER BY s.date_mid ASC;
    """
    df = query_df(db, sql_recent, {"tmin": tmin, "tmax": tmax})

    if "filepath" in df.columns:
        df["telescope"] = df["filepath"].map(_tprefix)
    else:
        df["telescope"] = None

    df["site"] = df["site"].astype(str).str.strip()
    df["obs_date_utc"] = pd.to_datetime(df["obs_date"], utc=True, errors="coerce")

    if not df.empty:
        print("[OBS] Per-site counts in window:")
        print(df.groupby("site").size().rename("rows").sort_index())

    return df


# ========================= Ephemerides (MPC) =========================

def _parse_rate_as_asmin(x):
    """Parse rate strings to arcsec/min float values."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().lower()
    m = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", s)
    if not m:
        return np.nan
    val = float(m.group(0))
    if ('hour' in s) or ('/hr' in s) or ('/h' in s):
        val = val / 60.0
    elif ('/day' in s) or ('/d' in s):
        val = val / 1440.0
    elif ('/s' in s) or ('/sec' in s) or ('/second' in s):
        val = val * 60.0
    return val


# ---- safe wrapper to avoid proper-motion unit conversion crashes ----
from numpy.core._exceptions import _UFuncNoLoopError as _UFuncNoLoopError_

def _safe_get_ephemeris(target_name, **kw):
    """
    Call MPC.get_ephemeris with a fallback: if unit-conversion fails due to
    non-numeric proper-motion fields, retry without proper_motion args.
    """
    try:
        return MPC.get_ephemeris(target_name, **kw)
    except _UFuncNoLoopError_:
        # Remove potentially problematic keywords and retry
        kw2 = dict(kw)
        kw2.pop('proper_motion', None)
        kw2.pop('proper_motion_unit', None)
        return MPC.get_ephemeris(target_name, **kw2)


def fetch_ephemeris(target_name: str, start_date: str, step_str: str, step_count: int,
                    site_lon: float, site_lat: float, site_height: float) -> pd.DataFrame:
    step = step_str
    loc  = (site_lon*u.deg, site_lat*u.deg, site_height*u.m)

    # Use safe wrapper to avoid crashes with some MPC rows (e.g., non-numeric proper motions)
    table = _safe_get_ephemeris(
        target_name,
        start=start_date,
        step=step,
        number=step_count,
        location=loc,
        eph_type='equatorial',
        proper_motion='coordinate',
        suppress_daytime=False
    )

    df = table.to_pandas()

    rename_map = {
        'Date'        : 'Date',
        'RA'          : 'RA',
        'Dec'         : 'DEC',
        'Delta'       : 'Delta (AU)',
        'r'           : 'r (AU)',
        'elong'       : 'Elongation',
        'phase'       : 'Phase',
        'V'           : 'Vmag',
        'Altitude'    : 'Alt',
        'Sun altitude': 'Sun_Alt',
        'dRA'         : 'RA_rate',
        'dDec'        : 'Dec_rate',
        'Uncertainty 3sig' : 'unc',
    }
    for k in list(rename_map.keys()):
        if k not in df.columns:
            del rename_map[k]
    df = df.rename(columns=rename_map)

    df['Object'] = target_name

    # Robust RA/DEC to degrees (avoid sexagesimal strings later)
    c = SkyCoord(df['RA'], df['DEC'], unit=('hourangle', 'deg'))
    df['RA_deg']  = c.ra.deg
    df['DEC_deg'] = c.dec.deg

    # Parse rates to numeric arcsec/min
    src_ra = None
    src_de = None
    for cand in ['RA_rate', 'SkyMotion (RA)']:
        if cand in df.columns:
            src_ra = df[cand]
            break
    for cand in ['Dec_rate', 'SkyMotion (DEC)']:
        if cand in df.columns:
            src_de = df[cand]
            break
    df['ra_rate']  = pd.Series(np.nan, index=df.index) if src_ra is None else src_ra.apply(_parse_rate_as_asmin)
    df['dec_rate'] = pd.Series(np.nan, index=df.index) if src_de is None else src_de.apply(_parse_rate_as_asmin)

    if 'RA_rate' in df.columns and 'Dec_rate' in df.columns:
        df['d_total'] = np.sqrt(pd.to_numeric(df['ra_rate'], errors='coerce')**2 +
                                pd.to_numeric(df['dec_rate'], errors='coerce')**2)
    elif 'd_total' not in df.columns:
        df['d_total'] = 0.0

    want = ['Date','RA','DEC','RA_deg','DEC_deg','Vmag','Alt','Sun_Alt',
            'Elongation','Phase','Delta (AU)','r (AU)','d_total','ra_rate','dec_rate','Object','unc']
    out_cols = [c for c in want if c in df.columns]

    return df[out_cols]



# =========================

def _circ(a, b):
    try:
        return (float(a) - float(b) + 540.0) % 360.0 - 180.0
    except Exception:
        return float('nan')

def interp_radec_minutely_slerp(t_list, ra_deg_list, dec_deg_list, n_points=None):
    """
    Segment-wise shortest-arc great-circle interpolation (SLERP).
    Returns (t_str, RA_deg, DEC_deg) sampled on a 1-minute time grid.
    - t_list: original time series (may mix str and pandas.Timestamp)
    - ra_deg_list, dec_deg_list: original RA/Dec in degrees (may include 0–360° wrapping)
    """
    import numpy as np
    import pandas as pd

    t = pd.to_datetime(pd.Series(t_list), errors="coerce", utc=True)
    ra = pd.to_numeric(pd.Series(ra_deg_list), errors="coerce")
    de = pd.to_numeric(pd.Series(dec_deg_list), errors="coerce")
    m = t.notna() & ra.notna() & de.notna()
    t, ra, de = t[m], ra[m], de[m]
    if len(t) == 0:
        return [], np.array([]), np.array([])
    if len(t) == 1:
        t0 = t.iloc[0].floor("T")
        t1 = t.iloc[0].ceil("T")
        grid = pd.date_range(t0, t1, freq="T")
        ra0 = float(ra.iloc[0]) % 360.0
        de0 = float(de.iloc[0])
        if n_points and len(grid) > n_points:
            grid = grid[:n_points]
        return [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in grid], \
               np.full(len(grid), ra0), np.full(len(grid), de0)

    order = np.argsort(t.values)
    t  = t.iloc[order].reset_index(drop=True)
    ra = (ra.iloc[order].astype(float) % 360.0).reset_index(drop=True)
    de =  de.iloc[order].astype(float).reset_index(drop=True)

    t0 = t.min().floor("T")
    t1 = t.max().ceil("T")
    grid = pd.date_range(t0, t1, freq="T")
    if len(grid) == 0:
        grid = pd.DatetimeIndex([t0])

    src_sec = (t - t0).dt.total_seconds().to_numpy()
    dst_sec = np.asarray((grid - t0).total_seconds(), dtype=float)

    right = np.searchsorted(src_sec, dst_sec, side="right")
    left  = np.clip(right - 1, 0, len(src_sec) - 2)
    right = left + 1

    seg_len = src_sec[right] - src_sec[left]
    seg_len[seg_len == 0] = 1.0
    alpha = (dst_sec - src_sec[left]) / seg_len
    alpha = np.asarray(alpha, dtype=float)

    ra_l = np.deg2rad(ra.to_numpy())
    de_l = np.deg2rad(de.to_numpy())
    c = np.cos(de_l)
    x = c * np.cos(ra_l)
    y = c * np.sin(ra_l)
    z = np.sin(de_l)

    # 6) vectorization
    v0 = np.stack([x[left],  y[left],  z[left]],  axis=1)  # (N,3)
    v1 = np.stack([x[right], y[right], z[right]], axis=1)  # (N,3)

    # 7) SLERP
    dot = np.einsum("ij,ij->i", v0, v1)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_th = np.sin(theta)
    use_lerp = sin_th < 1e-12

    out = np.empty_like(v0)

    # linear interpolation
    if np.any(use_lerp):
        a_lin = alpha[use_lerp][:, None]
        out[use_lerp] = (1 - a_lin) * v0[use_lerp] + a_lin * v1[use_lerp]

    # SLERP
    if np.any(~use_lerp):
        a = np.asarray(alpha[~use_lerp], dtype=float)
        th = np.asarray(theta[~use_lerp], dtype=float)
        s0 = np.sin((1 - a) * th) / np.sin(th)
        s1 = np.sin(a * th)        / np.sin(th)
        s0 = np.asarray(s0, dtype=float)[:, None]
        s1 = np.asarray(s1, dtype=float)[:, None]
        out[~use_lerp] = s0 * v0[~use_lerp] + s1 * v1[~use_lerp]

    # normalization back to RA/DEC
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    out /= nrm
    ox, oy, oz = out[:, 0], out[:, 1], out[:, 2]
    ra_i  = (np.degrees(np.arctan2(oy, ox)) + 360.0) % 360.0
    dec_i =  np.degrees(np.arcsin(np.clip(oz, -1.0, 1.0)))

    t_str = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in grid]
    if n_points and len(t_str) > n_points:
        t_str  = t_str[:n_points]
        ra_i   = ra_i[:n_points]
        dec_i  = dec_i[:n_points]

    return t_str, ra_i, dec_i



def gp_interpolate_curve_alt(t, y, n_points=600, length_scale=1.0, noise_level=1e-2, rbf_length_scale=10000):
    """Alternative GP interpolation with simple periodic component detection (FFT heuristic)."""
    t_ser = pd.to_datetime(pd.Series(t), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    y_ser = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = t_ser.notna() & y_ser.notna()
    t_ser = t_ser[mask]
    y_ser = y_ser[mask].astype(float)

    if len(t_ser) == 0:
        return [], np.array([])
    if len(t_ser) == 1:
        return [t_ser.iloc[0].strftime("%Y-%m-%d %H:%M:%S")] * n_points, np.full(n_points, float(y_ser.iloc[0]))

    t0 = t_ser.min()
    t_sec = ((t_ser - t0) / np.timedelta64(1, 's')).to_numpy()

    t_uniform = np.linspace(t_sec.min(), t_sec.max(), len(t_sec))
    y_uniform = np.interp(t_uniform, t_sec, y_ser.to_numpy())
    y_detrended = detrend(y_uniform)
    yf = rfft(y_detrended)
    xf = rfftfreq(len(t_uniform), d=(t_uniform[1] - t_uniform[0]))
    peak = np.argmax(np.abs(yf[1:])) + 1
    period = 1 / xf[peak] if xf[peak] != 0 else (t_uniform.max() - t_uniform.min() + 1.0)

    kernel = (ExpSineSquared(length_scale=length_scale, periodicity=period) + WhiteKernel(noise_level=noise_level)) * RBF(length_scale=rbf_length_scale)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(t_sec.reshape(-1, 1), y_ser.to_numpy())

    t_interp_sec = np.linspace(t_sec.min(), t_sec.max(), n_points)
    y_interp = gp.predict(t_interp_sec.reshape(-1, 1))

    t_interp_dt  = [t0 + timedelta(seconds=float(s)) for s in t_interp_sec]
    t_interp_str = [d.strftime("%Y-%m-%d %H:%M:%S") for d in t_interp_dt]
    return t_interp_str, y_interp


def sort_ephemerides_by_object_df(data: pd.DataFrame):
    """Group rows by object and prepare series for interpolation."""
    data = data.copy()
    data['datetime'] = pd.to_datetime(data['Date'], errors='coerce', utc=True)
    unique_objects = data["Object"].unique()
    out = []
    for obj in unique_objects:
        obj_data = data[data["Object"] == obj]
        t_l    = obj_data["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

        # Prefer numeric degree coordinates to avoid sexagesimal strings (FIX)
        x = obj_data.get("RA_deg",  obj_data.get("RA",  pd.Series([np.nan]*len(obj_data)))).tolist()
        y = obj_data.get("DEC_deg", obj_data.get("DEC", pd.Series([np.nan]*len(obj_data)))).tolist()

        z      = obj_data.get("Alt", pd.Series([np.nan]*len(obj_data))).tolist()
        z_sun  = obj_data.get("Sun_Alt", pd.Series([np.nan]*len(obj_data))).tolist()
        mag    = obj_data.get("Vmag", pd.Series([np.nan]*len(obj_data))).tolist()
        d_total= obj_data.get("d_total", pd.Series([0.0]*len(obj_data))).tolist()
        unc    = obj_data.get("unc", pd.Series([0.0]*len(obj_data))).tolist()

        xrate_src = obj_data.get("ra_rate", pd.Series([np.nan]*len(obj_data)))
        if xrate_src.isna().all() and "SkyMotion (RA)" in obj_data.columns:
            xrate_src = obj_data["SkyMotion (RA)"]
        yrate_src = obj_data.get("dec_rate", pd.Series([np.nan]*len(obj_data)))
        if yrate_src.isna().all() and "SkyMotion (DEC)" in obj_data.columns:
            yrate_src = obj_data["SkyMotion (DEC)"]
        xrate  = pd.to_numeric(xrate_src, errors='coerce').tolist()
        yrate  = pd.to_numeric(yrate_src, errors='coerce').tolist()

        out.append([t_l, x, y, z, z_sun, mag, None, d_total, xrate, yrate, unc])
    return out


def batch_fetch_ephemerides(obj_list, start_date, step_str, step_count,
                            site_lon, site_lat, site_height,
                            n_points):
    """
    Build dense/interpolated ephemerides per object for a given site.
    Returns columns: datetime, RA, DEC, ALT, Sun_Alt, obj_name, Vmag, unc, d_total, ra_rate, dec_rate
    """
    if isinstance(obj_list, pd.DataFrame):
        if 'Obj_Value' in obj_list.columns:
            obj_list = obj_list['Obj_Value'].tolist()
        elif 'Obj_Name' in obj_list.columns:
            obj_list = obj_list['Obj_Name'].tolist()
        else:
            raise ValueError("DataFrame must contain 'Obj_Value' or 'Obj_Name'.")
    elif isinstance(obj_list, pd.Series):
        obj_list = obj_list.tolist()
    elif not isinstance(obj_list, list):
        raise TypeError("obj_list must be list/Series/DataFrame")

    df_all = pd.DataFrame(columns=[
        "datetime","RA","DEC","ALT","Sun_Alt","obj_name",
        "Vmag",'unc',"d_total","ra_rate","dec_rate"
    ])

    for obj_value in tqdm(obj_list, desc="Ephemerides", unit="obj"):
        try:
            df = fetch_ephemeris(
                target_name=obj_value,
                start_date=start_date,
                step_str=step_str,
                step_count=step_count,
                site_lon=site_lon,
                site_lat=site_lat,
                site_height=site_height
            )
            if df.empty:
                continue

            sorted_data = sort_ephemerides_by_object_df(df)
            if not sorted_data:
                continue

            for item in sorted_data:
                (t_l, ra_l, dec_l, alt_l, sun_l, mag_l, _, dtot_l, xrate_l, yrate_l, unc_l) = item

                t_int, ra_int, dec_int = interp_radec_minutely_slerp(t_l, ra_l, dec_l, n_points=n_points)

                L = min(len(t_int), len(ra_int), len(dec_int))
                t_int   = list(t_int[:L])
                ra_int  = ensure_list_length(ra_int,  L, fill=np.nan)
                dec_int = ensure_list_length(dec_int, L, fill=np.nan)

                _, alt_int   = gp_interpolate_curve_alt(t_l, alt_l,  n_points=n_points, length_scale=1.0, noise_level=1e-2)
                _, s_alt_int = gp_interpolate_curve_alt(t_l, sun_l,  n_points=n_points, length_scale=1.0, noise_level=1e-2)
                _, mag_int   = gp_interpolate_curve_alt(t_l, mag_l,  n_points=n_points, length_scale=1.0, noise_level=1e-2)
                _, dtot_int  = gp_interpolate_curve_alt(t_l, dtot_l, n_points=n_points, length_scale=1.0, noise_level=1e-2)
                _, unc_int   = gp_interpolate_curve_alt(t_l, unc_l,  n_points=n_points, length_scale=1.0, noise_level=1e-2)

                alt_int   = ensure_list_length(alt_int,   L, fill=np.nan)
                s_alt_int = ensure_list_length(s_alt_int, L, fill=np.nan)
                mag_int   = ensure_list_length(mag_int,   L, fill=np.nan)
                dtot_int  = ensure_list_length(dtot_int,  L, fill=np.nan)
                unc_int   = ensure_list_length(unc_int,   L, fill=np.nan)

                def _safe_interp_rate(vals):
                    _, arr = gp_interpolate_curve_alt(t_l, vals, n_points=n_points, length_scale=1.0, noise_level=1e-2)
                    return arr if arr is not None and len(arr) > 0 else []

                ra_rate_int  = ensure_list_length(_safe_interp_rate(xrate_l), L, fill=np.nan)
                dec_rate_int = ensure_list_length(_safe_interp_rate(yrate_l), L, fill=np.nan)

                df_temp = pd.DataFrame({"datetime": t_int})
                df_temp["RA"]        = ra_int
                df_temp["DEC"]       = dec_int
                df_temp["ALT"]       = alt_int
                df_temp["Sun_Alt"]   = s_alt_int
                df_temp["obj_name"]  = obj_value
                df_temp["Vmag"]      = mag_int
                df_temp["unc"]       = unc_int
                df_temp["d_total"]   = dtot_int
                df_temp["ra_rate"]   = ra_rate_int
                df_temp["dec_rate"]  = dec_rate_int

                df_all = pd.concat([df_all, df_temp], ignore_index=True)

        except Exception as e:
            tqdm.write(f" Failed to fetch {obj_value}: {e}")
            continue

    return df_all


# ========================= Matching =========================

def _compute_bounds(polys):
    if _SHAPELY_V2:
        return shapely_bounds(polys)
    # fallback for shapely<2
    out = []
    for p in polys:
        if p is None:
            out.append((np.nan, np.nan, np.nan, np.nan))
        else:
            out.append(p.bounds)
    return np.array(out)


def _make_points(ras, decs):
    if _SHAPELY_V2:
        return shapely_points(ras, decs)
    return [Point(float(x), float(y)) for x, y in zip(ras, decs)]


def match_ephemeris_by_site(combined_df, recent_obs_df,
                            time_tolerance_sec=2, per_minute_cap=None):
    """
    Vectorized spherical polygon inside-test using half-space normals.
    For each observation, only compare the time-nearest ephemeris point per object.
    Output schema matches the previous function.

    Requires:
      - combined_df columns: datetime, ra, dec, (optional) obj_name or name, site
      - recent_obs_df columns: obs_date_utc, astrom_footprint, (optional) site, telescope, limiting_mag, filepath, id, obs_date
      - safe_parse_footprint(v) -> shapely Polygon with (lon=RA_deg, lat=DEC_deg) exterior
    """
    import pandas as pd
    import numpy as np
    from tqdm import tqdm as _tqdm

    # ---------- helpers ----------
    def _require_cols(df, cols, name):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name} missing columns: {', '.join(missing)}")

    def _coerce_ra_deg_series(s):
        """Coerce RA (already in degrees) and wrap into [0,360). (FIX: remove hours->deg guessing)"""
        a = pd.to_numeric(s, errors='coerce').astype(float)
        return np.mod(a, 360.0)

    def _radec_deg_to_unit(ra_deg, dec_deg):
        ra = np.deg2rad(ra_deg)
        dec = np.deg2rad(dec_deg)
        c = np.cos(dec)
        return np.stack([c*np.cos(ra), c*np.sin(ra), np.sin(dec)], axis=-1)

    def _normalize_rows(v):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        n = np.where(n == 0, 1.0, n)
        return v / n

    def _poly_to_unit_vertices(poly):
        xs, ys = np.asarray(poly.exterior.coords.xy[0]), np.asarray(poly.exterior.coords.xy[1])
        coords = np.stack([xs, ys], axis=1)
        if len(coords) >= 2 and np.allclose(coords[0], coords[-1], atol=1e-9):
            coords = coords[:-1]
        if coords.shape[0] < 3:
            return None, None

        # FIX: RA unwrap to be robust to 0/360 wrap (and any pre-unwrapped values)
        ra_raw = np.asarray(coords[:, 0], dtype=float)
        ra_rad = np.deg2rad(ra_raw)
        ra_rad = np.unwrap(ra_rad)
        ra_deg = np.rad2deg(ra_rad)

        dec_deg = coords[:, 1]
        V = _radec_deg_to_unit(ra_deg, dec_deg)
        Nv = V.shape[0]
        E = np.stack([np.arange(Nv), (np.arange(Nv)+1) % Nv], axis=1)
        return V, E

    def _edge_inward_normals(V, E):
        Vi = V[E[:, 0]]
        Vj = V[E[:, 1]]
        n = np.cross(Vi, Vj)
        n = _normalize_rows(n)
        c = _normalize_rows(V.mean(axis=0, keepdims=True))
        sign = np.sign(np.einsum('ij,kj->i', n, c))
        sign[sign == 0] = 1.0
        n *= sign[:, None]
        return n

    # -------- normalize inputs --------
    comb = combined_df.copy()
    rec  = recent_obs_df.copy()
    comb.columns = [c.strip().lower() for c in comb.columns]
    rec.columns  = [c.strip().lower() for c in rec.columns]

    _require_cols(comb, ['datetime', 'ra', 'dec'], "combined_df")
    _require_cols(rec,  ['obs_date_utc', 'astrom_footprint'], "recent_obs_df")

    comb['site_key'] = comb.get('site', "").astype(str).str.strip().str.lower()
    rec['site_key']  = rec.get('site', "").astype(str).str.strip().str.lower()

    comb['datetime_utc'] = pd.to_datetime(comb['datetime'], utc=True,  errors='coerce')
    rec['obs_date_utc']  = pd.to_datetime(rec['obs_date_utc'], utc=True, errors='coerce')

    comb['ra']  = _coerce_ra_deg_series(comb['ra'])
    comb['dec'] = pd.to_numeric(comb['dec'], errors='coerce')

    if 'obj_name' in comb.columns:
        obj_col = 'obj_name'
    elif 'name' in comb.columns:
        obj_col = 'name'
    else:
        obj_col = None

    comb = comb.dropna(subset=['datetime_utc','site_key','ra','dec'])
    rec  = rec.dropna(subset=['obs_date_utc','site_key','astrom_footprint'])

    comb['site_orig'] = combined_df['site'] if 'site' in combined_df.columns else comb['site_key']
    rec['site_orig']  = recent_obs_df['site'] if 'site' in recent_obs_df.columns else rec['site_key']

    # parse footprints
    normals_pack = []
    valid_idx = []
    for i, v in enumerate(_tqdm(rec['astrom_footprint'].tolist(),
                                desc="Footprints→normals", unit="fp")):
        poly = safe_parse_footprint(v)
        if poly is None:
            continue
        V, E = _poly_to_unit_vertices(poly)
        if V is None:
            continue
        n = _edge_inward_normals(V, E)
        normals_pack.append(n)
        valid_idx.append(i)

    if len(valid_idx) == 0 or comb.empty:
        print("[END] Nothing to match after filtering.")
        return pd.DataFrame(columns=[
            'name','RA','DEC','site','obs_time','telescope','filepath','limiting_mag',
            'Vmag','unc','d_total','ra_rate','dec_rate','time_diff_s','id'
        ])

    rec = rec.iloc[valid_idx].copy()
    rec['normals_idx'] = np.arange(len(valid_idx))

    # precompute ephemeris unit vectors
    comb['eph_vec'] = list(_radec_deg_to_unit(comb['ra'].values, comb['dec'].values))

    meta_cols = [c for c in ['telescope','limiting_mag','filepath','id','obs_date'] if c in rec.columns]
    rec['__obs_idx__'] = np.arange(len(rec))
    meta_map = rec.set_index('__obs_idx__')[meta_cols].to_dict('index') if meta_cols else {}

    out_rows = []

    # per-site
    site_iters = list(rec.groupby('site_key', sort=False))
    for site_key, grp_obs in _tqdm(site_iters, desc="Sites", unit="site"):
        eph_site = comb[comb['site_key'] == site_key]
        if eph_site.empty or grp_obs.empty:
            continue

        groups = list(eph_site.groupby(obj_col, sort=False)) if obj_col else [(None, eph_site)]

        T_obs = grp_obs['obs_date_utc'].to_numpy(dtype='datetime64[ns]')
        T_obs_ns = T_obs.astype('datetime64[ns]').astype('int64')
        tol_ns = int(float(time_tolerance_sec) * 1e9)
        obs_rows = grp_obs.reset_index(drop=True)

        # per-object
        for obj_name, g in _tqdm(groups, desc=f"Objs@{site_key}", unit="obj", leave=False):
            g_sorted = g.sort_values('datetime_utc').reset_index(drop=False)
            if g_sorted.empty:
                continue
            t_eph_ns = g_sorted['datetime_utc'].to_numpy(dtype='datetime64[ns]').astype('int64')

            pos = np.searchsorted(t_eph_ns, T_obs_ns, side='left')
            pos0 = np.clip(pos - 1, 0, len(t_eph_ns)-1)
            pos1 = np.clip(pos,     0, len(t_eph_ns)-1)
            left_dt  = np.abs(T_obs_ns - t_eph_ns[pos0])
            right_dt = np.abs(T_obs_ns - t_eph_ns[pos1])
            nearest_idx_in_group = np.where(right_dt < left_dt, pos1, pos0)

            cand_rows = g_sorted.iloc[nearest_idx_in_group].reset_index(drop=True)

            dt_ns = np.abs(T_obs_ns - t_eph_ns[nearest_idx_in_group])
            time_ok = dt_ns <= tol_ns
            if not np.any(time_ok):
                continue

            ok_obs_idx = np.nonzero(time_ok)[0]
            cand_ok = cand_rows.iloc[ok_obs_idx].reset_index(drop=True)
            eph_vecs = np.stack(cand_ok['eph_vec'].to_list(), axis=0)
            dt_ns_ok = dt_ns[ok_obs_idx]

            for j_local, j_obs in enumerate(ok_obs_idx):
                normals = normals_pack[int(obs_rows.loc[j_obs, 'normals_idx'])]
                if normals is None or normals.size == 0:
                    continue
                v = eph_vecs[j_local:j_local+1, :]  # 与 cand_ok 第 j_local 行严格对应

                # inside-test
                if not np.all((normals @ v.T) > 0.0):
                    continue

                eph_row = cand_ok.iloc[j_local]
                obs_row = obs_rows.iloc[j_obs]
                meta    = meta_map.get(int(obs_row['__obs_idx__']), {}) if meta_map else {}

                # derive RA/DEC from vector
                vx, vy, vz = float(v[0,0]), float(v[0,1]), float(v[0,2])
                dec_from_v = np.degrees(np.arcsin(np.clip(vz, -1.0, 1.0)))
                ra_from_v  = (np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0

                name     = eph_row.get('obj_name', eph_row.get('name', None))
                vmag     = eph_row.get('vmag',     None)
                unc      = eph_row.get('unc',      None)
                d_total  = eph_row.get('d_total',  None)
                ra_rate  = eph_row.get('ra_rate',  None)
                dec_rate = eph_row.get('dec_rate', None)
                dt_s     = float(dt_ns_ok[j_local] * 1e-9)

                out_rows.append({
                    'name'        : None if pd.isna(name)     else name,
                    'RA'          : float(ra_from_v),
                    'DEC'         : float(dec_from_v),
                    'site'        : eph_row['site_orig'] if 'site_orig' in eph_row else site_key,

                    # FIX: use obs_date_utc (UTC-aware) instead of raw obs_date
                    'obs_time'    : obs_row.get('obs_date_utc', meta.get('obs_date', None)),

                    'telescope'   : meta.get('telescope'),
                    'filepath'    : meta.get('filepath'),
                    'limiting_mag': meta.get('limiting_mag'),
                    'Vmag'        : None if (vmag     is None or pd.isna(vmag))     else float(vmag),
                    'unc'         : None if (unc      is None or pd.isna(unc))      else float(unc),
                    'd_total'     : None if (d_total  is None or pd.isna(d_total))  else float(d_total),
                    'ra_rate'     : None if (ra_rate  is None or pd.isna(ra_rate))  else float(ra_rate),
                    'dec_rate'    : None if (dec_rate is None or pd.isna(dec_rate)) else float(dec_rate),
                    'time_diff_s' : dt_s,
                    'id'          : meta.get('id'),
                })

    return pd.DataFrame(out_rows, columns=[
        'name','RA','DEC','site','obs_time','telescope','filepath','limiting_mag',
        'Vmag','unc','d_total','ra_rate','dec_rate','time_diff_s','id'
    ])



# ========================= Optional enrichment & JPL per-row refine =========================

def _enrich_df_from_ids(df_res, db=None, do_rollback=False):
    """Join context columns (e.g., zp/seeing) from image.single/raw.science by image id."""
    import pandas as _pd

    if 'id' not in df_res.columns:
        return df_res.copy()

    ids = (_pd.to_numeric(df_res['id'], errors='coerce').dropna().astype(int).tolist())
    if len(ids) == 0:
        return df_res.copy()

    close_db = False
    if db is None:
        db = goto_silent()
        close_db = True

    try:
        if do_rollback and hasattr(db, "rollback"):
            try:
                db.rollback()
            except Exception:
                pass

        sql = """
        SELECT
          s.id                              AS image_id,
          s.photom_5sigma_detection_magnitude AS Photom_5sigma_limit,
          s.detect_hfd_mean                 AS hfd,
          s.photom_zeropoint                AS zero_point,
          s.photom_zeropoint_uncert         AS zero_point_unc,
          r.id                              AS raw_id,
          r.airmass, r.satcloud, r.seeing, r.dust,
          r.exptime, r.filt
        FROM "image"."single" AS s
        JOIN "raw"."science" AS r
          ON r.id = s.raw_id
        WHERE s.id = ANY(%(ids)s)
        ORDER BY s.id;
        """
        df_ctx = query_df(db, sql, {"ids": ids})

    finally:
        if close_db:
            try:
                if hasattr(db, "close"):
                    db.close()
            except Exception:
                pass

    df_final = (
        df_res.merge(df_ctx, left_on="id", right_on="image_id", how="left")
              .drop(columns=[c for c in ["image_id"] if c in df_res.columns], errors="ignore")
    )
    return df_final


# ==== JPL refine (per-row, minute precision) ====
import astropy.units as _u
from astropy.time import Time as _APTime


def _to_minute_utc_str(ts):
    """Convert timestamp to 'YYYY-MM-DD HH:MM' UTC string."""
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(t):
        return None
    return t.strftime("%Y-%m-%d %H:%M")


def _norm_site(name: str):
    """Normalize site alias to canonical name and return astropy location mapping."""
    s = str(name).strip()
    s = ALIASES.get(s, s)
    if s not in SITE_PARAMS:
        raise ValueError(f"unknown site: {name!r} → normalized {s!r} not in {list(SITE_PARAMS.keys())}")
    p = SITE_PARAMS[s]
    return s, {"lon": p["lon"] * _u.deg, "lat": p["lat"] * _u.deg, "elevation": (p["alt"]/1000.0) * _u.km}


def _choose_target_name(row):
    """Best-effort extract a target name from row fields."""
    for key in ["target", "object", "name", "designation", "desig", "mpc_name"]:
        v = row.get(key) if hasattr(row, "get") else row[key] if key in row.index else None
        if v and str(v).strip():
            return str(v).strip()
    return None


def _fetch_one_horizons(site_name: str, time_minute_iso: str, target_name: str, max_retries: int = 2, sleep_s: float = 0.8):
    """Query JPL Horizons for a single (site, time, target) triplet."""
    from astroquery.jplhorizons import Horizons
    import time as _time

    site_key, loc = _norm_site(site_name)
    t_jd = float(_APTime(time_minute_iso, format="iso", scale="utc").jd)
    epochs_jd = [t_jd]

    tries = []
    if target_name:
        tries.append((target_name, None))
        tries.append((f"DES={target_name}", None))

    last_err = None
    for id_val, id_type in tries:
        for attempt in range(1, max_retries+1):
            try:
                obj = Horizons(id=id_val, id_type=id_type, location=loc, epochs=epochs_jd)
                tab = obj.ephemerides()
                df = tab.to_pandas()
                return dict(
                    RA=float(df.iloc[0]["RA"]),
                    DEC=float(df.iloc[0].get("DEC", df.iloc[0]["DEC"])),
                    RA_rate=float(df.iloc[0].get("RA_rate", float("nan"))),
                    DEC_rate=float(df.iloc[0].get("DEC_rate", float("nan"))),
                )
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    _time.sleep(sleep_s)
    if last_err:
        raise last_err
    raise RuntimeError("Horizons unknown failure")


def refine_ephemerides_per_row(df, site_col="site", time_col="obs_time", target_col=None, print_limit=10):
    """Optionally refine ephemerides per matched row using JPL Horizons at minute precision."""
    df_out = df.copy()
    for c in ["RA_pre","DEC_pre","RA_rate_pre","DEC_rate_pre"]:
        if c not in df_out.columns:
            df_out[c] = np.nan

    cache = {}
    ok, fail, printed = 0, 0, 0

    it = tqdm(df_out.iterrows(), total=len(df_out), desc="JPL per-row", disable=False)
    for idx, row in it:
        site_val = row.get(site_col) if hasattr(row, "get") else row[site_col] if site_col in df_out.columns else None
        t_min = _to_minute_utc_str(row.get(time_col) if hasattr(row, "get") else row[time_col] if time_col in df_out.columns else None)
        if not site_val or not t_min:
            if printed < print_limit:
                print(f"[warn] skip: site={site_val!r}, time={row.get(time_col) if hasattr(row,'get') else None!r}")
                printed += 1
            fail += 1
            continue

        tgt = None
        if target_col and target_col in df_out.columns:
            tgt = str(row[target_col]).strip() if row[target_col] is not None else None
        if not tgt:
            tgt = _choose_target_name(row)

        site_norm = ALIASES.get(str(site_val).strip(), str(site_val).strip())
        key = (site_norm, t_min, tgt or "")

        if key not in cache:
            try:
                cache[key] = _fetch_one_horizons(site_norm, t_min, tgt)
            except Exception as e:
                cache[key] = None
                if printed < print_limit:
                    print(f"[fail] site={site_norm}, t={t_min}, target={tgt!r}, err={repr(e)}")
                    printed += 1

        res = cache[key]
        if res is None:
            fail += 1
            continue

        df_out.at[idx, "RA_pre"]       = res["RA"]
        df_out.at[idx, "DEC_pre"]      = res["DEC"]
        df_out.at[idx, "RA_rate_pre"]  = res["RA_rate"]
        df_out.at[idx, "DEC_rate_pre"] = res["DEC_rate"]
        ok += 1

    print(f"[JPL] {ok} row(s) succeeded, {fail} row(s) failed; unique requests={len(cache)}")
    return df_out


# ========================= CLI / Main =========================

def _parse_targets_arg(targets_str: str) -> list:
    """Split a user-provided targets string (comma/semicolon/newline separated)."""
    if not targets_str:
        return []
    parts = re.split(r'[,;，；\n]+', targets_str.strip())
    return [p.strip() for p in parts if p.strip()]


def _read_targets_file(path: str) -> list:
    """Read a plain text file of targets (one per line or comma/space separated)."""
    paths = [p for p in [path] if p]
    if not paths:
        return []
    p = paths[0]
    with open(p, 'r', encoding='utf-8') as f:
        text = f.read()
    return _parse_targets_arg(text)


def _select_sites(site_flag: str):
    """Return list of (site_name, lat, lon, alt_m)."""
    site_flag = (site_flag or 'BOTH').strip().upper()
    if site_flag == 'LP':
        return [('La Palma', 28.76012, -17.87929, 2348)]
    if site_flag == 'SSO':
        return [('Siding Spring', -31.2734, 149.06411, 1137)]
    return [
        ('La Palma', 28.76012, -17.87929, 2348),
        ('Siding Spring', -31.2734, 149.06411, 1137),
    ]


def _plan_sampling(days: int, interval_min: int, max_points: int = 1441):
    """Return (step_str, step_count) for MPC so that number ≤ max_points.
    Prefer minutes; if exceeding the cap, switch to hour-based step with enlarged step_hours.
    """
    total_minutes = days * 24 * 60
    steps_min = int(math.ceil(total_minutes / max(1, interval_min))) + 1
    if steps_min <= max_points:
        return f"{interval_min} minute", steps_min
    # fallback to hours
    total_hours = days * 24
    step_hours = max(1, int(math.ceil(total_hours / (max_points - 1))))
    steps_hr = int(math.ceil(total_hours / step_hours)) + 1
    return f"{step_hours} hour", steps_hr


def main():
    parser = argparse.ArgumentParser(
        description="Target-based GOTO matcher (user-provided targets; no MPC list scrape)")

    # --- user inputs ---
    parser.add_argument('--targets', type=str, default='',
                    help='Multiple targets separated by comma/semicolon/newline, e.g. "2025 TF, 2010 CG121; 3I/ATLAS"')
    parser.add_argument('--targets-file', type=str, default='',
                        help='Plain text of target names (one per line or comma/space separated)')
    parser.add_argument('--days', type=int, default=3, help='Lookback days (used for observation window & ephemeris span)')

    # --- ephemeris controls ---
    parser.add_argument('--interval', type=int, default=36, help='Base sampling step (minutes). If resulting #points > 1441, switch to hourly sampling so total ≤ 1441, then densify to 1-min by interpolation.')
    parser.add_argument('--start-utc', type=str, default=None,
                        help='Ephemeris start time in UTC (YYYY-MM-DD or full ISO). Default = now - days.')
    parser.add_argument('--site', type=str, default='BOTH', choices=SITE_CHOICES,
                        help='Site selection: BOTH / LP / SSO')

    # --- matching controls ---
    parser.add_argument('--time-tol', type=int, default=60, help='Time tolerance (seconds)')

    # --- enrichment / refine ---
    parser.add_argument('--enrich', action='store_true', default=False,
                        help='Join context fields from image.single/raw.science (zp, seeing, etc.)')
    parser.add_argument('--jpl-refine', action='store_true', default=True,
                        help='Per-row JPL Horizons refinement (minute precision)')

    # --- output ---
    today_str = datetime.utcnow().strftime('%Y-%m-%d')
    parser.add_argument('--outfile', type=str, default=os.path.join('data', f'{today_str}_target_search.csv'))

    args = parser.parse_args()

    # --- gather targets ---
    tgt_list = []
    tgt_list += _parse_targets_arg(args.targets)
    if args.targets_file:
        tgt_list += _read_targets_file(args.targets_file)

    if not tgt_list:
        # interactive fallback (press Enter to skip)
        try:
            user_in = input("Enter targets (comma/space separated, Enter to finish): ").strip()
        except EOFError:
            user_in = ''
        tgt_list = _parse_targets_arg(user_in)

    if not tgt_list:
        print("[ERROR] No targets provided. Use --targets or --targets-file.")
        return

    print(f"[INFO] Targets: {len(tgt_list)} → {tgt_list}")

    # --- derive ephemeris span from days & interval ---
    days = max(1, int(args.days))
    interval_min = max(1, int(args.interval))
    step_str, steps = _plan_sampling(days, interval_min, max_points=1441)
    npoints = days * 24 * 60 + 1   # per-minute dense grid
    print(f"[EPH] Sampling plan → step='{step_str}', number={steps} (days={days}, base_interval_min={interval_min})")

    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d") if not args.start_utc else args.start_utc

    # --- build ephemerides per site ---
    combined_df = pd.DataFrame(columns=[
        "datetime","RA","DEC","ALT","Sun_Alt","obj_name",
        "Vmag","unc","d_total","ra_rate","dec_rate","site"
    ])

    for (site_name, lat, lon, h) in tqdm(_select_sites(args.site), desc="Sites(ephem)", unit="site"):
        temp = batch_fetch_ephemerides(
            obj_list=tgt_list,
            start_date=start_date,
            step_str=step_str,
            step_count=steps,
            site_lon=lon, site_lat=lat, site_height=h,
            n_points=npoints
        )
        temp["site"] = site_name
        combined_df = pd.concat([combined_df, temp], ignore_index=True)

    print(f"[INFO] Ephemerides rows: {len(combined_df)}")

    # --- load observations ---
    recent_obs_df = load_observations_from_image_single(lookback_days=days)
    print(f"[INFO] Observations rows: {len(recent_obs_df)}")

    # --- match ---
    matched_df = match_ephemeris_by_site(combined_df, recent_obs_df, time_tolerance_sec=args.time_tol)

    # --- optional enrich ---
    if args.enrich:
        try:
            matched_df = _enrich_df_from_ids(matched_df)
        except Exception as e:
            print("[WARN] enrich failed:", repr(e))

    # --- optional JPL refine ---
    if args.jpl_refine and not matched_df.empty:
        try:
            matched_df = refine_ephemerides_per_row(matched_df, site_col="site", time_col="obs_time", target_col="name")
        except Exception as e:
            print("[WARN] JPL refine failed:", repr(e))

    # --- final dedupe name + obs_time ---
    print(f"[INFO] Before final deduplication: {len(matched_df)} rows")
    if 'name' in matched_df.columns and 'obs_time' in matched_df.columns:
        matched_df = matched_df.drop_duplicates(subset=['name', 'obs_time'], keep='first')
        print(f"[INFO] After deduplication by (name, obs_time): {len(matched_df)} rows")

    # --- write ---
    os.makedirs(os.path.dirname(args.outfile) or '.', exist_ok=True)
    matched_df.to_csv(args.outfile, index=False)

    # --- brief summary ---
    if not matched_df.empty:
        try:
            summary = matched_df.groupby(['name','site']).size().rename('hits').reset_index().sort_values(['name','site'])
            print("\n[SUMMARY] hits per target × site:\n", summary.to_string(index=False))
        except Exception:
            pass
    print(f"\nWrote: {args.outfile} (rows={len(matched_df)})")

    def angsep(ra1, de1, ra2, de2):
        r1, d1 = np.radians(ra1), np.radians(de1)
        r2, d2 = np.radians(ra2), np.radians(de2)
        cosd = np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1-r2)
        return np.degrees(np.arccos(np.clip(cosd, -1, 1))) * 3600.0  # arcsec

if __name__ == "__main__":
    main()
