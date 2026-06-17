#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-process detection tracks to recover original pixel/WCS coordinates
and (optionally) run pill photometry per-frame.

This utility reads the ``group_manifest.csv`` produced by ``daily_workflow.py``
and, for each subgroup where the detector reported a positive detection, it
performs the following steps:

1. Load the per-frame track CSV saved by ``detect_mover_gif.py``.  The track
   contains pixel coordinates measured on PNG thumbnails that were mirrored
   along the X axis and then rotated 90° counter-clockwise before being served
   by the remote thumbnail generator.
2. Undo the geometric transformations (using the per-row sky center ra/dec and
   window size_px) to recover the pixel coordinates in the original FITS reference frame.
3. Query the remote WCS solution for every FITS image participating in the
   subgroup via :mod:`wcs_reader_client` and convert the recovered pixel
   coordinates to equatorial sky coordinates (RA, Dec).
4. Store the results back into the manifest in two JSON columns:

   ``detector_track_pixels_orig``
       List of dictionaries containing the original (un-flipped/un-rotated)
       pixel coordinates for each frame together with the frame index and the
       contributing FITS path.

   ``detector_track_radec``
       List of dictionaries with the corresponding sky coordinates
       (``ra_deg``/``dec_deg``).

5. (Optional) If ``--photom-out`` is specified, for each per-frame (ra,dec,path)
   run pill photometry:
     - fetch zero point from pipeline at (ra,dec) for that image
     - download a FITS cutout via SSH
     - run run_pill_photometry()
     - append results to a CSV with columns:
         target_name, obs_time, ra, dec, snr, mag, mag_err, path, frame

The updated manifest overwrites the original CSV file unless ``--dry-run`` is
specified.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd

from pathlib import Path
import wcs_reader_client
from astroquery.jplhorizons import Horizons
from astropy.time import Time

# --- optional photometry deps (import inside when used) ---
# from run_client_fits_cutouts_batch import FitsCutoutClient
# from pill_matched_phot import run_pill_photometry_4images
# from zpquery import fetch_photometry
# from astropy.io import fits


# ----------------------------- Data containers ------------------------------

@dataclass
class TrackEntry:
    """Container for per-frame detection information."""
    frame: int
    path: str
    x_proc: float  # PNG 上的 x（已翻转+旋转之后）
    y_proc: float  # PNG 上的 y（已翻转+旋转之后）


@dataclass
class TrackSolution:
    """Recovered pixel and sky coordinates for a single frame."""
    frame: int
    path: str
    x_pix: float   # 原始 FITS 像素坐标
    y_pix: float
    ra_deg: float
    dec_deg: float


# ---------------------------- Helper functions -----------------------------
def _run_date_str() -> str:
    """返回脚本运行当天（本机时区）的 YYYY-MM-DD。"""
    return date.today().strftime("%Y-%m-%d")

def autofill_detector_track_csv(m: pd.DataFrame, base_dir: Path, verbose: bool = False) -> tuple[int, pd.DataFrame]:
    """
    回填：对 detected==True 且 detector_track_csv 为空的行，
    生成 <base_dir>/<site>__<object>__sub<sub_id:02d>/<object>__<site>__sub<sub_id:02d>_track.csv
    若文件存在则写入该路径，否则跳过。返回 (updated_count, df)。
    """
    if "detector_track_csv" not in m.columns:
        m["detector_track_csv"] = ""

    def _norm(x):  # 仅把空格换下划线，保持既有命名风格
        return str(x).strip().replace(" ", "_")

    updated = 0
    for i, r in m.iterrows():
        det = r.get("detected", False)
        if isinstance(det, str):
            det = det.strip().lower() == "true"
        if not det:
            continue
        cur = r.get("detector_track_csv")
        if pd.notna(cur) and str(cur).strip():
            continue

        obj = _norm(r.get("object", ""))
        site = _norm(r.get("site", ""))
        try:
            sub_id = int(r.get("sub_id"))
        except Exception:
            continue
        if not obj or not site:
            continue

        group_dir = base_dir / f"{site}__{obj}__sub{sub_id:02d}"
        track = group_dir / f"{obj}__{site}__sub{sub_id:02d}_track_snap.csv"
        if track.exists():
            m.at[i, "detector_track_csv"] = str(track.resolve())
            updated += 1

    if verbose:
        print(f"[INFO] autofill: updated {updated} row(s) for detector_track_csv")
    return updated, m


def _parse_paths(raw: str) -> List[str]:
    """Return a cleaned list of FITS paths encoded in the manifest."""
    if not isinstance(raw, str):
        return []
    return [p.strip() for p in raw.split(";") if p and p.strip()]


def _load_track_csv(path: str) -> pd.DataFrame:
    """Load a track CSV and ensure required columns are present."""
    df = pd.read_csv(path)
    expected = {"frame", "x_snap", "y_snap"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Track CSV {path!r} missing columns: {sorted(missing)}")
    df = df[list(sorted(expected))].copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["x_snap"] = pd.to_numeric(df["x_snap"], errors="coerce")
    df["y_snap"] = pd.to_numeric(df["y_snap"], errors="coerce")
    df = df.dropna(subset=["frame", "x_snap", "y_snap"])
    df["frame"] = df["frame"].astype(int)
    return df.sort_values("frame").reset_index(drop=True)


def _chunked(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    """Yield ``size`` sized chunks from *seq*."""
    if size <= 0:
        yield seq
        return
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _fetch_wcs_bulk(paths: Sequence[str], *, server: str, hdu: int, remote_script: str, chunk_size: int, verbose: bool) -> Dict[str, dict]:
    """Fetch WCS information for all ``paths`` and return a lookup dictionary."""
    out: Dict[str, dict] = {}
    errors: List[Tuple[str, str]] = []
    for chunk in _chunked(list(paths), chunk_size):
        info_list = wcs_reader_client.fetch_wcs(
            list(chunk),
            server=server,
            hdu=hdu,
            remote_script=remote_script,
            verbose=verbose,
        )
        for item in info_list:
            path = item.get("path")
            if not path:
                continue
            if "error" in item:
                errors.append((path, str(item["error"])) )
            else:
                out[path] = item
    if errors:
        problems = "; ".join(f"{p}: {msg}" for p, msg in errors)
        raise RuntimeError(f"Failed to fetch WCS for some paths: {problems}")
    return out


def _build_track_entries(track_df: pd.DataFrame, paths: Sequence[str]) -> List[TrackEntry]:
    """Match track rows to FITS paths by order (frame i -> paths[i])."""
    entries: List[TrackEntry] = []
    usable = min(len(track_df), len(paths))
    for i in range(usable):
        row = track_df.iloc[i]
        entries.append(
            TrackEntry(
                frame=int(row["frame"]),
                path=paths[i],  # 注意：按照顺序配对
                x_proc=float(row["x_snap"]),
                y_proc=float(row["y_snap"]),
            )
        )
    return entries


# ---------------------- Geometry inverse: PNG -> FITS -----------------------

def _invert_png_to_fits_pixel(
    *,
    u: float, v: float,
    wcs_obj,
    ra_center_deg: float,
    dec_center_deg: float,
    size_px: int,
    origin: int = 0,
    flip_mode: str = "vertical",
) -> Tuple[float, float]:
    """
    把 PNG 坐标 (u,v) 反演为原始 FITS 像素坐标 (x_fit,y_fit)。

    反演步骤：
      1) 用 WCS 计算窗口中心在 FITS 像素中的位置 (x_c, y_c)，并据此得到窗口左上角 (x0, y0)。
      2) 逆旋转：对 (u,v) 先做顺时针 90°，再做 vertical 翻转的逆。
         - 若 flip_mode="vertical"（上下翻）：X = S-1 - v,  Y = S-1 - u
         - 若 flip_mode="horizontal"（左右翻）：X = v,      Y = u
      3) 原始像素：x_fit = x0 + X, y_fit = y0 + Y
    """
    # 窗口中心在像素坐标的连续位置
    x_c, y_c = wcs_obj.all_world2pix(ra_center_deg, dec_center_deg, origin)
    half = (float(size_px) - 1.0) / 2.0
    x0 = x_c - half  # 不取整，保持连续坐标
    y0 = y_c - half

    if flip_mode == "vertical":
        X = float(size_px - 1) - float(v)
        Y = float(size_px - 1) - float(u)
    elif flip_mode == "horizontal":
        X = float(v)
        Y = float(u)
    else:
        raise ValueError("flip_mode must be 'vertical' or 'horizontal'")

    x_fit = x0 + X
    y_fit = y0 + Y
    return x_fit, y_fit


def _solve_tracks_with_center(
    entries: Sequence[TrackEntry],
    wcs_lookup: Dict[str, dict],
    *,
    ra_center_deg: float,
    dec_center_deg: float,
    size_px: int,
    origin: int = 0,
    flip_mode: str = "vertical",
) -> List[TrackSolution]:
    """使用每帧的 WCS + (ra,dec,size_px) 把 PNG 像素反演为 FITS 像素，再转成天球坐标。"""
    solutions: List[TrackSolution] = []
    for ent in entries:
        meta = wcs_lookup.get(ent.path)
        if meta is None:
            raise KeyError(f"No WCS info for path {ent.path!r}")
        wcs_obj = meta.get("wcs")
        if wcs_obj is None:
            raise ValueError(f"Incomplete WCS metadata for {ent.path!r}")

        # PNG (u,v) -> 原 FITS 像素
        x_fit, y_fit = _invert_png_to_fits_pixel(
            u=ent.x_proc, v=ent.y_proc,
            wcs_obj=wcs_obj,
            ra_center_deg=ra_center_deg,
            dec_center_deg=dec_center_deg,
            size_px=size_px,
            origin=origin,
            flip_mode=flip_mode,
        )

        # FITS 像素 -> (RA,Dec)
        ra_deg, dec_deg = wcs_obj.all_pix2world(x_fit, y_fit, origin)

        solutions.append(
            TrackSolution(
                frame=ent.frame,
                path=ent.path,
                x_pix=float(x_fit),
                y_pix=float(y_fit),
                ra_deg=float(ra_deg),
                dec_deg=float(dec_deg),
            )
        )
    return solutions


def _serialise_solutions(solutions: Sequence[TrackSolution]) -> Tuple[str, str]:
    """Serialise solutions into JSON strings for the manifest."""
    pix_payload = [
        {"frame": s.frame, "path": s.path, "x_snap": s.x_pix, "y_snap": s.y_pix}
        for s in solutions
    ]
    sky_payload = [
        {"frame": s.frame, "path": s.path, "ra_deg": s.ra_deg, "dec_deg": s.dec_deg}
        for s in solutions
    ]
    return json.dumps(pix_payload, ensure_ascii=False), json.dumps(sky_payload, ensure_ascii=False)


# ----------------------------- Main workflow -------------------------------

def process_manifest(
    manifest_path: str,
    *,
    server: str,
    remote_script: str,
    hdu: int,
    chunk_size: int,
    verbose: bool,
) -> pd.DataFrame:
    """Load, enrich, and return the updated manifest DataFrame."""
    df = pd.read_csv(manifest_path)

    if "paths" not in df.columns:
        raise KeyError("Manifest is missing the 'paths' column.")
    for col in ("detector_track_pixels_orig", "detector_track_radec"):
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), dtype="object")
        else:
            df[col] = df[col].astype("object")

    rows_to_process: List[Tuple[int, List[str], pd.DataFrame, float, float, int]] = []
    unique_paths: List[str] = []

    for idx, row in df.iterrows():
        # 资格判断
        detected_raw = row.get("detected")
        if pd.isna(detected_raw):
            detected = False
        elif isinstance(detected_raw, str):
            detected = detected_raw.strip().lower() == "true"
        else:
            detected = bool(detected_raw)
        track_csv = str(row.get("detector_track_csv", "")).strip()
        if not (detected and track_csv and track_csv.upper() != "NA" and osp.isfile(track_csv)):
            continue

        # 必要字段：paths、ra/dec、size_px
        paths = _parse_paths(row.get("paths", ""))
        if not paths:
            if verbose:
                print(f"[WARN] Row {idx}: no FITS paths listed; skipping.")
            continue
        try:
            ra0 = float(row["ra_deg"]); dec0 = float(row["dec_deg"]); sz = int(row["size_px"])
        except Exception as exc:
            if verbose:
                print(f"[WARN] Row {idx}: missing/invalid ra_deg/dec_deg/size_px: {exc}; skipping.")
            continue

        # 轨迹
        try:
            track_df = _load_track_csv(track_csv)
        except Exception as exc:
            if verbose:
                print(f"[WARN] Row {idx}: failed to load track CSV {track_csv!r}: {exc}")
            continue
        if track_df.empty:
            if verbose:
                print(f"[WARN] Row {idx}: track CSV {track_csv!r} has no valid rows; skipping.")
            continue

        usable_paths = paths[: len(track_df)]
        if len(track_df) != len(paths) and verbose:
            print(
                f"[WARN] Row {idx}: frame count ({len(track_df)}) != number of paths ({len(paths)}); "
                f"using first {len(usable_paths)} path(s)."
            )
        rows_to_process.append((idx, usable_paths, track_df, ra0, dec0, sz))
        unique_paths.extend(usable_paths)

    if not rows_to_process:
        if verbose:
            print("[INFO] No rows required processing.")
        return df

    # 去重（保持顺序）
    seen = set()
    ordered_unique_paths: List[str] = []
    for p in unique_paths:
        if p not in seen:
            ordered_unique_paths.append(p)
            seen.add(p)

    if verbose:
        print(f"[INFO] Fetching WCS for {len(ordered_unique_paths)} unique image(s)...")

    # 拉取每张图的 WCS
    wcs_lookup = _fetch_wcs_bulk(
        ordered_unique_paths,
        server=server,
        hdu=hdu,
        remote_script=remote_script,
        chunk_size=chunk_size,
        verbose=verbose,
    )

    # 逐行求解
    for idx, paths, track_df, ra0, dec0, size_px in rows_to_process:
        entries = _build_track_entries(track_df, paths)
        try:
            solutions = _solve_tracks_with_center(
                entries,
                wcs_lookup,
                ra_center_deg=ra0,
                dec_center_deg=dec0,
                size_px=size_px,
                origin=0,              # 你的像素坐标体系按 0 基
                flip_mode="vertical",  # “x 轴翻转”默认解读为上下翻转（vertical）
            )
        except Exception as exc:
            if verbose:
                print(f"[WARN] Row {idx}: failed to resolve WCS coordinates: {exc}")
            continue

        pix_json, sky_json = _serialise_solutions(solutions)
        df.at[idx, "detector_track_pixels_orig"] = pix_json
        df.at[idx, "detector_track_radec"] = sky_json

    return df


def _default_manifest_dir() -> str:
    """Return the default manifest directory for today's workflow run."""
    today = date.today().strftime("%Y-%m-%d")
    return osp.join(".", "thumbs", f"{today}_MPC")


# --------------------------- Photometry workflow ----------------------------

def _json_rows(s: str) -> List[dict]:
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass
    return []


def _scalar(v) -> Optional[float]:
    """Safely coerce pandas scalar/Series to python float or None."""
    try:
        if v is None or (hasattr(v, "size") and v.size == 0):
            return None
        if hasattr(v, "iloc"):
            return float(v.iloc[0])
        return float(v)
    except Exception:
        return None


def _fits_date_obs(local_fits_path: str) -> Optional[str]:
    """Read DATE-OBS (or fallback) from local FITS cutout."""
    try:
        from astropy.io import fits
        with fits.open(local_fits_path) as hdul:
            for ext in (0, 1):
                if ext < len(hdul):
                    hdr = hdul[ext].header
                    for k in ("DATE-MID", "TIME-OBS"):
                        if k in hdr and hdr[k]:
                            return str(hdr[k])
    except Exception:
        return None
    return None


def _query_jpl_vmag(target_name: str, obs_time: str, verbose: bool = False) -> Optional[float]:
    """
    Query JPL Horizons for V magnitude of the target at the given observation time.
    
    Parameters
    ----------
    target_name : str
        Target designation (e.g., "2025 VJ3")
    obs_time : str
        Observation time in ISO format (e.g., "2025-11-15T01:42:16.796000+00:00")
    verbose : bool
        Print debug information
        
    Returns
    -------
    Optional[float]
        V magnitude if successful, None otherwise
    """
    try:
        # Parse observation time - handle various ISO formats including timezone
        if isinstance(obs_time, str):
            # Remove timezone info if present for astropy Time parsing
            # astropy Time can handle it but we standardize to UTC
            obs_time_clean = obs_time.replace('+00:00', '').replace('Z', '')
            t = Time(obs_time_clean, format='isot', scale='utc')
        else:
            t = Time(obs_time)
        
        if verbose:
            print(f"[INFO] Querying JPL for {target_name} at JD={t.jd:.6f} ({obs_time})")
        
        # Query JPL Horizons
        # Use geocentric location (500@399 = geocenter)
        obj = Horizons(id=target_name, location='500@399', epochs=t.jd)
        eph = obj.ephemerides()
        
        # Extract V magnitude
        if 'V' in eph.colnames:
            vmag = float(eph['V'][0])
            if verbose:
                print(f"[INFO] JPL Vmag for {target_name} at {obs_time}: {vmag:.2f}")
            return vmag
        else:
            if verbose:
                print(f"[WARN] No V magnitude in JPL response for {target_name}")
            return None
            
    except Exception as e:
        if verbose:
            print(f"[WARN] JPL query failed for {target_name}: {e}")
        return None


def run_photometry_from_manifest(
    manifest_path: str,
    *,
    server: str,
    photom_out: str,
    cutout_size: int,
    verbose: bool,
    use_template: bool = True,
    phot_half_size: int = 75,
) -> None:
    """
    Batch photometry with pre-qualification checks:
    1. Skip if detector_L_fit_px > 4.5 (slow_mover)
    2. Skip if JPL Vmag > 18.1 (faint_fastmover)
    3. Otherwise run pill_photometry_4images on all 4 frames together
    
    对每个subgroup（4帧图片）进行资格检测后，使用批量测光。
    
    Parameters
    ----------
    manifest_path : str
        Path to group_manifest.csv
    server : str
        Server alias for cutout download
    photom_out : str
        Output CSV file path
    cutout_size : int
        Cutout size in pixels
    verbose : bool
        Verbose logging
    use_template : bool
        Whether to use template subtraction in photometry
    phot_half_size : int
        Half size for pill photometry
    """
    from run_client_fits_cutouts_batch import FitsCutoutClient
    from pill_matched_phot import run_pill_photometry_4images
    from zpquery import fetch_photometry

    df = pd.read_csv(manifest_path)
    rows: List[dict] = []

    # 目标 PNG 根目录：cutouts/<运行当天>
    date_str = _run_date_str()
    base_dir = Path("cutouts") / date_str
    base_dir.mkdir(parents=True, exist_ok=True)
	
    for idx, row in df.iterrows():
        sky_json = row.get("detector_track_radec")
        
        # 读取检测参数
        L = float(row.get("detector_L_fit_px", 0))
        Angle_target = -float(row.get("detector_angle_deg", 0)) + 270
        W = float(row.get("detector_w_fit_px", 0))
        
        if pd.isna(sky_json) or not str(sky_json).strip():
            continue
        targets = _json_rows(sky_json)
        if not targets:
            continue
        if len(targets) != 4:
            if verbose:
                print(f"[WARN] Subgroup {idx} has {len(targets)} frames, expected 4. Skipping.")
            continue

        target_name = str(row.get("object", "")).strip() or "unknown"
        target_tag = target_name.replace(" ", "_")
        
        # ============ 资格检测 1: detector_L_fit_px < 4.5 ============
        if L < 4.5:
            if verbose:
                print(f"[INFO] {target_name} sub{idx}: L={L:.2f} < 4.5, marked as slow_mover")
            # 直接输出ra,dec，不测光
            for item in targets:
                try:
                    frame = int(item["frame"])
                    path = str(item["path"])
                    ra = float(item["ra_deg"])
                    dec = float(item["dec_deg"])
                    obs_time = row.get("t_start", "")
                    
                    rows.append({
                        "target_name": target_name,
                        "obs_time": obs_time,
                        "ra": ra,
                        "dec": dec,
                        "snr": None,
                        "mag": None,
                        "mag_err": None,
                        "path": path,
                        "frame": frame,
                        "zp_local": None,
                        "zp_medium": None,
                        "fwhm": None,
                        "flag": "slow_mover",
                    })
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Error processing slow_mover entry: {e}")
            continue
        
        # ============ 资格检测 2: JPL Vmag > 18.1 ============
        # 使用第一帧的观测时间查询
        first_obs_time = row.get("t_start", "")
        if not first_obs_time:
            if verbose:
                print(f"[WARN] {target_name} sub{idx}: No observation time available")
            continue
            
        vmag = _query_jpl_vmag(target_name, first_obs_time, verbose=verbose)
        
        if vmag is not None and vmag > 18.5:
            if verbose:
                print(f"[INFO] {target_name} sub{idx}: Vmag={vmag:.2f} > 18.5, marked as faint_fastmover")
            # 直接输出ra,dec，不测光
            for item in targets:
                try:
                    frame = int(item["frame"])
                    path = str(item["path"])
                    ra = float(item["ra_deg"])
                    dec = float(item["dec_deg"])
                    
                    rows.append({
                        "target_name": target_name,
                        "obs_time": first_obs_time,
                        "ra": ra,
                        "dec": dec,
                        "snr": None,
                        "mag": None,
                        "mag_err": None,
                        "path": path,
                        "frame": frame,
                        "zp_local": None,
                        "zp_medium": None,
                        "fwhm": None,
                        "flag": "faint_fastmover",
                        "vmag_jpl": vmag,
                    })
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Error processing faint_fastmover entry: {e}")
            continue
        
        # ============ 通过资格检测，开始批量测光 ============
        if verbose:
            print(f"[INFO] {target_name} sub{idx}: Passed qualification checks, starting photometry")
        
        # 准备4帧的参数列表
        fits_paths = []
        ra_list = []
        dec_list = []
        zp_list = []
        fwhm_list = []
        path_list = []
        
        for item in targets:
            try:
                frame = int(item["frame"])
                path = str(item["path"])
                ra = float(item["ra_deg"])
                dec = float(item["dec_deg"])
                
                # 获取ZP和FWHM
                zp = None
                zp_m = None
                fwhm = None
                try:
                    dfzp = fetch_photometry(path, ra, dec)
                    zp = _scalar(dfzp.get("zp_at_radec"))
                    zp_m = _scalar(dfzp.get("photom_zeropoint"))
                    fwhm = _scalar(dfzp.get("hfd_median_centre"))
                    if fwhm is not None:
                        fwhm = 1 * fwhm
                except Exception as e:
                    if verbose:
                        print(f"[WARN] fetch_photometry failed for {path}: {e}")
                
                ra_list.append(ra)
                dec_list.append(dec)
                zp_list.append(zp)
                fwhm_list.append(fwhm)
                path_list.append(path)
                
            except Exception as e:
                if verbose:
                    print(f"[WARN] Error preparing frame {frame}: {e}")
                continue
        
        # 检查是否收集到4帧的数据
        if len(ra_list) != 4:
            if verbose:
                print(f"[WARN] {target_name} sub{idx}: Only {len(ra_list)} frames prepared, skipping")
            continue

        ra_c = np.mean(ra_list)
        dec_c = np.mean(dec_list)
        # 批量下载4帧的cutouts
        downloaded_local_list = []
        try:
            for i, (ra, dec, path) in enumerate(zip(ra_list, dec_list, path_list)):
                client = FitsCutoutClient(
                    server='warwick',
                    images=[path],
                    ra=ra_c, dec=dec_c,
                    size=int(cutout_size)*1.2,
                    local_temp=str(base_dir),
                    display=False,
                )
                downloaded = client.client() or []
                if downloaded:
                    downloaded_local_list.append(downloaded[0])
                else:
                    if verbose:
                        print(f"[WARN] cutout failed for frame {i}: {path}")
                    downloaded_local_list.append(None)
        except Exception as e:
            if verbose:
                print(
                    f"[WARN] cutout download failed for {target_name} "
                    f"(server={server}, base_dir={base_dir}): {e!r}"
                )
            continue
        
        # 检查是否所有cutouts都成功下载
        if None in downloaded_local_list or len(downloaded_local_list) != 4:
            if verbose:
                print(f"[WARN] {target_name} sub{idx}: Not all cutouts downloaded successfully")
            # 清理已下载的文件
            for fp in downloaded_local_list:
                if fp:
                    try:
                        os.remove(fp)
                    except Exception:
                        pass
            continue
        
        # 准备4-panel输出PNG路径
        fourpanel_outpng = f"pill_{target_tag}_sub{idx:02d}_4panel.png"
        local_dir = base_dir
        
        # ============ 调用batch测光 ============
        try:
            if verbose:
                print(f"[INFO] Running batch photometry for {target_name} sub{idx}")
                               
            results = run_pill_photometry_4images(
                fits_paths=downloaded_local_list,
                ra_list=ra_list,
                dec_list=dec_list,
                half_size=phot_half_size,
                out4_path=str(local_dir / fourpanel_outpng),
                use_template= True,
                debug=True,
                zero_point_list=zp_list,
                fwhm_pix=fwhm_list,  # 如果需要传FWHM，可以添加
            )
            
            # 解析结果并保存
            if results and len(results) == 4:
                for i, (item, result) in enumerate(zip(targets, results)):
                    frame = int(item["frame"])
                    path = path_list[i]

                    snr = _scalar(result.get("snr"))
                    mag = _scalar(result.get("magnitude"))
                    mag_err = _scalar(result.get("magnitude_err"))
                    ra_fin = _scalar(result.get("ra"))
                    dec_fin = _scalar(result.get("dec"))
                    obs_time = result.get("obs_time")
                    
                    rows.append({
                        "target_name": target_name,
                        "obs_time": obs_time if obs_time else first_obs_time,
                        "ra": ra_fin if ra_fin else ra_list[i],
                        "dec": dec_fin if dec_fin else dec_list[i],
                        "snr": snr,
                        "mag": mag,
                        "mag_err": mag_err,
                        "path": path,
                        "frame": frame,
                        "zp_local": zp_list[i],
                        "zp_medium": None,  # 可以从dfzp获取
                        "fwhm": fwhm_list[i],
                        "flag": "measured",
                    })
            else:
                if verbose:
                    print(f"[WARN] Photometry returned unexpected results for {target_name}")
                
        except Exception as e:
            if verbose:
                print(f"[WARN] pill photometry failed for {target_name}: {e}")
        
        finally:
            # 清理本地FITS cutouts
            for fp in downloaded_local_list:
                if fp:
                    try:
                        os.remove(fp)
                    except Exception:
                        pass

    # 保存结果
    if not rows:
        if verbose:
            print("[INFO] No photometry rows produced; skip writing CSV.")
        return

    out_df = pd.DataFrame(rows)
    header = not osp.isfile(photom_out)
    out_df.to_csv(photom_out, mode="a", index=False, header=header)
    if verbose:
        print(f"[INFO] Photometry rows written to: {photom_out} (+={len(out_df)})")

# ---------------------------------- CLI -------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    default_photom_out = Path("thumbs") / f"{date.today():%Y-%m-%d}.csv"
    default_photom_out.parent.mkdir(parents=True, exist_ok=True)
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        help="Path to group_manifest.csv produced by daily_workflow.py (defaults to today's thumbs folder)",
    )
    parser.add_argument(
        "--manifest-dir",
        default=_default_manifest_dir(),
        help="Directory containing group_manifest.csv (default: ./thumbs/<today>_MPC)",
    )
    parser.add_argument("--server", default="warwick", help="Server alias understood by wcs_reader_client (default: warwick)")
    parser.add_argument("--remote-script", default=wcs_reader_client.DEFAULT_REMOTE_SCRIPT, help="Remote read_wcs_server.py path")
    parser.add_argument("--hdu", type=int, default=1, help="HDU index to request WCS from (default: 1)")
    parser.add_argument("--chunk-size", type=int, default=4, help="Number of FITS paths per SSH request (default: 16)")
    parser.add_argument("--dry-run", action="store_true", help="Process but do not overwrite the manifest")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    # photometry options
    parser.add_argument("--photom-out", default=str(default_photom_out), help="If set, run pill photometry and write results to this CSV")
    parser.add_argument("--cutout-size", type=int, default=200, help="Cutout size in pixels for photometry (default: 150)")
    parser.add_argument("--use-template", action="store_true",default = True, help="Use template subtraction in photometry")
    parser.add_argument("--phot-half-size", type=int, default=75, help="Half size for pill photometry (default: 75)")

    args = parser.parse_args(argv)

    if args.manifest:
        manifest_path = osp.abspath(args.manifest)
    else:
        manifest_dir = osp.abspath(args.manifest_dir)
        manifest_path = osp.join(manifest_dir, "group_manifest.csv")

    if not osp.isfile(manifest_path):
        parser.error(f"Manifest file not found: {manifest_path}")

    # 读入 + 结果列 dtype 保护
    m = pd.read_csv(
        manifest_path,
        dtype={"detector_track_pixels_orig": "object", "detector_track_radec": "object"},
    )

    # 自动回填 detector_track_csv（存在就写，不兜底创建）
    updated_count, m = autofill_detector_track_csv(m, Path(manifest_path).parent, verbose=args.verbose)
    if updated_count > 0 and not args.dry_run:
        if args.verbose:
            print(f"[INFO] writing autofill changes to {manifest_path}")
        m.to_csv(manifest_path, index=False)

    # 主流程：PNG 像素 -> FITS 像素 -> RA/Dec
    try:
        df_updated = process_manifest(
            manifest_path,
            server=args.server,
            remote_script=args.remote_script,
            hdu=args.hdu,
            chunk_size=args.chunk_size,
            verbose=args.verbose,
        )
    except Exception as exc:
        parser.error(str(exc))


    if not args.dry_run and df_updated is not None:
        df_updated.to_csv(manifest_path, index=False)
        if args.verbose:
            print(f"[INFO] Manifest updated: {manifest_path}")
    elif args.dry_run and args.verbose:
        print("[INFO] Dry run requested; manifest not written.")

    
    if args.photom_out:
        if args.verbose:
            print(f"[INFO] Running pill photometry → {args.photom_out}")
            
        run_photometry_from_manifest(
            manifest_path,
            server=args.server,
            photom_out=osp.abspath(args.photom_out),
            cutout_size=args.cutout_size,
            verbose=args.verbose,
            use_template=args.use_template,
            phot_half_size=args.phot_half_size,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())