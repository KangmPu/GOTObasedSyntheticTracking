from __future__ import annotations

import os
import pickle
from functools import lru_cache
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import gotodb as gdb

# ----------------------------------------------------------------------
# Global defaults for remote WCS server
# ----------------------------------------------------------------------
DEFAULT_WCS_SERVER = "warwick"
DEFAULT_WCS_REMOTE_SCRIPT = "~/read_wcs_server.py"
DEFAULT_WCS_HDU = 1

# --- Remote WCS integration (remote-only; no local fallback) -----------------
# We optionally import the SSH client that fetches WCS from the server.
try:
    from wcs_reader_client import fetch_wcs as _fetch_wcs_remote  # type: ignore
except Exception:
    _fetch_wcs_remote = None  # type: ignore


def _radec_to_xy_remote(
    filepath: str,
    ra_deg: float,
    dec_deg: float,
    *,
    server: str = DEFAULT_WCS_SERVER,
    remote_script: str = DEFAULT_WCS_REMOTE_SCRIPT,
    hdu: int = DEFAULT_WCS_HDU,
    verbose: bool = False,
):
    """
    Convert (RA, Dec) [deg] -> (x, y) pixels using WCS fetched from the *remote* machine.

    This calls the SSH client (``wcs_reader_client.fetch_wcs``), which runs a small
    script on the server to read the FITS WCS (no image payload) and returns a WCS-only
    header. We rebuild an ``astropy.wcs.WCS`` locally and evaluate world->pixel here.

    Returns
    -------
    (x, y, 'remote', None)  on success
    (nan, nan, 'remote', reason) on failure, where `reason` is a short diagnostic tag.
    """
    if _fetch_wcs_remote is None:
        return float("nan"), float("nan"), "remote", "no_client"

    # Ask the server for WCS (no data transfer)
    try:
        info = _fetch_wcs_remote(
            images=[filepath],
            server=server,
            hdu=hdu,
            remote_script=remote_script,
            verbose=verbose,
        )
    except Exception as e:
        return float("nan"), float("nan"), "remote", f"client_exc:{type(e).__name__}"

    if not info or "wcs" not in info[0]:
        reason = info[0].get("error", "no_wcs") if info else "empty"
        return float("nan"), float("nan"), "remote", str(reason)

    # Evaluate world->pixel using the returned WCS
    try:
        w = info[0]["wcs"]
        x, y = w.world_to_pixel_values(float(ra_deg), float(dec_deg))
        if not (np.isfinite(x) and np.isfinite(y)):
            return float("nan"), float("nan"), "remote", "nonfinite"
        return float(x), float(y), "remote", None
    except Exception as e:
        return float("nan"), float("nan"), "remote", f"w2p_exc:{type(e).__name__}"


# --- DB helpers ---------------------------------------------------------------

# Respect the existing authentication configuration used throughout the repo.
_PGPASS = os.path.expanduser("~/.pgpass")
if os.path.exists(_PGPASS):
    os.environ.setdefault("PGPASSFILE", _PGPASS)


@lru_cache()
def get_db():
    """Return a cached connection to the GOTO database."""
    return gdb.goto()


def query_df(sql: str, params: Optional[Mapping[str, Any]] = None, *, db=None):
    """Execute *sql* with optional *params* and return a DataFrame.

    Parameters
    ----------
    sql:
        The SQL statement to execute. Use ``%(name)s`` placeholders if passing
        ``params``.
    params:
        Optional dictionary of query parameters. Defaults to an empty dict.
    db:
        Optional explicit database connection returned by :func:`get_db`.
    """
    if db is None:
        db = get_db()
    return db.query(sql, params or {})


# --- RBF decoding and ZP evaluation ------------------------------------------

def _join_bytea_like(val):
    """Join a PostgreSQL bytea[]-like container into bytes.

    Supports: bytes/bytearray/memoryview, list/tuple of bytes-like,
    numpy.ndarray (dtype=object or uint8), and any object exposing .tobytes().
    """
    import numpy as _np

    # Direct bytes-like
    if isinstance(val, (bytes, bytearray, memoryview)):
        return bytes(val)

    # numpy arrays: uint8 or object-of-bytes
    if isinstance(val, _np.ndarray):
        try:
            if val.dtype == _np.uint8:
                return bytes(val.tolist())
            # object array: each element should be bytes-like
            return b"".join(bytes(x) for x in val.ravel())
        except Exception:
            pass

    # list/tuple of bytes-like
    if isinstance(val, (list, tuple)):
        try:
            return b"".join(bytes(x) for x in val)
        except Exception:
            pass

    # generic buffer providers
    if hasattr(val, "tobytes"):
        try:
            return val.tobytes()
        except Exception:
            pass

    raise TypeError(f"Unsupported RBF field type: {type(val)}")



def _load_rbf_callable(cell):
    """Unpickle RBF payload into a callable f(x, y).
    Prefer dill if available; fall back to pickle.
    """
    import numpy as _np
    if isinstance(cell, (bytes, bytearray, memoryview)):
        blob = bytes(cell)
    elif isinstance(cell, (list, tuple)):
        blob = b"".join(bytes(x) for x in cell)
    elif isinstance(cell, _np.ndarray):
        blob = bytes(cell.tolist()) if cell.dtype == _np.uint8 else b"".join(bytes(x) for x in cell.ravel())
    elif hasattr(cell, "tobytes"):
        blob = cell.tobytes()
    else:
        return None

    try:
        import dill as _pickle
    except Exception:
        import pickle as _pickle

    try:
        obj = _pickle.loads(blob)
    except Exception:
        return None

    if callable(obj):
        return obj
    if isinstance(obj, dict) and callable(obj.get("rbf")):
        return obj["rbf"]
    return None


def _zp_at_xy_row(row: pd.Series, x: float, y: float) -> float:
    """Compute ZP at pixel (x, y) for a row from image.single join.
    ZP(x,y) = photom_zeropoint + photom_deltazeropoint_rbf(x,y), with optional clipping.
    Returns float('nan') on any failure or missing columns.
    """
    try:
        zp0 = float(row.get("photom_zeropoint"))
    except Exception:
        return float("nan")

    try:
        rbf_cell = row.get("photom_deltazeropoint_rbf")
        rbf = _load_rbf_callable(rbf_cell)
        if rbf is None or not np.isfinite(x) or not np.isfinite(y):
            return float("nan")
        dzp = float(rbf(x, y))
        dzp_min = row.get("photom_deltazeropoint_min")
        dzp_max = row.get("photom_deltazeropoint_max")
        if dzp_min is not None and dzp_max is not None:
            try:
                dzp = float(np.clip(dzp, float(dzp_min), float(dzp_max)))
            except Exception:
                pass
        return zp0 + dzp
    except Exception:
        return float("nan")


def _eval_zp_debug(
    row: pd.Series,
    ra_deg: float,
    dec_deg: float,
    *,
    server: str = DEFAULT_WCS_SERVER,
    remote_script: str = DEFAULT_WCS_REMOTE_SCRIPT,
    hdu: int = DEFAULT_WCS_HDU,
    verbose: bool = False,
) -> pd.Series:
    """Return a Series with zp_at_radec and debug columns for one row."""
    reason = ""
    # photom_zeropoint
    try:
        zp0 = float(row.get("photom_zeropoint"))
        has_zp0 = True
    except Exception:
        zp0 = float("nan")
        has_zp0 = False
        reason = "no_zp0"

    # WCS projection (remote-only)
    x, y, wcs_src, wcs_err = _radec_to_xy_remote(
        row.get("filepath"), ra_deg, dec_deg,
        server=server, remote_script=remote_script, hdu=hdu, verbose=verbose
    )
    wcs_ok = np.isfinite(x) and np.isfinite(y)

    # RBF load
    rbf_cell = row.get("photom_deltazeropoint_rbf")
    rbf = _load_rbf_callable(rbf_cell)
    rbf_ok = callable(rbf)
    dzp_raw = float("nan")
    zp = float("nan")

    if has_zp0 and wcs_ok and rbf_ok:
        dzp_raw_val = None
        # Try a few common call signatures in order
        try:
            dzp_raw_val = float(rbf(float(x), float(y)))          # (x, y)
        except Exception:
            try:
                dzp_raw_val = float(rbf([float(x), float(y)]))    # [x, y]
            except Exception:
                try:
                    dzp_raw_val = float(np.ravel(
                        rbf(np.array([[float(x), float(y)]], dtype=float))
                    )[0])                                          # [[x, y]]
                except Exception:
                    # Try common method names: evaluate/predict/__call__
                    for _name in ("evaluate", "predict", "__call__"):
                        _fn = getattr(rbf, _name, None)
                        if callable(_fn):
                            try:
                                dzp_raw_val = float(np.ravel(
                                    _fn(np.array([[float(x), float(y)]], dtype=float))
                                )[0])
                                break
                            except Exception:
                                try:
                                    dzp_raw_val = float(_fn(float(x), float(y)))
                                    break
                                except Exception:
                                    pass

        if dzp_raw_val is not None and np.isfinite(dzp_raw_val):
            dzp_raw = dzp_raw_val
            dzp = dzp_raw
            dzp_min = row.get("photom_deltazeropoint_min")
            dzp_max = row.get("photom_deltazeropoint_max")
            if dzp_min is not None and dzp_max is not None:
                dzp = float(np.clip(dzp, float(dzp_min), float(dzp_max)))
            zp = float(zp0 + dzp)
            reason = "ok"
        else:
            reason = "rbf_eval:signature"

    return pd.Series({
        "zp_at_radec": zp,
        "_dbg_x": x, "_dbg_y": y,
        "_dbg_wcs_ok": wcs_ok,
        "_dbg_wcs_src": wcs_src,
        "_dbg_wcs_err": wcs_err,
        "_dbg_has_zp0": has_zp0,
        "_dbg_rbf_ok": rbf_ok,
        "_dbg_dzp_raw": dzp_raw,
        "_dbg_reason": reason
    })


# --- Public API ---------------------------------------------------------------

def fetch_photometry(
    filepath: str,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    *,
    db=None,
    debug: bool = False,
    wcs_server: str = DEFAULT_WCS_SERVER,
    wcs_remote_script: str = DEFAULT_WCS_REMOTE_SCRIPT,
    wcs_hdu: int = DEFAULT_WCS_HDU,
    wcs_verbose: bool = False,
):
    """Return zero-point photometry information for *filepath*.

    Parameters
    ----------
    filepath:
        Absolute or relative path recorded in ``image.single.filepath``.
    ra_deg, dec_deg:
        Sky coordinates (degrees). If both are provided, a new column ``zp_at_radec``
        is added with ZP evaluated at that sky position.
    db:
        Optional explicit database connection returned by :func:`get_db`.
    debug:
        If True, include extra debug columns (WCS status, raw delta-ZP, etc.).
    wcs_*:
        Remote WCS options passed to the SSH client.

    Returns
    -------
    pandas.DataFrame
        The photometric calibration row(s) for the image. If no image matches,
        an empty DataFrame is returned.
    """
    sql = """
        SELECT s.*, sq.hfd_median_centre
        FROM image.single AS s
        JOIN image.single_quality AS sq
          ON s.id = sq.single_id
        WHERE s.filepath = %(filepath)s;
    """

    df = pd.DataFrame()

    for attempt in (0, 1):
        try:
            df = query_df(sql, {"filepath": filepath}, db=db)
            break
        except Exception:
            if attempt == 0:
                try:
                    gdb.goto().rollback()
                except Exception:
                    pass
                continue
            raise
    if df is None:
        df = pd.DataFrame()

    if ra_deg is not None and dec_deg is not None and not df.empty:
        series = df.apply(lambda row: _eval_zp_debug(
            row, ra_deg, dec_deg,
            server=wcs_server,
            remote_script=wcs_remote_script,
            hdu=wcs_hdu,
            verbose=wcs_verbose
        ), axis=1)
        df = df.copy()
        df = pd.concat([df, series], axis=1)
        if not debug:
            drop_cols = [c for c in [
                "_dbg_x","_dbg_y","_dbg_wcs_ok","_dbg_has_zp0","_dbg_rbf_ok",
                "_dbg_dzp_raw","_dbg_reason","_dbg_wcs_src","_dbg_wcs_err"
            ] if c in df.columns]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)

    return df


__all__ = ["get_db", "query_df", "fetch_photometry"]
