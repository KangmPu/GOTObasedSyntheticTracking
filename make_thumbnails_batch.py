
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import os.path as osp
import argparse
from typing import Tuple

import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import ZScaleInterval
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

interval = ZScaleInterval()

def _pick_image_hdu(hdul: fits.HDUList) -> Tuple[int, fits.ImageHDU]:
    """Prefer HDU[1]; fallback to first IMAGE HDU with data."""
    if len(hdul) > 1 and getattr(hdul[1], "data", None) is not None:
        return 1, hdul[1]
    for i, h in enumerate(hdul):
        if getattr(h, "data", None) is not None:
            return i, h
    raise RuntimeError("No image HDU with data found.")

def _scale_to_uint8_zscale(arr: np.ndarray):
    """Z-scale -> clip -> linear map to uint8 [0..255]."""
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8), 0.0, 1.0
    vmin, vmax = interval.get_limits(arr[finite])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(arr[finite]))
        vmax = float(np.nanmax(arr[finite]))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.zeros(arr.shape, dtype=np.uint8), 0.0, 1.0
    clipped = np.clip(arr, vmin, vmax)
    scaled = (clipped - vmin) / (vmax - vmin)
    u8 = np.round(scaled * 255.0).astype(np.uint8)
    return u8, float(vmin), float(vmax)

def make_thumbnail(image_path: str, ra_deg: float, dec_deg: float, size_px: int, outdir: str) -> str:
    """Create a borderless 8-bit PNG (z-scale stretched) cutout centered at (ra, dec),
    then flip vertically and rotate 90° counter-clockwise before saving."""
    ra_deg = float(ra_deg); dec_deg = float(dec_deg); size_px = int(size_px)
    os.makedirs(outdir, exist_ok=True)
    sc = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")

    with fits.open(image_path, memmap=True) as hdul:
        idx, ihdu = _pick_image_hdu(hdul)
        data = ihdu.data
        hdr  = ihdu.header
        if data is None or getattr(data, "ndim", 0) < 2:
            raise RuntimeError(f"{image_path}: empty/invalid image data in HDU[{idx}]")

        w = wcs.WCS(hdr)
        cut = Cutout2D(data, sc, size_px, wcs=w).data

    u8, vmin, vmax = _scale_to_uint8_zscale(cut)

    # Convert to image, apply flip (上下翻转) then 90° CCW rotation
    im = Image.fromarray(u8, mode="L")
    im = ImageOps.flip(im)                 # 上下翻转 (vertical flip)
    im = im.transpose(Image.ROTATE_90)     # 逆时针 90°（Pillow 的 ROTATE_90 是 CCW）

    # Save borderless grayscale PNG
    runid = osp.splitext(osp.basename(image_path))[0]
    outname = f"{runid}_{int(round(ra_deg))}_{int(round(dec_deg))}.png"
    outpath = osp.join(outdir, outname)

    meta = PngInfo()
    meta.add_text("SRCFILE", osp.basename(image_path))
    meta.add_text("CUTSIZE", str(size_px))
    meta.add_text("CUTRA", f"{ra_deg:.8f}")
    meta.add_text("CUTDEC", f"{dec_deg:.8f}")
    meta.add_text("STRETCH", "zscale")
    meta.add_text("VMIN", f"{vmin:.12g}")
    meta.add_text("VMAX", f"{vmax:.12g}")
    meta.add_text("ORIENT", "flipud+rot90_ccw")
    im.save(outpath, pnginfo=meta, optimize=False, compress_level=6)
    return outpath

def main():
    ap = argparse.ArgumentParser(description="Make borderless 8-bit PNG thumbnails (z-scale) with flip+CCW90.")
    ap.add_argument("--ra", type=float, required=True, help="ICRS RA (deg)")
    ap.add_argument("--dec", type=float, required=True, help="ICRS Dec (deg)")
    ap.add_argument("--size", type=int, required=True, help="Box size in pixels")
    ap.add_argument("--outdir", required=True, help="Output directory for PNGs")
    ap.add_argument("images", nargs="+", help="Input FITS file(s)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ok, fail = 0, 0
    for path in args.images:
        try:
            out = make_thumbnail(path, args.ra, args.dec, args.size, args.outdir)
            print(f"[OK] {path} -> {out}")
            ok += 1
        except Exception as e:
            print(f"[ERROR] {path}: {e}", file=sys.stderr)
            fail += 1

    if ok == 0:
        sys.exit(2)
    sys.exit(0)

if __name__ == "__main__":
    main()
