#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Server-side tool: open FITS on the server, extract WCS-only header
# (via astropy.wcs.WCS.to_header) and return JSON to stdout.
#
# Usage:
#   python3 read_wcs_server.py [--hdu 1] image1.fits [image2.fits ...]
#
# Output JSON schema:
# {
#   "images": [
#     {"path": "...", "hdu": 1, "naxis1": 2048, "naxis2": 2048, "wcs_header": "<FITS header string>"},
#     {"path": "...", "error": "message ..."}
#   ]
# }

import sys
import argparse
import json
from astropy.io import fits
from astropy.wcs import WCS

def _pick_hdu(hdul, prefer_index=1):
    """Prefer HDU[prefer_index] if it has data; otherwise first HDU with 2D data."""
    try:
        h = hdul[prefer_index]
        if getattr(h, "data", None) is not None and getattr(h.data, "ndim", 0) >= 2:
            return prefer_index
    except Exception:
        pass
    for i, h in enumerate(hdul):
        if getattr(h, "data", None) is not None and getattr(h.data, "ndim", 0) >= 2:
            return i
    return 0

def main():
    ap = argparse.ArgumentParser(description="Read WCS from FITS and emit JSON (WCS-only header).")
    ap.add_argument("--hdu", type=int, default=1, help="HDU index to use (default 1). If invalid, auto-pick first 2D data HDU.")
    ap.add_argument("images", nargs="+", help="Remote FITS paths")
    args = ap.parse_args()

    out = {"images": []}
    for p in args.images:
        rec = {"path": p}
        try:
            with fits.open(p, memmap=True) as hdul:
                idx = _pick_hdu(hdul, prefer_index=args.hdu)
                hdu = hdul[idx]
                hdr = hdu.header
                data = hdu.data
                naxis1 = int(hdr.get("NAXIS1", (data.shape[1] if data is not None and data.ndim >= 2 else -1)))
                naxis2 = int(hdr.get("NAXIS2", (data.shape[0] if data is not None and data.ndim >= 2 else -1)))
                w = WCS(hdr)
                wcs_hdr = w.to_header(relax=True)
                wcs_str = wcs_hdr.tostring(sep="\n", padding=False)
                rec.update({"hdu": idx, "naxis1": naxis1, "naxis2": naxis2, "wcs_header": wcs_str})
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        out["images"].append(rec)

    json.dump(out, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")

if __name__ == "__main__":
    main()
