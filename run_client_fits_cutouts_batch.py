#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Remote client for ``make_fits_cutouts_batch.py``.

It provides the same SSH discovery logic (host aliases resolved via
``~/.ssh/config``), legacy single-image fallbacks, and optional WCS loosening,
while focusing on collecting the FITS (and any other) files produced by the
remote batch job.

Usage mirrors ``run_client_batch.py``::

    python run_client_fits_cutouts_batch.py \
        --server monash --ra 123.45 --dec -54.321 --size 256 \
        --images /path/on/remote/image1.fits /path/on/remote/image2.fits \
        --local-temp ./cutouts

After running the remote script, every newly created file in the output
directory is downloaded to ``local_temp``.  By default, all new files are
fetched, but you may restrict the extensions using ``--extensions``.
"""

from __future__ import annotations

import argparse
import getpass
import os
import os.path as osp
import shlex
import sys
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import paramiko
from paramiko.proxy import ProxyCommand
from scp import SCPClient, SCPException


class FitsCutoutClient:
    """SSH client used to invoke ``make_fits_cutouts_batch.py`` remotely."""
    def _expect_png_for(self, image_path: str, ra_deg: float, dec_deg: float, out_dir: str) -> str:
        """
        Predict the remote output filename for a given input image and RA/Dec.

        This **must** mirror the naming used by the remote cutout script.

        For make_fits_cutouts_batch.py we expect something like:

            input:  /.../t1_r0868936_ut4.fits
            RA,Dec: 20.63907, 15.60471
            output: <out_dir>/t1_r0868936_ut4_20p63907_15p60471.fits

        Note: although this helper is called *_png_for* for historical
        reasons, here it returns a .fits path, because the remote script
        writes FITS cutouts.
        """
        # basename without extension, e.g. "t1_r0868936_ut4"
        base = osp.splitext(osp.basename(str(image_path)))[0]

        # format RA/Dec to 5 decimals and replace '.' with 'p'
        ra = float(ra_deg)
        dec = float(dec_deg)
        ra_str = f"{ra:.5f}".replace(".", "p")
        dec_str = f"{dec:.5f}".replace(".", "p")

        # Remote cutout is a FITS file
        return f"{out_dir.rstrip('/')}/{base}_{ra_str}_{dec_str}.fits"

    def __init__(
        self,
        date: Optional[str] = None,
        runid: Optional[str] = None,
        camera: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        median: bool = True,
        server: Optional[str] = None,
        size: Optional[int] = None,
        display: Optional[bool] = None,
        outpath: Optional[str] = None,
        local_temp: Optional[str] = None,
        verbose: bool = False,
        images: Optional[Sequence[str]] = None,
        image: Optional[str] = None,
        loose_wcs: bool = False,
        remote_script: Optional[str] = None,
        extensions: Optional[Sequence[str]] = None,
        remote_args: Optional[Sequence[str]] = None,
        overwrite: bool = True,
    ) -> None:
        self.server = server or "monash"
        self.verbose = bool(verbose)
        self.display = True if display is None else bool(display)

        # legacy params
        self.date = date
        self.runid = runid
        self.camera = camera
        self.median = "-median" if median else ""
        self.ra = float(ra) if ra is not None else None
        self.dec = float(dec) if dec is not None else None
        self.size = int(size or 30)

        self.images = list(images or ([] if image is None else [image]))

        self.outpath = outpath
        self.local_temp = local_temp or "./"
        self.loose_wcs = bool(loose_wcs)
        self.remote_script = remote_script
        self.remote_args = list(remote_args or [])
        self.overwrite = bool(overwrite)

        if extensions:
            self.extensions = [self._normalize_ext(ext) for ext in extensions if ext]
        else:
            self.extensions = []  # download everything new

    # -------- helpers --------
    @staticmethod
    def _normalize_ext(ext: str) -> str:
        """Normalize extension to start with a dot ('.fits' instead of 'fits')."""
        ext = ext.strip()
        if not ext:
            return ""
        return ext if ext.startswith(".") else f".{ext}"

    @staticmethod
    def _clean_host_token(s: str) -> str:
        """Remove inline # comments and extra spaces; return first token."""
        return (s.split('#', 1)[0]).strip().split()[0] if s else s

    @staticmethod
    def _expand_remote_path(path: str, remote_user: str) -> str:
        """Expand ~ to /home/<user> in remote paths; leave others as-is."""
        if not path:
            return path
        p = path.strip()
        if p == "~":
            return f"/home/{remote_user}"
        if p.startswith("~/"):
            return f"/home/{remote_user}/{p[2:]}"
        return p

    def _ssh_lookup(self):
        """
        Resolve SSH host using ~/.ssh/config.
    
        For simplicity, we always use the same host alias that you use
        on the command line, e.g. `ssh gotocompute4`.
    
        Returns (hostname, port, username, keyfile, proxy_cmd_or_None).
        """
        import paramiko as _pm
        ssh_cfg = _pm.SSHConfig()
        cfg_path = os.path.expanduser("~/.ssh/config")
    
        # Use the same alias as your normal SSH command
        host_alias = "gotocompute4"
    
        host = {}
        if osp.exists(cfg_path):
            with open(cfg_path) as f:
                ssh_cfg.parse(f)
            host = ssh_cfg.lookup(host_alias)
    
        # Fall back to alias if hostname is not explicitly set
        hostname = self._clean_host_token(host.get("hostname", host_alias))
        username = host.get("user", getpass.getuser())
        port = int(host.get("port", 22))
    
        identityfiles = host.get("identityfile") or [os.path.expanduser("~/.ssh/id_rsa")]
        keyfile = os.path.expanduser(identityfiles[0])
    
        proxy_cmd = host.get("proxycommand")  # usually None for direct ssh gotocompute4
    
        return hostname, port, username, keyfile, proxy_cmd


    def _remote_defaults(self, remote_user: str) -> Tuple[str, List[str]]:
        """Decide default remote outdir and script location."""
        out_dir = self._expand_remote_path(self.outpath, remote_user) if self.outpath else None
        if out_dir is None:
            if self.server == "monash":
                out_dir = f"/mnt4/data/{remote_user}/mp_tmp"
            else:
                out_dir = f"/home/{remote_user}/out"
        out_dir = out_dir.rstrip("/")

        if self.remote_script:
            script_candidates = [self._expand_remote_path(self.remote_script, remote_user)]
        else:
            base_dir = osp.dirname(out_dir) or f"/home/{remote_user}"
            script_candidates = [
                f"{base_dir}/make_fits_cutouts_batch.py",
                f"/home/{remote_user}/Asteroid_backup/package/make_fits_cutouts_batch.py",
                f"/home/{remote_user}/make_fits_cutouts_batch.py",
            ]
        return out_dir, script_candidates

    def _legacy_single_image(self, remote_username: str) -> str:
        """Build a single remote FITS path from legacy date/runid/camera fields."""
        if self.server == "monash":
            if not (self.date and self.runid and self.camera):
                raise ValueError("Need --date/--runid/--camera or provide --images/--image.")
            return (
                f"/mnt3/data/public/goto/commissioning/pipeline/{self.date}/final/"
                f"{self.runid}_{self.camera}{self.median}.fits"
            )
        elif self.server == "warwick":
            if not (self.date and self.runid and self.camera):
                raise ValueError("Need --date/--runid/--camera or provide --images/--image.")
            return (
                f"/export/gotodata2/gotophoto/storage/pipeline/{self.date}/final/"
                f"{self.runid}_{self.camera}.fits"
            )
        else:
            raise ValueError('server must be "monash" or "warwick"')

    @staticmethod
    def _resolve_remote_script(client: paramiko.SSHClient, candidates: Sequence[str]) -> str:
        """Pick the first existing remote path from the candidate list."""
        for p in candidates:
            cmd = f"test -f {shlex.quote(p)} && echo OK || echo NO"
            _, out, _ = client.exec_command(cmd)
            if out.read().decode().strip() == "OK":
                return p
        raise RuntimeError(
            "Could not find make_fits_cutouts_batch.py on remote host. "
            f"Tried: {', '.join(candidates)}"
        )

    @staticmethod
    def _list_remote_files(client: paramiko.SSHClient, directory: str) -> List[str]:
        """Return a simple list of file names (no directories) in the remote directory."""
        cmd = f"cd {shlex.quote(directory)} && ls -1"
        _, stdout, stderr = client.exec_command(cmd)
        err = stderr.read().decode("utf-8", errors="ignore")
        if err and "No such" in err:
            return []
        out = stdout.read().decode("utf-8", errors="ignore")
        files = [line.strip() for line in out.splitlines() if line.strip()]
        return files

    def _filter_new_files(self, files: Iterable[str]) -> List[str]:
        """Filter remote files by extension if extensions are specified."""
        if not self.extensions:
            return sorted(set(files))
        filtered = []
        for f in files:
            lower = f.lower()
            if any(lower.endswith(ext) for ext in self.extensions):
                filtered.append(f)
        return sorted(set(filtered))

    def _build_remote_command(self, script_path: str,
                              images: Sequence[str],
                              out_dir: str) -> str:
        """Build the full remote python command to run the cutout script."""
        base_cmd = [
            "/usr/bin/python3",
            shlex.quote(script_path),
            "--ra",
            str(self.ra),
            "--dec",
            str(self.dec),
            "--size",
            str(int(self.size)),
            "--outdir",
            shlex.quote(out_dir),
        ]
        if self.loose_wcs:
            base_cmd.append("--loose-wcs")
        if self.overwrite:
            base_cmd.append("--overwrite")
        if self.remote_args:
            base_cmd.extend([shlex.quote(arg) for arg in self.remote_args])
        base_cmd.extend(shlex.quote(p) for p in images)
        return " ".join(base_cmd)

    @staticmethod
    def _parse_stdout_for_outputs(stdout: str) -> List[str]:
        """
        Parse the remote script stdout for output file hints.

        Expected format lines like:
            [OK] input.fits -> /path/to/output.fits
        """
        outputs: List[str] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("[OK]"):
                continue
            parts = line.split("->", 1)
            if len(parts) != 2:
                continue
            remote_path = parts[1].strip()
            if not remote_path:
                continue
            outputs.append(os.path.basename(remote_path))
        return outputs

    @staticmethod
    def _safe_scp_get(
        scp: SCPClient,
        remote_path: str,
        local_path: str,
        max_retries: int = 3,
        delay_s: float = 5.0,
    ) -> None:
        """
        Wrapper around scp.get() with simple retry logic.

        - max_retries: number of attempts before giving up
        - delay_s: seconds to wait between attempts
        """
        for attempt in range(1, max_retries + 1):
            try:
                scp.get(remote_path, local_path=local_path)

                # Sanity check: file exists and is non-empty
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    return

                raise SCPException("Downloaded file is empty")

            except Exception as e:
                if attempt >= max_retries:
                    print(
                        f"[ERR] SCP GET failed after {max_retries} attempts "
                        f"for {remote_path}: {e}"
                    )
                    # Propagate error to caller
                    raise

                print(
                    f"[WARN] SCP GET failed (attempt {attempt}/{max_retries}) "
                    f"for {remote_path}: {e}; retry in {delay_s}s ..."
                )

                # Clean up any partial local file before retry
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                except OSError:
                    pass

                time.sleep(delay_s)

    def client(self) -> List[str]:
        """Connect to remote host, run script, download new files, and return local paths."""
        if self.ra is None or self.dec is None:
            raise ValueError("Please provide --ra and --dec (degrees).")
        if self.size is None:
            raise ValueError("Please provide --size (pixels).")

        hostname, port, username, keyfile, proxy_cmd = self._ssh_lookup()

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        sock = ProxyCommand(proxy_cmd) if proxy_cmd else None
        if self.verbose:
            print("proxy_cmd =", proxy_cmd)
            print("keyfile  =", keyfile)
            print("target   =", f"{username}@{hostname}:{port}")

        client.connect(
            hostname=hostname,
            port=port,
            username=username,
            key_filename=osp.expanduser(keyfile),
            look_for_keys=True,
            allow_agent=True,
            sock=sock,
            timeout=120,
            banner_timeout=120,
            auth_timeout=120,
        )

        os.makedirs(self.local_temp, exist_ok=True)

        downloaded: List[str] = []
        try:
            out_dir, script_candidates = self._remote_defaults(username)
            script_path = self._resolve_remote_script(client, script_candidates)

            # Ensure remote output directory exists
            client.exec_command(f"mkdir -p {shlex.quote(out_dir)}")

            # Resolve image list
            imgs = self.images[:] if self.images else [self._legacy_single_image(username)]
            imgs = [self._expand_remote_path(p, username) for p in imgs]

            # Snapshot existing files before running the job
            before = set(self._list_remote_files(client, out_dir))

            cmd = self._build_remote_command(script_path, imgs, out_dir)
            if self.verbose:
                print("[remote cmd]", cmd)

            _, stdout, stderr = client.exec_command(cmd)
            out_b = stdout.read()
            err_b = stderr.read()
            out_text = out_b.decode("utf-8", errors="ignore")
            err_text = err_b.decode("utf-8", errors="ignore")
            if out_text:
                sys.stdout.write(out_text)
            if err_text:
                sys.stderr.write(err_text)

            # List files again and compute "new" set
            after = set(self._list_remote_files(client, out_dir))
            new_files = self._filter_new_files(after - before)

            # Fallback: try parsing stdout for explicit output paths
            if not new_files:
                parsed = self._parse_stdout_for_outputs(out_text)
                if parsed:
                    parsed = [f for f in parsed if f in after]
                    new_files = self._filter_new_files(parsed)

            if not new_files:
                print("[WARN] No new files detected in remote output directory.")
                return []

            # Download new files via SCP (with retry and a more generous timeout)
            from scp import SCPClient, SCPException

            with SCPClient(client.get_transport(), socket_timeout=60.0) as scp:
                for im in imgs:
                    expect_png = self._expect_png_for(im, self.ra, self.dec, out_dir)
                    cmd_chk = f"test -f {shlex.quote(expect_png)} && echo OK || echo NO"
                    _, out, _ = client.exec_command(cmd_chk)
                    ok = out.read().decode().strip() == "OK"
                    remote_png = expect_png
            
                    if not ok:
                        print(f"[WARN] No PNG found for {im} in {out_dir}")
                        continue
            
                    local_png = os.path.join(self.local_temp, osp.basename(remote_png))
                    try:
                        scp.get(remote_png, local_path=local_png)
                    except SCPException as e:
                        # Don't kill the whole workflow just because one frame failed
                        print(f"[WARN] SCP failed for {remote_png}: {e}")
                        continue
            
                    downloaded.append(local_png)


            if self.display and downloaded:
                print("[INFO] Downloaded files:")
                for path in downloaded:
                    print("    ", path)

            return downloaded
        finally:
            client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remote batch client for make_fits_cutouts_batch.py"
    )
    parser.add_argument(
        "--server",
        default="warwick",
        type=str,
        help='Set server to "monash" or "warwick"',
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Turn on logging and verbosity",
    )
    parser.add_argument(
        "--display",
        default=True,
        help="Display list of downloaded files",
    )

    # legacy parameters
    parser.add_argument(
        "--date",
        default=None,
        type=str,
        help="Night ID for observation",
    )
    parser.add_argument(
        "--runid",
        default=None,
        type=str,
        help="Image ID for observation",
    )
    parser.add_argument(
        "--camera",
        default=None,
        type=str,
        help='UT ID. Give as "UT?"',
    )
    parser.add_argument(
        "--median",
        default=True,
        help="Use median-combined images in legacy mode",
    )

    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="One or more full remote FITS paths",
    )
    parser.add_argument(
        "--image",
        default=None,
        type=str,
        help="(Deprecated) Single remote FITS path",
    )

    parser.add_argument(
        "--ra",
        required=True,
        type=float,
        help="RA of center (deg)",
    )
    parser.add_argument(
        "--dec",
        required=True,
        type=float,
        help="DEC of center (deg)",
    )
    parser.add_argument(
        "--size",
        required=True,
        type=int,
        help="Box size in pixels",
    )

    parser.add_argument(
        "--outpath",
        default=None,
        type=str,
        help="Remote out directory on server (absolute or ~)",
    )
    parser.add_argument(
        "--local_temp",
        default=".",
        type=str,
        help="Local directory to save cutouts",
    )

    parser.add_argument(
        "--loose-wcs",
        action="store_true",
        help="Pass --loose-wcs to the remote script",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite existing files in the remote output directory",
    )
    parser.set_defaults(overwrite=True)
    parser.add_argument(
        "--remote-script",
        default=None,
        type=str,
        help="Explicit remote path to make_fits_cutouts_batch.py (overrides default lookup)",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Limit downloads to files ending with these extensions "
             "(e.g. --extensions fits png)",
    )
    parser.add_argument(
        "--remote-arg",
        dest="remote_args",
        action="append",
        default=None,
        help="Extra argument to append to the remote script command (use multiple times)",
    )

    args = parser.parse_args()
    client = FitsCutoutClient(**vars(args))
    client.client()
