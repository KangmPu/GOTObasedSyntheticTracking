#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import os.path as osp
import shlex
import argparse
import getpass
import time
from typing import List, Tuple
from datetime import datetime

import paramiko
from paramiko.proxy import ProxyCommand
from scp import SCPClient, SCPException


class Client:
    def __init__(self, date=None, runid=None, camera=None, ra=None, dec=None,
                 median=True, server=None, size=None,
                 display=None, outpath=None, local_temp=None,
                 verbose=False, images=None, image=None,
                 loose_wcs=False,
                 gif_out=None, gif_duration=0.2, gif_loop=0, keep_png=False):
        """
        images: list of remote FITS paths; if provided, overrides legacy date/runid/camera.
        image : kept for backward-compat (single path); if given and images is None, it will be used.
        loose_wcs: if True, pass --loose-wcs to the server script to enable relaxed WCS handling.
        gif_out: local path for the output GIF. If None, defaults to <local_temp>/thumbs_<timestamp>.gif
        gif_duration: per-frame duration (seconds) for GIF.
        gif_loop: GIF loop count (0 = infinite; 1 = play once then stop; 2 = play twice; ...).
        keep_png: if False (default), remove the downloaded PNGs after making GIF.
        """

        self.server = server or "monash"
        self.verbose = verbose
        self.display = True if display is None else bool(display)

        # legacy params (still supported for single image fallback)
        self.date = date
        self.runid = runid
        self.camera = camera
        self.median = "-median" if median else ""
        self.ra = float(ra) if ra is not None else None
        self.dec = float(dec) if dec is not None else None
        self.size = int(size or 30)

        # batch inputs
        self.images = images or ([] if image is None else [image])

        self.outpath = outpath
        self.local_temp = local_temp or "./"
        self.loose_wcs = bool(loose_wcs)

        # GIF options
        self.gif_duration = float(gif_duration)
        self.gif_loop = int(gif_loop)
        self.keep_png = bool(keep_png)
        self.gif_out = gif_out  # resolved later once local_temp is known

    # -------- helpers --------
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
        Read ~/.ssh/config to resolve host alias and ProxyCommand/ProxyJump.
        We'll map server name: monash->goto2, warwick->gotohead (same as your old client).
        Returns (hostname, port, username, keyfile, proxy_cmd_str or None)
        """
        import paramiko as _pm
        ssh_cfg = _pm.SSHConfig()
        cfg_path = os.path.expanduser("~/.ssh/config")
        host_alias = "goto2" if str(self.server).lower().startswith("monash") else "gotohead"
        host = {}

        if osp.exists(cfg_path):
            with open(cfg_path) as f:
                ssh_cfg.parse(f)
            host = ssh_cfg.lookup(host_alias)

        hostname = self._clean_host_token(host.get("hostname", host_alias))
        username = host.get("user", getpass.getuser())
        port = int(host.get("port", 22))

        identityfiles = host.get("identityfile") or [os.path.expanduser("~/.ssh/id_rsa")]
        keyfile = os.path.expanduser(identityfiles[0])

        proxy_cmd = host.get("proxycommand")
        if not proxy_cmd:
            proxy_jump = host.get("proxyjump")
            if proxy_jump:
                gw_alias = self._clean_host_token(proxy_jump.split(",")[0])
                gw = ssh_cfg.lookup(gw_alias)
                gw_keyfiles = [os.path.expanduser(p) for p in (gw.get("identityfile") or [])]
                gw_keyopt = f"-o IdentitiesOnly=yes -i {shlex.quote(gw_keyfiles[0])}" if gw_keyfiles else ""
                proxy_cmd = (
                    f"ssh -W {shlex.quote(hostname)}:{port} {gw_keyopt} "
                    f"-o BatchMode=yes -o ConnectTimeout=10 {shlex.quote(gw_alias)}"
                )

        return hostname, port, username, keyfile, proxy_cmd

    def _remote_defaults(self, remote_user: str):
        """
        Decide default remote outdir and script location.
        We assume your server-side script is named 'make_thumbnails_batch.py'.
        If outpath is provided, we honor it; otherwise choose server-specific default.
        Script path is the parent of outdir by default (e.g., ~/Asteroid_backup/package).
        """
        # outdir
        out_dir = self._expand_remote_path(self.outpath, remote_user) if self.outpath else None
        if out_dir is None:
            if self.server == "monash":
                out_dir = f"/mnt4/data/{remote_user}/mp_tmp"
            else:
                out_dir = f"/home/{remote_user}/out"
        # normalize
        out_dir = out_dir.rstrip("/")

        # script path: try alongside typical location, else fallback to ~/Asteroid_backup/package
        base_dir = osp.dirname(out_dir) or f"/home/{remote_user}"
        candidates = [
            f"{base_dir}/make_thumbnails_batch.py",
            f"/home/{remote_user}/Asteroid_backup/package/make_thumbnails_batch.py",
            f"/home/{remote_user}/make_thumbnails_batch.py",
        ]
        return out_dir, candidates

    def _legacy_single_image(self, remote_username: str) -> str:
        """Build a single image path using legacy date/runid/camera if none explicitly provided."""
        if self.server == "monash":
            if not (self.date and self.runid and self.camera):
                raise ValueError("Need --date/--runid/--camera or provide --images/--image.")
            return (
                "/mnt3/data/public/goto/commissioning/pipeline/"
                f"{self.date}/final/{self.runid}_{self.camera}{self.median}.fits"
            )
        elif self.server == "warwick":
            if not (self.date and self.runid and self.camera):
                raise ValueError("Need --date/--runid/--camera or provide --images/--image.")
            return (
                "/export/gotodata2/gotophoto/storage/pipeline/"
                f"{self.date}/final/{self.runid}_{self.camera}.fits"
            )
        else:
            raise ValueError('server must be "monash" or "warwick"')

    @staticmethod
    def _expect_png_for(image_path: str, ra_deg: float, dec_deg: float, out_dir: str) -> str:
        """Server names files as <runid>_<round(ra)>_<round(dec)>.png under out_dir."""
        runid = osp.splitext(osp.basename(image_path))[0]
        ra_i, dec_i = int(round(ra_deg)), int(round(dec_deg))
        return f"{out_dir}/{runid}_{ra_i}_{dec_i}.png"

    def _resolve_remote_script(self, client: paramiko.SSHClient, candidates: List[str]) -> str:
        """Pick the first existing path from candidates on the remote host."""
        for p in candidates:
            cmd = f"test -f {shlex.quote(p)} && echo OK || echo NO"
            _, out, _ = client.exec_command(cmd)
            if out.read().decode().strip() == "OK":
                return p
        raise RuntimeError(
            "Could not find make_thumbnails_batch.py on remote host. "
            f"Tried: {', '.join(candidates)}"
        )

    @staticmethod
    def _make_gif_from_pngs(png_paths: List[str], gif_path: str,
                            duration: float = 0.2, loop: int = 0):
        """Create a GIF from ordered PNG paths and save to gif_path."""
        pngs = [p for p in png_paths if p and osp.isfile(p) and p.lower().endswith(".png")]
        if not pngs:
            raise RuntimeError("No PNGs to build GIF.")

        # Try imageio first, fallback to Pillow
        try:
            import imageio.v2 as imageio
            frames = [imageio.imread(p) for p in pngs]
            imageio.mimsave(gif_path, frames, duration=duration, loop=loop)
            return
        except Exception as e:
            try:
                from PIL import Image
                imgs = [Image.open(p).convert("P") for p in pngs]
                imgs[0].save(
                    gif_path,
                    save_all=True,
                    append_images=imgs[1:],
                    duration=int(duration * 1000),
                    loop=loop,
                    disposal=2,
                )
                return
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to create GIF with imageio and Pillow: {e}; {e2}"
                )

    @staticmethod
    def _safe_scp_get(scp: SCPClient, remote_path: str, local_path: str,
                      max_retries: int = 3, delay_s: float = 5.0):
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
                    # propagate the error so caller can mark this subgroup as failed
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

    def client(self) -> Tuple[str, List[str]]:
        """Run remote job, download PNGs, stitch GIF locally, and optionally remove PNGs."""
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
            key_filename=os.path.expanduser(keyfile),
            look_for_keys=True,
            allow_agent=True,
            sock=sock,
            timeout=60,
            banner_timeout=60,
            auth_timeout=60,
        )

        # Decide gif output path
        os.makedirs(self.local_temp, exist_ok=True)
        if not self.gif_out:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.gif_out = osp.join(self.local_temp, f"thumbs_{stamp}.gif")
        else:
            self.gif_out = osp.expanduser(self.gif_out)

        downloaded: List[str] = []
        try:
            out_dir, script_candidates = self._remote_defaults(username)
            script_path = self._resolve_remote_script(client, script_candidates)

            # Ensure outdir exists
            client.exec_command(f"mkdir -p {shlex.quote(out_dir)}")

            # Resolve image list
            imgs = self.images[:] if self.images else [self._legacy_single_image(username)]
            imgs = [self._expand_remote_path(p, username) for p in imgs]

            # Assemble remote command
            base_cmd = [
                "/usr/bin/python3",
                shlex.quote(script_path),
                "--ra", str(self.ra),
                "--dec", str(self.dec),
                "--size", str(int(self.size)),
                "--outdir", shlex.quote(out_dir),
            ]

            if self.loose_wcs:
                base_cmd.append("--loose-wcs")

            base_cmd += [shlex.quote(p) for p in imgs]

            cmd = " ".join(base_cmd)
            if self.verbose:
                print("[remote cmd]", cmd)

            _, stdout, stderr = client.exec_command(cmd)
            out_b = stdout.read()
            err_b = stderr.read()
            if out_b:
                sys.stdout.write(out_b.decode("utf-8", errors="ignore"))
            if err_b:
                sys.stderr.write(err_b.decode("utf-8", errors="ignore"))

            # Download results in the same order as imgs
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
                    # Use the retry wrapper to mitigate transient network / latency issues
                    self._safe_scp_get(scp, remote_png, local_png)
                    downloaded.append(local_png)

            # Build GIF from downloaded list (keeps order of imgs/fallbacks)
            if downloaded:
                self._make_gif_from_pngs(
                    downloaded,
                    self.gif_out,
                    duration=self.gif_duration,
                    loop=self.gif_loop,
                )
                print(f"[GIF] Saved: {self.gif_out} (loop={self.gif_loop})")
                # Remove PNGs if not keeping
                if not self.keep_png:
                    for p in downloaded:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
            else:
                print("[WARN] No PNGs downloaded; GIF not created.")

            if self.display and downloaded:
                print(f"[INFO] GIF ready at: {self.gif_out}")

            return self.gif_out, downloaded

        finally:
            client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remote batch thumbnail client: download PNGs, stitch GIF, optionally remove PNGs."
    )
    parser.add_argument("--server", default="monash", type=str,
                        help='Set server to "monash" or "warwick"')
    parser.add_argument("--verbose", action="store_true",
                        help="Turn on logging and verbosity")
    parser.add_argument("--display", default=True,
                        help="Display info about the generated GIF")

    # legacy parameters (still usable for single-image fallback)
    parser.add_argument("--date", default=None, type=str,
                        help="Night ID for observation")
    parser.add_argument("--runid", default=None, type=str,
                        help="Image ID for observation")
    parser.add_argument("--camera", default=None, type=str,
                        help='UT ID. Give as "UT?"')
    parser.add_argument("--median", default=True,
                        help="Use median-combined images in legacy mode")

    # batch inputs
    parser.add_argument("--images", nargs="+", default=None,
                        help="One or more full remote FITS paths")
    parser.add_argument("--image", default=None, type=str,
                        help="(Deprecated) Single remote FITS path")

    parser.add_argument("--ra", required=True, type=float,
                        help="RA of center (deg)")
    parser.add_argument("--dec", required=True, type=float,
                        help="DEC of center (deg)")
    parser.add_argument("--size", required=True, type=int,
                        help="Box size in pixels")

    parser.add_argument("--outpath", default="~/out", type=str,
                        help="Remote out directory on server (absolute or ~)")
    parser.add_argument("--local_temp", default=".", type=str,
                        help="Local directory to save thumbnails and GIF")

    parser.add_argument("--loose-wcs", action="store_true",
                        help="Pass --loose-wcs to the server to relax WCS handling. Default off.")

    # GIF options
    parser.add_argument("--gif-out", default=None, type=str,
                        help="Local output GIF path; default <local_temp>/thumbs_<timestamp>.gif")
    parser.add_argument("--gif-duration", default=0.2, type=float,
                        help="Per-frame duration (seconds) for the GIF (default 0.2s)")
    parser.add_argument("--gif-loop", default=0, type=int,
                        help="GIF loop count (0=infinite; 1=play once; 2=play twice; ...)")
    parser.add_argument("--keep-png", action="store_true",
                        help="Keep PNGs after making GIF (default: remove)")

    args = vars(parser.parse_args())
    r = Client(**args)
    r.client()
