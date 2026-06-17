#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Local client: SSH to server, run read_wcs_server.py remotely, parse JSON,
# reconstruct astropy.wcs.WCS locally for each image.
#
# Examples:
#   python3 wcs_reader_client.py --server monash --images /path/on/server/a.fits /path/on/server/b.fits
#
# Programmatic use:
#   from wcs_reader_client import fetch_wcs
#   info = fetch_wcs(images=["/remote/a.fits"], server="monash")
#   w = info[0]["wcs"];  x, y = w.world_to_pixel(ra_deg, dec_deg)

import os
import os.path as osp
import sys
import shlex
import json
import argparse
import getpass
from typing import List, Dict, Any, Tuple

import paramiko
from paramiko.proxy import ProxyCommand
from astropy.wcs import WCS
from astropy.io import fits

DEFAULT_REMOTE_SCRIPT = "~/Asteroid_backup/package/read_wcs_server.py"

def _clean_host_token(s: str) -> str:
    return (s.split("#", 1)[0]).strip().split()[0] if s else s

def _ssh_lookup(server_alias: str):
    """Resolve SSH settings from ~/.ssh/config. monash->goto2, else gotohead."""
    import paramiko as _pm
    ssh_cfg = _pm.SSHConfig()
    cfg_path = os.path.expanduser("~/.ssh/config")
    host_alias = "goto2" if str(server_alias).lower().startswith("monash") else "gotohead"
    host = {}
    if osp.exists(cfg_path):
        with open(cfg_path) as f:
            ssh_cfg.parse(f)
        host = ssh_cfg.lookup(host_alias)

    hostname = _clean_host_token(host.get("hostname", host_alias))
    username = host.get("user", getpass.getuser())
    port = int(host.get("port", 22))

    identityfiles = host.get("identityfile") or [os.path.expanduser("~/.ssh/id_rsa")]
    keyfile = os.path.expanduser(identityfiles[0])

    proxy_cmd = host.get("proxycommand")
    if not proxy_cmd:
        proxy_jump = host.get("proxyjump")
        if proxy_jump:
            gw_alias = _clean_host_token(proxy_jump.split(",")[0])
            gw = ssh_cfg.lookup(gw_alias)
            gw_keyfiles = [os.path.expanduser(p) for p in (gw.get("identityfile") or [])]
            gw_keyopt = f"-o IdentitiesOnly=yes -i {shlex.quote(gw_keyfiles[0])}" if gw_keyfiles else ""
            proxy_cmd = f"ssh -W {shlex.quote(hostname)}:{port} {gw_keyopt} -o BatchMode=yes -o ConnectTimeout=10 {shlex.quote(gw_alias)}"

    return hostname, port, username, keyfile, proxy_cmd

def _resolve_remote_script(client: paramiko.SSHClient, remote_user: str, preferred: str = DEFAULT_REMOTE_SCRIPT) -> str:
    candidates = [
        preferred,
        f"/home/{remote_user}/Asteroid_backup/package/read_wcs_server.py",
        f"/home/{remote_user}/read_wcs_server.py",
    ]
    for p in candidates:
        cmd = f"test -f {shlex.quote(p)} && echo OK || echo NO"
        _, out, _ = client.exec_command(cmd)
        if out.read().decode().strip() == "OK":
            return p
    raise RuntimeError(f"Cannot find read_wcs_server.py on the remote host. Tried: {', '.join(candidates)}")

def fetch_wcs(images: List[str], server: str = "monash", hdu: int = 1, remote_script: str = DEFAULT_REMOTE_SCRIPT, verbose: bool=False) -> List[Dict[str, Any]]:
    """Return list of dicts: [{'path','hdu','naxis1','naxis2','wcs_header','wcs'}] or with 'error'."""
    hostname, port, username, keyfile, proxy_cmd = _ssh_lookup(server)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sock = ProxyCommand(proxy_cmd) if proxy_cmd else None

    if verbose:
        print("Connecting:", f"{username}@{hostname}:{port}")
        if proxy_cmd:
            print("ProxyCommand:", proxy_cmd)

    client.connect(hostname=hostname, port=port, username=username,
                   key_filename=os.path.expanduser(keyfile),
                   look_for_keys=True, allow_agent=True, sock=sock,
                   timeout=30, banner_timeout=30, auth_timeout=30)

    try:
        script_path = _resolve_remote_script(client, username, preferred=remote_script)
        base_cmd = ["/usr/bin/python3", shlex.quote(script_path), "--hdu", str(int(hdu))] + [shlex.quote(p) for p in images]
        cmd = " ".join(base_cmd)
        if verbose:
            print("[remote cmd]", cmd)

        _, stdout, stderr = client.exec_command(cmd)
        out_b = stdout.read()
        err_b = stderr.read()
        if verbose and err_b:
            sys.stderr.write(err_b.decode("utf-8", errors="ignore"))

        payload = json.loads(out_b.decode("utf-8"))
        out_list = []
        for rec in payload.get("images", []):
            if "error" in rec:
                out_list.append(rec)
                continue
            hdr = fits.Header.fromstring(rec["wcs_header"], sep="\n")
            w = WCS(hdr)
            out_list.append({
                "path": rec.get("path"),
                "hdu": rec.get("hdu"),
                "naxis1": rec.get("naxis1"),
                "naxis2": rec.get("naxis2"),
                "wcs_header": hdr,
                "wcs": w
            })
        return out_list
    finally:
        client.close()

def main():
    ap = argparse.ArgumentParser(description="Fetch WCS from remote FITS via SSH and print a summary.")
    ap.add_argument("--server", default="monash", help='Server name; "monash" maps to SSH host alias goto2')
    ap.add_argument("--hdu", type=int, default=1, help="HDU index to request (default 1)")
    ap.add_argument("--remote-script", default=DEFAULT_REMOTE_SCRIPT, help="Remote path to read_wcs_server.py")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    ap.add_argument("--images", nargs="+", required=True, help="Remote FITS paths")
    ap.add_argument("--test-ra", type=float, default=None, help="Optional RA (deg) to test world->pixel locally")
    ap.add_argument("--test-dec", type=float, default=None, help="Optional Dec (deg) to test world->pixel locally")
    args = ap.parse_args()

    info = fetch_wcs(images=args.images, server=args.server, hdu=args.hdu,
                     remote_script=args.remote_script, verbose=args.verbose)
    for i, rec in enumerate(info, 1):
        if "error" in rec:
            print(f"[{i}] {rec['path']}  ERROR: {rec['error']}")
            continue
        naxis1, naxis2 = rec["naxis1"], rec["naxis2"]
        print(f"[{i}] {rec['path']}  HDU={rec['hdu']}  size=({naxis1}x{naxis2})")
        if args.test_ra is not None and args.test_dec is not None:
            try:
                x, y = rec["wcs"].world_to_pixel_values(args.test_ra, args.test_dec)
                print(f"     world({args.test_ra},{args.test_dec}) -> pixel({x:.2f},{y:.2f})")
            except Exception as e:
                print(f"     world->pixel failed: {e}")

if __name__ == "__main__":
    main()
