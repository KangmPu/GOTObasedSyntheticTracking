The system now integrates remote server processing, photometric calibration, GIF-based motion detection and ppt generation for displaying result.

The workflow is designed to run daily under a scheduled task (e.g., Windows Task Scheduler or cron) and produces both tabular and visual reports.

---

## Directory Structure
```
package/
├── daily_workflow.py              # Master workflow; runs daily to generate GIFs, tables, and reports
├── detect_mover_gif.py            # Detects motion from multi-frame GIFs
├── autochecker_MPClist_pipeline.py# Cross-matches recent GOTO images from pipline with MPC targets lists and builds result table
├── pill_matched_phot.py           # performs forced (pill-aperture) photometry (in developing)
├── target_searching_match.py      # Matches observation records with ephemerides (can also generate ephemeris)
├── run_client_batch.py            # Handles SSH connection and remote thumbnail generation (place in local)
├── run_client_fits_cutouts_batch.py # Handles SSH connection and remote fits cutout generation (place in local)
├── build_daily_ppt.py             # Builds PowerPoint reports from daily outputs 
├── make_thumbnails_batch.py 	   # place in server side to generate edgeless thumbnail.
├── make_fits_cutouts_batch.py     # place in server side to generate cut of fits file.
├── target_searching.py            # fetching asteroids ephems and goto obs records, matching the results.
└── etc.
```

---

## Installation
Ensure `gotodb` install before proceeding

Create and activate the environment, then install dependencies from the repository root:
```bash
conda create -n gotodb_env python=3.11
conda activate gotodb_env
pip -m install -r requirements.txt

```
install google-chrome for using (Ubuntu 22.04)
```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install -y ./google-chrome-stable_current_amd64.deb
```
Check Configuration to ensure you can connect server via `ssh gotocompute4` correctly.

Remenmber to copy the `read_wcs_server.py` `make_thumbnails_batch.py` `make_fits_cutouts_batch.py` three script to your personal folder on server.

Notable dependencies (see `requirements.txt` for the full list):
- **Astropy**, **Photutils**, **Numpy**, **Pandas**, **TQDM**
- **gotodb** (internal module for database access)
- **Pillow**, **ImageIO** (image/GIF handling)
- **paramiko**, **scp** (SSH file transfer to/from Warwick servers)
- **selenium**, **astroquery**, **shapely**, **scikit-learn**, **webdriver-manager**

---

## Workflow Summary

### Recommended daily order
- Run `daily_workflow.py` first to gather observations, build GIFs, and summarize the night.
- If `daily_workflow.py` produces targets of interest, follow up with `AP_workflow.py` to fetch WCS, run photometry, and package the final products.

### 1. `daily_workflow.py`
Main entry point for daily processing.
- Collects observation data
- Generates thumbnails remotely via SSH
- Builds multi-frame GIFs
- Runs motion detection (`detect_mover_gif.py`)
- Aggregates photometry & metadata
- Produces PPT and CSV summary reports

Run manually:
```bash
python daily_workflow.py
```
To schedule automatic runs, use Windows Task Scheduler or Linux `cron`.

### 1b. `AP_workflow.py`
Optional follow-up once `daily_workflow.py` finds candidates. Uses remote WCS lookups and optional forced photometry to build submission-ready products.

Run manually after a successful daily run:
```bash
python AP_workflow.py --obs-date YYYY-MM-DD
```
Key inputs are the daily output table/GIFs; see script args for filter options.

---

### 2. `detect_mover_gif.py`
Detects moving objects in 4-frame or multi-frame GIFs using a combination of image differencing, trail fitting, and prior-angle filtering.  
Key options:
```bash
python detect_mover_gif.py <input.gif>     --vmax 50 (length pred from speed, unit in pix) --step 0.5 --prior-angle-deg 45 (x to right 0, y to downward 90) --trail-filter    --subframes 4 --fwhm-px 3.0 --empirical-p (statistically comparing snr from target to whole image)
```
Output: if detection, an annotated gif and a stacked image; if not only a stacked image.

---

### 3. `autochecker_MPClist_pipeline.py`
Fetches MPC object lists, generates ephemerides, and cross-matches with GOTO observations.  
Features:
- Selenium automation for MPC "Customize" page scraping  
- Ephemeris generation via `astroquery.mpc`  
- Footprint intersection using `shapely` polygons  
- Optional Gaussian Process interpolation for smooth motion curves
- request accurate epherm from JPL using obs_time from matched result

Run manually:
```bash
python autochecker_MPClist_pipeline.py
```
Outputs include CSV tables (`YYYY-MM-DD_MPC.csv`) and diagnostic logs.

---

### 4. `run_client_batch.py`
Handles remote execution and file transfer via SSH:
- Connects to Warwick or Monash compute nodes (only tested on Warwick)
- Runs `make_thumbnails_batch.py` remotely (placed on server)
- Downloads PNGs and assemble GIFs localy

### 4b. `run_client_fits_cutouts_batch.py`
Similar to `run_client_batch.py`, but requests FITS cutouts instead of PNG thumbnails. Requires the server-side `make_fits_cutouts_batch.py`.

---

### 5. `pill_matched_phot.py`
Performs forced aperture (“pill”) photometry on detected trails, calibrated using Gaia stars.

Run manually:
```bash
python pill_matched_phot.py <raw_image.fits> --ra (deg) --dec (deg)
```

---

### 6. `build_daily_ppt.py`
Creates a PowerPoint summary (`daily_report_<date>.pptx`) combining GIFs, measurement tables, and metadata for quick review and archiving.

---

### 7. `target_searching.py`
Fetches ephemerides for user-defined objects, builds dense (1-min) interpolated tracks, loads recent GOTO observations, and checks which ephemeris points fall inside image footprints within a given time tolerance. Request JPL giving the accurate ephemerides point for the result at end.

Run manually:
```bash
# Single target, both sites, 3-day window
python target_searching.py --targets "2025 TK8" --days 3 --time-tol 60 --outfile ./result/2025-10-22-2025TF.csv --jpl-refine

# Multiple targets (comma separated)
python target_searching.py --targets "2025 TK8, 3I/ATLAS" --days 90 --site SSO --outfile ./result/2025-10-22.csv
```

---

## Server-side utilities (Warwick)
The following scripts should live on the Warwick server side and are invoked remotely by the client scripts:

- `read_wcs_server.py`: Reads WCS headers from FITS files on the server and returns JSON. Example:
  ```bash
  python read_wcs_server.py --hdu 1 /data/run1234/*.fits > wcs.json
  ```
- `make_thumbnails_batch.py`: Generates PNG thumbnails (HDU[1], ZScale stretch) for given RA/Dec and saves them server-side.
  ```bash
  python make_thumbnails_batch.py --ra 123.45 --dec -12.34 --size 120 --outdir /tmp/thumbs /data/run1234/*.fits
  ```
- `make_fits_cutouts_batch.py`: Builds FITS cutouts around RA/Dec and writes them to an output directory for later download.
  ```bash
  python make_fits_cutouts_batch.py --ra 123.45 --dec -12.34 --size 120 --outdir /tmp/cutouts /data/run1234/*.fits
  ```

Both `run_client_batch.py` and `run_client_fits_cutouts_batch.py` expect these utilities to be accessible via SSH on the Warwick host configured in `~/.ssh/config`.

## Configuration
Network access for Warwick-side utilities relies on SSH key authentication:

- Ensure you have an SSH key (e.g., `~/.ssh/id_rsa`) and that the public key is added to the Warwick account authorized keys.
- Configure `~/.ssh/config` with host, user, and port information, for example:
  ```
    Host warwick-gw
        HostName goto-observatory.warwick.ac.uk
        User <username>
        IdentityFile <ssh key path>
        IdentitiesOnly yes
    
    # Used by AP_workflow / FitsCutoutClient when server="warwick"
    Host gotohead
        HostName gotocompute4
        User <username>
        IdentityFile <ssh key path>
        IdentitiesOnly yes
        ProxyCommand ssh -W gotocompute4:22 warwick-gw
        StrictHostKeyChecking accept-new
    
    # Optional: for interactive login: `ssh gotocompute4`
    # (Same settings as gotohead so you can use either name)
    Host gotocompute4
        HostName gotocompute4
        User <username>
        IdentityFile <ssh key path>
        IdentitiesOnly yes
        ProxyCommand ssh -W gotocompute4:22 warwick-gw
        StrictHostKeyChecking accept-new

  ```
- The client scripts (`run_client_batch.py`, `run_client_fits_cutouts_batch.py`) use this host alias; test with `ssh warwick "hostname"` before running the workflows.

Database credentials are read from `~/.pgpass`.

## Output
After each daily run:
```
/thumbs/YYYY-MM-DD_MPC/
    ├── object1__SSO__sub00.gif
    ├── object1__LMO__sub01.gif
    ├── object2__LMO__sub00.gif
    ...
/out/YYYY-MM-DD_MPC.csv
/daily_report_<date>.pptx
```

---

## Configuration
SSH credentials and server routes are resolved via `~/.ssh/config`.  
Database credentials are read from `~/.pgpass`.  
You can modify runtime behavior using CLI arguments in each module.

---
