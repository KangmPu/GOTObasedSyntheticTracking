#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pill-aperture forced photometry with iterative center/shape search,
growth-curve refinement ("brakes"), and limited in-window Gaia/peak masks.

Jupyter Notebook:

fits =["t1_r0867736_ut8.fits",
        "t1_r0867737_ut8.fits",
        "t1_r0867738_ut8.fits",
        "t1_r0867739_ut8.fits"]
    
ra_list = [54.01876678347303, 54.02030468057942, 54.022812624926004,54.0256280672828]
dec_list = [16.264615961776162, 16.268309446943775, 16.273664650316398, 16.278735344513695]
zp_list = [21.6,21.6,21.6,21.6]  #can be found using zp_query for each ra,dec for specific see photom_tool_ing
fwhm_pix = [3,3,3,3]

results = run_pill_photometry_4images(
    fits_paths=fits,
    ra_list=ra_list,
    dec_list=dec_list,
    half_size=phot_half_size,
    out4_path=str(local_dir / fourpanel_outpng),
    use_template=True,
    debug=False,
    zero_point_list=zp_list, 
    # fwhm_pix=fwhm_pix,
)       

Notes:
- a, b parameters represent semi-length and semi-width respectively
  (i.e., L = 2a, W = 2b).
- all fits should use the same center(for align image)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import map_coordinates
from astroquery.vizier import Vizier
from types import SimpleNamespace
import os
import tempfile
from matplotlib.patches import PathPatch
# -------- tool kit --------
def create_psf_profile_debug_plot(raw_win, cy, cx, a, b, angle_deg, 
                                  out_path=None, title_prefix=""):
    """创建展示主轴和副轴PSF拟合的debug图"""
    from matplotlib.gridspec import GridSpec
    from scipy.optimize import curve_fit
    
    # 构建剖面
    s_centers, F_s = build_major_axis_profile(
        raw_win, cy, cx, a, b, angle_deg,
        xspan_pix=None, binw=1.0
    )
    
    t_centers, F_t = build_minor_axis_profile(
        raw_win, cy, cx, a, b, angle_deg,
        s_center=0.0, s_halfwidth=None,
        tspan_pix=None, binw=1.0
    )
    
    # 创建2x2布局
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 左上: 原始图像 + pill轮廓 + 主副轴方向
    ax1 = fig.add_subplot(gs[0, 0])
    vmin, vmax = np.nanpercentile(raw_win, [2, 98])
    ax1.imshow(raw_win, cmap='gray', origin='upper', vmin=vmin, vmax=vmax)
    
    # 绘制pill轮廓
    x_outline, y_outline = _pill_outline_xy(cx, cy, a, b, angle_deg)
    ax1.plot(x_outline, y_outline, 'c-', lw=2, alpha=0.9, label='Aperture')
    ax1.plot(cx, cy, 'c+', ms=12, mew=2, label='Center')
    
    # 绘制主轴方向（红色箭头）
    th_rad = np.deg2rad(angle_deg)
    dx_major = a * np.cos(th_rad)
    dy_major = a * np.sin(th_rad)
    ax1.arrow(cx, cy, dx_major, dy_major, head_width=2, head_length=2, 
             fc='red', ec='red', alpha=0.7, label='Major axis')
    
    # 绘制副轴方向（黄色箭头）
    dx_minor = -b * np.sin(th_rad)
    dy_minor = b * np.cos(th_rad)
    ax1.arrow(cx, cy, dx_minor, dy_minor, head_width=2, head_length=2,
             fc='yellow', ec='yellow', alpha=0.7, label='Minor axis')
    
    ax1.set_title(f'{title_prefix}Raw Window + Aperture')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.axis('off')
    
    # 右上: 主轴光度剖面 + 拟合
    ax2 = fig.add_subplot(gs[0, 1])
    valid = np.isfinite(F_s)
    if np.any(valid):
        ax2.plot(s_centers[valid], F_s[valid], 'b.-', alpha=0.7, 
                label='Measured', markersize=3)
        
        # 拟合
        try:
            s0, sigma, A, C, F_fit = fit_midpoint_from_profile(
                s_centers, F_s, a, b, smooth=True
            )
            ax2.plot(s_centers[valid], F_fit[valid], 'r-', lw=2, alpha=0.8,
                    label=f'Fit (s₀={s0:.2f}, σ={sigma:.2f})')
            ax2.axvline(s0, color='green', ls='--', alpha=0.5, label='Center')
        except Exception as e:
            print(f"[debug] Major axis fit failed: {e}")
        
        ax2.axvline(-a, color='gray', ls=':', alpha=0.5)
        ax2.axvline(a, color='gray', ls=':', alpha=0.5)
        ax2.set_xlabel('Position along major axis (pixels)')
        ax2.set_ylabel('Integrated flux')
        ax2.set_title('Major Axis Profile')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    # 左下: 副轴光度剖面 + Gaussian拟合
    ax3 = fig.add_subplot(gs[1, 0])
    valid = np.isfinite(F_t)
    if np.any(valid):
        ax3.plot(t_centers[valid], F_t[valid], 'b.-', alpha=0.7,
                label='Measured', markersize=3)
        
        # Gaussian拟合
        try:
            def gaussian(x, amp, mu, sig, offset):
                return amp * np.exp(-0.5 * ((x - mu) / sig)**2) + offset
            
            x_data = t_centers[valid]
            y_data = F_t[valid]
            
            amp_guess = np.max(y_data) - np.min(y_data)
            popt, _ = curve_fit(gaussian, x_data, y_data,
                               p0=[amp_guess, 0.0, b, np.min(y_data)],
                               maxfev=2000)
            
            y_fit = gaussian(t_centers[valid], *popt)
            ax3.plot(t_centers[valid], y_fit, 'r-', lw=2, alpha=0.8,
                    label=f'Gaussian (μ={popt[1]:.2f}, σ={popt[2]:.2f})')
            ax3.axvline(popt[1], color='green', ls='--', alpha=0.5, label='Center')
        except Exception as e:
            print(f"[debug] Minor axis fit failed: {e}")
        
        ax3.axvline(-b, color='gray', ls=':', alpha=0.5)
        ax3.axvline(b, color='gray', ls=':', alpha=0.5)
        ax3.set_xlabel('Position along minor axis (pixels)')
        ax3.set_ylabel('Integrated flux')
        ax3.set_title('Minor Axis Profile')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    
    # 右下: 参数信息
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    info_text = f"""
Aperture Parameters:
━━━━━━━━━━━━━━━━━━━━━━
Center: ({cx:.2f}, {cy:.2f})
Semi-length a: {a:.2f} px
Semi-width b: {b:.2f} px
Angle: {angle_deg:.2f}°
Aspect ratio: {(2*a)/(2*b):.2f}

Profile Statistics:
━━━━━━━━━━━━━━━━━━━━━━
Major axis points: {np.sum(np.isfinite(F_s))}
Minor axis points: {np.sum(np.isfinite(F_t))}
    """
    
    ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes,
            fontsize=10, family='monospace', verticalalignment='center')
    
    plt.suptitle(f'{title_prefix}PSF Profile Analysis', 
                fontsize=14, fontweight='bold')
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"[debug] PSF profile plot saved to: {out_path}")
    
    return fig

def draw_aperture_overlay(ax, x_center, y_center, a, b, angle_deg, 
                         shape, color='cyan', linewidth=1.8, alpha=0.95):
    """
    在axis上绘制pill孔径轮廓和背景环
    统一使用(x_center, y_center)坐标系统
    直接使用测光mask，不重新创建轮廓
    """
    # 直接使用测光时的pill_mask - pill_mask使用(y, x)顺序
    ap_mask = pill_mask(shape, y_center, x_center, a, b, angle_deg)
    # 沿y轴翻转孔径mask以修正显示
    ap_mask_flipped = np.flipud(ap_mask)
    # 使用contour绘制孔径边界
    ax.contour(ap_mask_flipped, levels=[0.5], origin="upper", colors=color, 
              linewidths=linewidth, alpha=alpha)
    
    # 绘制背景环 - pill_annulus_mask使用(y, x)顺序
    BG_IN_SCALE, BG_OUT_SCALE = 1.3, 1.8
    bg_mask = pill_annulus_mask(shape, y_center, x_center, a, b, angle_deg, 
                               BG_IN_SCALE, BG_OUT_SCALE)
    # 沿y轴翻转背景环mask以修正显示
    bg_mask_flipped = np.flipud(bg_mask)
    ax.contour(bg_mask_flipped, levels=[0.5], origin="upper", colors=color, 
              linewidths=1, linestyles="--", alpha=0.7)
    
    # 绘制十字准星
    ax.plot([x_center-8, x_center+8], [y_center, y_center], 
           color=color, lw=0.9, alpha=0.9)
    ax.plot([x_center, x_center], [y_center-8, y_center+8], 
           color=color, lw=0.9, alpha=0.9)
# -------- helpers: display stretch & Epill correction --------

def _robust_sigma(x):
    import numpy as np
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(x)

def _asinh_stretch(img):
    import numpy as np
    med = np.nanmedian(img)
    sig = _robust_sigma(img)
    if not np.isfinite(sig) or sig <= 0:
        sig = np.nanstd(img)
        if not np.isfinite(sig) or sig <= 0:
            sig = 1.0
    s = 3.0 * sig
    arr = np.arcsinh((img - med) / s)
    vmin, vmax = np.nanpercentile(arr, [0.5, 99.5])
    return arr, float(vmin), float(vmax)


# -------- helpers: display stretch & Epill correction --------
def _fourier_shift2d_smooth(img, dx, dy, edge_taper=5):
    """改进的Fourier shift，边缘平滑处理"""
    # 添加边缘taper减少边界效应
    H, W = img.shape
    if edge_taper > 0:
        taper_y = np.ones(H)
        taper_x = np.ones(W)
        
        for i in range(edge_taper):
            w = (i + 1) / edge_taper
            taper_y[i] = w
            taper_y[-(i+1)] = w
            taper_x[i] = w
            taper_x[-(i+1)] = w
        
        taper_2d = np.outer(taper_y, taper_x)
        img_tapered = img * taper_2d + np.nanmedian(img) * (1 - taper_2d)
    else:
        img_tapered = img
    
    # 标准Fourier shift
    F = np.fft.rfftn(img_tapered)
    ny, nx = img.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.rfftfreq(nx)[None, :]
    phase = np.exp(-2j * np.pi * (kx*dx + ky*dy))
    shifted = np.fft.irfftn(F * phase, s=img.shape).real
    
    return shifted


def _align_windows_robust(raw_wins, target_masks=None, search_radius=20.0, debug=False):
    """改进的窗口对齐，排除目标区域"""
    n = len(raw_wins)
    if n == 0: return [], []
    if n == 1: return [raw_wins[0]], [(0.0, 0.0)]
    
    ref = raw_wins[0].copy()
    
    # Mask掉目标区域用于对齐
    if target_masks is not None and len(target_masks) > 0:
        ref_for_align = ref.copy()
        ref_for_align[target_masks[0]] = np.nanmedian(ref)
    else:
        ref_for_align = ref
    
    aligned = [ref]
    shifts = [(0.0, 0.0)]
    
    for k in range(1, n):
        tgt = raw_wins[k].copy()
        
        if target_masks is not None and len(target_masks) > k:
            tgt_for_align = tgt.copy()
            tgt_for_align[target_masks[k]] = np.nanmedian(tgt)
        else:
            tgt_for_align = tgt
        
        # 相位相关
        dx_pc, dy_pc = _phase_corr_shift2d(ref_for_align, tgt_for_align)
        
        # 亮星微调
        ref_pts = _detect_bright_peaks(ref_for_align)
        tgt_pts = _detect_bright_peaks(tgt_for_align)
        
        if ref_pts.size and tgt_pts.size:
            dx_pk, dy_pk = _estimate_shift_by_peaks(ref_pts, tgt_pts, 
                                                    search_radius=search_radius)
        else:
            dx_pk, dy_pk = 0.0, 0.0
        
        if abs(dx_pk) > search_radius or abs(dy_pk) > search_radius:
            dx, dy = dx_pc, dy_pc
        else:
            dx = 0.7*dx_pc + 0.3*dx_pk
            dy = 0.7*dy_pc + 0.3*dy_pk
        
        dx = float(np.clip(dx, -search_radius, search_radius))
        dy = float(np.clip(dy, -search_radius, search_radius))
        
        win_shifted = _fourier_shift2d_smooth(tgt, -dx, -dy)
        aligned.append(win_shifted)
        shifts.append((dx, dy))
        
        if debug:
            print(f"[align] frame={k}: phase({dx_pc:.2f},{dy_pc:.2f}) "
                  f"peaks({dx_pk:.2f},{dy_pk:.2f}) -> use({dx:.2f},{dy:.2f})")
    
    return aligned, shifts


def generate_template_with_mask(aligned_wins, target_masks=None, method='median'):
    """生成模板，排除目标区域"""
    n = len(aligned_wins)
    if n == 0:
        return None
    
    H, W = aligned_wins[0].shape
    
    if target_masks is None or len(target_masks) == 0:
        stack = np.stack(aligned_wins, axis=0)
        if method == 'median':
            return np.nanmedian(stack, axis=0)
        else:
            return np.nanmean(stack, axis=0)
    
    # 逐像素处理，排除目标区域
    template = np.zeros((H, W), dtype=np.float32)
    
    for y in range(H):
        for x in range(W):
            values = []
            for i in range(n):
                if i >= len(target_masks) or not target_masks[i][y, x]:
                    values.append(aligned_wins[i][y, x])
            
            if len(values) > 0:
                if method == 'median':
                    template[y, x] = np.nanmedian(values)
                else:
                    template[y, x] = np.nanmean(values)
            else:
                template[y, x] = np.nan
    
    # 填充NaN
    mask_nan = np.isnan(template)
    if np.any(mask_nan):
        from scipy.ndimage import median_filter
        filled = median_filter(np.nan_to_num(template), size=5)
        template[mask_nan] = filled[mask_nan]
    
    return template

def epill_throughput(a, b, fwhm_pix, L_trail_pix=None, sigma_s=None, sigma_t=None):
    """
    Estimate pill aperture throughput fraction Epill in [0,1].
    - a, b: semi-length/width (pixels) of the pill aperture.
    - fwhm_pix: FWHM of the seeing PSF (pixels). Used if sigma_s/t not provided.
    - L_trail_pix: trail length (pixels). If provided, along-track profile is modeled
      as a uniform line of length L convolved with a Gaussian and integrated inside [-a, a].
      Otherwise we fall back to pure Gaussian along-track.
    Returns: Epill (0..1), dm = -2.5*log10(Epill)
    """
    import numpy as np
    import math
    from math import erf, sqrt, log10  # << 确保 log10 在函数作用域内

    if sigma_t is None or sigma_s is None:
        sig = float(fwhm_pix) / 2.354820045  # 2*sqrt(2*ln2)
        if sigma_t is None: sigma_t = sig
        if sigma_s is None: sigma_s = sig

    # cross-track Gaussian fraction inside [-b, b]
    Et = float(erf(b / (sqrt(2.0) * sigma_t)))

    # along-track fraction
    if L_trail_pix is not None and np.isfinite(L_trail_pix) and L_trail_pix > 0:
        L = float(L_trail_pix)
        # p(s) = (1/L) * [Phi((s+L/2)/sigma_s) - Phi((s-L/2)/sigma_s)], integrate Es = ∫_{-a}^{a} p(s) ds
        s = np.linspace(-a, a, int(max(400, 8*a)))
        invrt2 = 1.0 / sqrt(2.0)
        def _Phi(z):  # z already scaled by sigma_s
            return 0.5 * (1.0 + erf(z * invrt2))
        Es = (1.0 / L) * np.trapz((_Phi((s + L/2.0) / sigma_s) - _Phi((s - L/2.0) / sigma_s)), s)
    else:
        # Fallback: pure Gaussian along-track
        Es = float(erf(a / (sqrt(2.0) * sigma_s)))

    E = float(max(1e-6, min(1.0, Et * Es)))
    dm = -2.5 * log10(E)
    return E, dm


import math
from math import erf

from skimage.filters import gaussian
from skimage import img_as_float
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def _pill_outline_xy(cx, cy, a, b, angle_deg, n=64):
    """
    在“数组坐标”（x 向右，y 向下）生成胶囊轮廓 (x[], y[])。
    a,b 为半长/半宽 (px)，angle_deg 相对 +x 轴的角度（数组坐标，逆时针为正）。
    """
    import numpy as np
    a = float(a); b = float(b)
    r = max(1e-6, float(b))

    # 本地坐标：主轴沿 x'，副轴沿 y'
    t = np.linspace(-np.pi/2, np.pi/2, n, endpoint=True)
    # 左帽中心 (-a,0)，从下到上走一半圆
    x1 = -a + r * np.cos(t)
    y1 =      r * np.sin(t)
    # 右帽中心 (+a,0)，从上到下走一半圆（保证外轮廓不“内凹”）
    x2 =  a + r * np.cos(np.pi - t)
    y2 =      r * np.sin(np.pi - t)
    # 连接两侧直边
    side = np.linspace(r, -r, n, endpoint=True)
    xl = -a * np.ones_like(side); yl = side
    xr =  a * np.ones_like(side); yr = -side

    x_local = np.concatenate([x1, xr, x2, xl])
    y_local = np.concatenate([y1, yr, y2, yl])

    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    # 数组坐标旋转（y 向下）：依旧用标准旋转矩阵即可
    x = cx + x_local * c - y_local * s
    y = cy + x_local * s + y_local * c
    return x, y


# -------- alignment helpers (robust): phase correlation + bright-peak refine --------
def _phase_corr_shift2d(ref, tgt):
    """
    纯 numpy 实现的相位相关（像素级）：返回 (dx, dy)，把 tgt 平移 (dx,dy) 后对齐到 ref
    约定：dx>0 往 +x(列) 方向移，dy>0 往 +y(行) 方向移
    """
    import numpy as np
    R = np.fft.rfftn(ref)
    T = np.fft.rfftn(tgt)
    CPS = R * np.conj(T)
    denom = np.abs(CPS)
    denom[denom == 0] = 1.0
    CPS /= denom
    cc = np.fft.irfftn(CPS, s=ref.shape).real
    # 峰位置
    y0, x0 = np.unravel_index(np.argmax(cc), cc.shape)
    H, W = ref.shape
    if x0 > W//2: x0 = x0 - W
    if y0 > H//2: y0 = y0 - H
    # 注意：ifft 峰的 (x0,y0) 表示把 tgt 向 (x0,y0) 平移使之对齐到 ref
    return float(x0), float(y0)

def _detect_bright_peaks(win, nsig=5.0, max_peaks=40, footprint=5):
    import numpy as np
    try:
        from scipy.ndimage import maximum_filter, gaussian_filter
    except Exception:
        def maximum_filter(a, size): return a
        def gaussian_filter(a, sigma, mode="nearest"): return a
    arr = np.asarray(win, dtype=float)
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    sig = 1.4826*mad if np.isfinite(mad) and mad > 0 else np.nanstd(arr)
    if not np.isfinite(sig) or sig <= 0: sig = 1.0
    thr = med + nsig*sig
    sm  = gaussian_filter(arr, sigma=1.0, mode="nearest")
    mx  = maximum_filter(sm, size=int(max(3, footprint)))
    ys, xs = np.nonzero((sm == mx) & (sm > thr))
    vals = sm[ys, xs]
    if len(vals) == 0: return np.empty((0,2), float)
    order = np.argsort(vals)[::-1][:max_peaks]
    xs = xs[order]; ys = ys[order]
    return np.stack([xs.astype(float), ys.astype(float)], axis=1)

def _estimate_shift_by_peaks(ref_pts, tgt_pts, bin_size=1.0, search_radius=20.0):
    import numpy as np
    if ref_pts.size == 0 or tgt_pts.size == 0: return 0.0, 0.0
    rx, ry = ref_pts[:,0], ref_pts[:,1]
    tx, ty = tgt_pts[:,0], tgt_pts[:,1]
    dxs, dys = [], []
    for x, y in zip(rx, ry):
        d2 = (tx - x)**2 + (ty - y)**2
        j  = int(np.argmin(d2))
        d  = float(np.sqrt(d2[j]))
        if d <= float(search_radius):
            dxs.append(tx[j] - x); dys.append(ty[j] - y)
    if not dxs: return 0.0, 0.0
    dxs = np.array(dxs); dys = np.array(dys)
    bins = int(np.ceil(2*search_radius / bin_size))
    H2, xedges, yedges = np.histogram2d(dxs, dys, bins=bins,
        range=[[-search_radius, search_radius], [-search_radius, search_radius]])
    i, j = np.unravel_index(np.argmax(H2), H2.shape)
    dx0 = 0.5*(xedges[i] + xedges[i+1]); dy0 = 0.5*(yedges[j] + yedges[j+1])
    m = (np.abs(dxs-dx0)<=bin_size) & (np.abs(dys-dy0)<=bin_size)
    if np.count_nonzero(m) >= 3:
        return float(np.median(dxs[m])), float(np.median(dys[m]))
    return float(dx0), float(dy0)

def _fourier_shift2d(img, dx, dy):
    import numpy as np
    F = np.fft.rfftn(img)
    ny, nx = img.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.rfftfreq(nx)[None, :]
    phase = np.exp(-2j * np.pi * (kx*dx + ky*dy))
    return np.fft.irfftn(F * phase, s=img.shape).real


def _align_windows(raw_wins, search_radius=20.0, debug=False):
    """
    Align N windows to the first window using phase correlation (robust) and
    optional bright-peak refinement. Returns (aligned_wins, shifts).
    shifts[i] is (dx,dy) such that shifting aligned_wins[i] by (+dx,+dy) approximates raw_wins[i].
    """
    import numpy as np
    n = len(raw_wins)
    if n == 0: return [], []
    if n == 1: return [raw_wins[0]], [(0.0, 0.0)]
    ref = raw_wins[0]
    aligned = [ref]; shifts = [(0.0, 0.0)]
    for k in range(1, n):
        tgt = raw_wins[k]
        dx_pc, dy_pc = _phase_corr_shift2d(ref, tgt)
        ref_pts = _detect_bright_peaks(ref); tgt_pts = _detect_bright_peaks(tgt)
        if ref_pts.size and tgt_pts.size:
            dx_pk, dy_pk = _estimate_shift_by_peaks(ref_pts, tgt_pts, search_radius=search_radius)
        else:
            dx_pk, dy_pk = 0.0, 0.0
        if abs(dx_pk) > search_radius or abs(dy_pk) > search_radius:
            dx, dy = dx_pc, dy_pc
        else:
            dx = 0.7*dx_pc + 0.3*dx_pk
            dy = 0.7*dy_pc + 0.3*dy_pk
        dx = float(np.clip(dx, -search_radius, search_radius))
        dy = float(np.clip(dy, -search_radius, search_radius))
        win_shifted = _fourier_shift2d(tgt, -dx, -dy)
        aligned.append(win_shifted)
        shifts.append((dx, dy))
        if debug:
            print(f"[align] frame={k}: phase({dx_pc:.2f},{dy_pc:.2f}) peaks({dx_pk:.2f},{dy_pk:.2f}) -> use({dx:.2f},{dy:.2f})")
    return aligned, shifts

    for k in range(1, 4):
        tgt = raw_wins[k]
        # 先相位相关（稳）
        dx_pc, dy_pc = _phase_corr_shift2d(ref, tgt)
        # 再用亮源微调
        ref_pts = _detect_bright_peaks(ref)
        tgt_pts = _detect_bright_peaks(tgt)
        dx_pk, dy_pk = _estimate_shift_by_peaks(ref_pts, tgt_pts, search_radius=search_radius) if (ref_pts.size and tgt_pts.size) else (0.0, 0.0)
        dx = dx_pc if abs(dx_pk) > search_radius or abs(dy_pk) > search_radius else 0.7*dx_pc + 0.3*dx_pk
        dy = dy_pc if abs(dx_pk) > search_radius or abs(dy_pk) > search_radius else 0.7*dy_pc + 0.3*dy_pk

        # 限幅，避免奇异解
        dx = float(np.clip(dx, -search_radius, search_radius))
        dy = float(np.clip(dy, -search_radius, search_radius))

        # 把目标移到参考：用 (-dx,-dy)
        win_shifted = _fourier_shift2d(tgt, -dx, -dy)
        aligned.append(win_shifted)
        shifts.append((dx, dy))
        if debug:
            print(f"[align] k={k}: phase({dx_pc:.2f},{dy_pc:.2f}) peaks({dx_pk:.2f},{dy_pk:.2f}) -> use({dx:.2f},{dy:.2f})")
    return aligned, shifts


# -------- helpers: display stretch & Epill correction --------




    if sigma_t is None or sigma_s is None:
        sig = float(fwhm_pix) / 2.354820045  # 2*sqrt(2*ln2)
        if sigma_t is None: sigma_t = sig
        if sigma_s is None: sigma_s = sig

    # cross-track Gaussian fraction inside [-b, b]
    Et = float(erf(b / (sqrt(2.0) * sigma_t)))

    # along-track fraction
    if L_trail_pix is not None and np.isfinite(L_trail_pix) and L_trail_pix > 0:
        L = float(L_trail_pix)
        # p(s) = (1/L) * [Phi((s+L/2)/sigma_s) - Phi((s-L/2)/sigma_s)]
        # We'll integrate numerically Es = ∫_{-a}^{a} p(s) ds (p is normalized).
        smax = max(a + L/2.0 + 6.0*sigma_s, 10.0)
        s = np.linspace(-a, a, int(max(400, 8*a)))
        # Standard normal CDF with std=sigma_s
        invrt2 = 1.0 / sqrt(2.0)
        def _Phi(z):  # z already scaled by sigma_s
            return 0.5 * (1.0 + erf(z * invrt2))
        Es = (1.0 / L) * np.trapz((_Phi((s + L/2.0) / sigma_s) - _Phi((s - L/2.0) / sigma_s)), s)
    else:
        # Fallback: pure Gaussian along-track
        Es = float(erf(a / (sqrt(2.0) * sigma_s)))

    E = float(max(1e-6, min(1.0, Et * Es)))
    dm = -2.5 * math.log10(E)
    return E, dm

# =================== Defaults (can be overridden via CLI) ===================
# File & target
DEF_FILE_PATH = 't3_r0212283_ut5.fits'
DEF_RA_DEG, DEF_DEC_DEG = 302.03729, -4.319045

# Cropping & preprocessing
HALF_SIZE = 30
SIGMA_SMOOTH = 0.2
NORM_MODE   = "zscore"    # "zscore" | "minmax" | "none"

# Iteration grids (coarse -> fine)
ANGLE_GRID = [
    np.arange(0.0, 180.0, 3.0),
    np.arange(0.0, 180.0, 1.0),
    np.arange(0.0, 180.0, 0.2),
    np.arange(0.0, 180.0, 0.05),
]
L_GRID     = [np.arange(10, 81, 6),  np.arange(10, 81, 4),  np.arange(10, 81, 2)]
W_GRID     = [np.array([2,3,4,5,6], float), np.array([2,3,4,5], float), np.array([2.5,3,3.5,4], float)]

REFINE_RADIUS = [4.0, 2.5, 1.5]
REFINE_STEP   = [0.5,  0.33, 0.25]

# Background & thresholds
BG_IN_SCALE, BG_OUT_SCALE = 1.3, 1.8
FALLBACK_RING_THICK = 8
MIN_AP_PIX, MIN_BG_PIX = 10, 80

# Convergence & limits
POS_TOL = 0.1
SNR_TOL = 0.05
MAX_ITERS = 4

# Growth-curve refinement
C_SCALES   = np.linspace(0.7, 1.6, 11)
C_EPS      = 0.05
C_SMOOTH_K = 3

# Elongation braking
TAU_RING   = 1.0
K_CONSEC   = 2
SYM_RATIO_MAX = 3.0
PHYS_LMAX_PIX = None

TRIM_FRAC_AP = 0.02  # trim brightest top fraction in aperture

# --------------------------- Utility functions -----------------------------
def pick_image_hdu(hdul):
    for idx in [1, 0]:
        if idx < len(hdul) and getattr(hdul[idx], "data", None) is not None:
            data = hdul[idx].data
            if data is not None and data.ndim == 2:
                try:
                    wcs = WCS(hdul[idx].header)
                    hdr = hdul[idx].header
                    if 'DATE-MID' in hdr:
                        print(hdr['DATE-MID'])
                except Exception:
                    wcs = None
                return hdul[idx], wcs
    raise RuntimeError("Could not find a 2D image HDU.")

def safe_crop(img, xc, yc, half):
    H, W = img.shape
    x_min, x_max = int(xc - half), int(xc + half)
    y_min, y_max = int(yc - half), int(yc + half)
    if x_min < 0 or y_min < 0 or x_max > W or y_max > H:
        raise ValueError("Cropping window out of bounds; increase --half-size or check coordinates.")
    return img[y_min:y_max, x_min:x_max], x_min, y_min

def preprocess_window(win, sigma=1.0, norm="zscore"):
    w = win.astype(np.float32)
    w = w - np.nanmedian(w)
    w = gaussian(w, sigma=sigma, preserve_range=True)
    w = img_as_float(w)
    if norm == "zscore":
        m, s = np.nanmean(w), np.nanstd(w)
        w = (w - m) / s if s > 0 else w * 0.0
    elif norm == "minmax":
        mn, mx = float(np.nanmin(w)), float(np.nanmax(w))
        if mx > mn:
            w = (w - mn) / (mx - mn)
    return w

def circular_annulus(shape, cy, cx, r_in, r_out):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    rr = np.hypot(yy - cy, xx - cx)
    return (rr >= r_in) & (rr <= r_out)

def robust_bg_stats(arr):
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    sigma = 1.4826 * mad
    if sigma <= 0:
        sigma = np.std(arr)
    return med, sigma

# ------------------------------ Pill aperture ------------------------------
def _rotated_coords(shape, cy, cx, angle_deg):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    dy = yy - cy
    dx = xx - cx
    th = np.deg2rad(angle_deg)
    ct, st = np.cos(th), np.sin(th)
    # Major axis S, minor axis T
    S =  dx * ct + dy * st
    T = -dx * st + dy * ct
    return S, T, ct, st

def pill_mask(shape, cy, cx, a, b, angle_deg):
    """
    Pill mask defined by semi-length a (=L/2) and semi-width b (=W/2).
    If a <= b, degenerates into a circle of radius a (fully capped).
    """
    S, T, _, _ = _rotated_coords(shape, cy, cx, angle_deg)
    core = max(0.0, a - b)
# length of half-rectangle along S (non-negative)
    r = b

    rect = (np.abs(T) <= r) & (np.abs(S) <= core)
    # Two semicircular caps
    right_cap = ((S - core)**2 + T**2) <= (r*r)
    left_cap  = ((S + core)**2 + T**2) <= (r*r)
    return rect | right_cap | left_cap

def pill_annulus_mask(shape, cy, cx, a, b, angle_deg, in_scale, out_scale):
    inner = pill_mask(shape, cy, cx, a*in_scale, b*in_scale, angle_deg)
    outer = pill_mask(shape, cy, cx, a*out_scale, b*out_scale, angle_deg)
    return outer & (~inner)

def forced_photometry(im, cy, cx, a, b, angle_deg,
                      bg_in_scale=1.3, bg_out_scale=1.8,
                      min_ap_pix=10, min_bg_pix=80,
                      fallback_ring_thick=8, use_fallback=True,
                      bad_mask=None, trim_frac=0.0):
    ap = pill_mask(im.shape, cy, cx, a, b, angle_deg)
    bg = pill_annulus_mask(im.shape, cy, cx, a, b, angle_deg, bg_in_scale, bg_out_scale)

    if bad_mask is not None:
        ap &= (~bad_mask)
        bg &= (~bad_mask)

    if ap.sum() < min_ap_pix or bg.sum() < min_bg_pix:
        if use_fallback:
            yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
            rr = np.hypot(yy - cy, xx - cx)
            r0 = max(a, b) * bg_in_scale
            bg = (rr >= r0) & (rr <= r0 + float(fallback_ring_thick))
            if bad_mask is not None:
                bg &= (~bad_mask)

    if ap.sum() < min_ap_pix or bg.sum() < min_bg_pix:
        return np.nan, np.nan, np.nan, int(ap.sum()), int(bg.sum())

    ap_vals = im[ap]
    bg_vals = im[bg]
    mu_bg, sig_bg = robust_bg_stats(bg_vals)
    ap_sub = ap_vals - mu_bg

    if trim_frac and trim_frac > 0.0:
        kth = np.nanquantile(ap_sub, 1.0 - min(0.49, float(trim_frac)))
        ap_sub = ap_sub[ap_sub <= kth]

    nap_eff = ap_sub.size
    if nap_eff < min_ap_pix:
        return np.nan, np.nan, np.nan, int(ap.sum()), int(bg.sum())

    flux = float(np.sum(ap_sub))
    err  = max(sig_bg, 1e-6) * np.sqrt(nap_eff)
    snr  = flux / err
    return flux, err, snr, int(nap_eff), int(bg.sum())

def subpixel_peak_quadratic(Z):
    if Z.shape != (3,3) or not np.isfinite(Z).all():
        return 0.0, 0.0
    ys, xs = np.indices((3,3)) - 1  # -1,0,1
    A = np.column_stack([xs.ravel()**2, ys.ravel()**2, (xs*ys).ravel(),
                         xs.ravel(), ys.ravel(), np.ones(9)])
    try:
        coeff, *_ = np.linalg.lstsq(A, Z.ravel(), rcond=None)
        a, b, c, d, e, f = coeff
        denom = 4*a*b - c**2
        if abs(denom) < 1e-8:
            return 0.0, 0.0
        x0 = (c*e - 2*b*d) / denom
        y0 = (c*d - 2*a*e) / denom
        if max(abs(x0), abs(y0)) > 1.0:
            return 0.0, 0.0
        return float(y0), float(x0)
    except Exception:
        return 0.0, 0.0

def _movavg(x, k=3):
    if k <= 1: return x
    k = int(k)
    pad = k//2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k)/k
    return np.convolve(xp, ker, mode="valid")

def refine_scale_by_growth_curve(
    im_raw, cy, cx, a, b, angle_deg,
    scales=np.linspace(0.7, 1.6, 11),
    eps=0.05, smooth_k=3,
    bg_in_scale=1.3, bg_out_scale=1.8,
    min_ap_pix=10, min_bg_pix=80, fallback_ring_thick=8,
    geom_margin_min=None,
    tau_ring=1.0, K_consec=2,
    sym_ratio_max=3.0,
    phys_Lmax_pix=None
):
    H, W = im_raw.shape
    S, T, ct, st = _rotated_coords(im_raw.shape, cy, cx, angle_deg)

    def _bg_stats(a_s, b_s):
        bg = pill_annulus_mask(im_raw.shape, cy, cx, a_s, b_s, angle_deg, bg_in_scale, bg_out_scale)
        if bg.sum() < min_bg_pix:
            r0 = max(a_s, b_s) * bg_in_scale
            bg = circular_annulus(im_raw.shape, cy, cx, r0, r0 + float(fallback_ring_thick))
        mu, sg = robust_bg_stats(im_raw[bg])
        return mu, max(sg, 1e-6)

    s_list, F_list, SNR_list, NAP_list = [], [], [], []
    ring_snr_list, ring_asym_list = [np.nan], [np.nan]

    a_prev, b_prev = None, None
    mu_bg_ref, sig_bg_ref = _bg_stats(a, b)

    for s in scales:
        a_s, b_s = a*s, b*s

        if geom_margin_min is not None and max(a_s, b_s) * bg_out_scale > geom_margin_min:
            break
        if phys_Lmax_pix is not None and 2*a_s > float(phys_Lmax_pix):
            break

        ap = pill_mask(im_raw.shape, cy, cx, a_s, b_s, angle_deg)
        if ap.sum() < min_ap_pix:
            continue

        flux = float(np.sum(im_raw[ap] - mu_bg_ref))
        err  = sig_bg_ref * np.sqrt(ap.sum())
        snr  = flux / err

        s_list.append(s); F_list.append(flux); SNR_list.append(snr); NAP_list.append(ap.sum())

        if a_prev is not None:
            ap_prev = pill_mask(im_raw.shape, cy, cx, a_prev, b_prev, angle_deg)
            ring = ap & (~ap_prev)
            n_ring = int(ring.sum())
            if n_ring > 0:
                vals = im_raw[ring] - mu_bg_ref
                ring_flux = float(np.sum(vals))
                ring_snr  = ring_flux / (sig_bg_ref * np.sqrt(n_ring))
                # Left/right asymmetry in S (major-axis) halves
                left_flux  = float(np.sum((im_raw[ring & (S < 0)]) - mu_bg_ref)) if np.any(ring & (S < 0)) else 0.0
                right_flux = float(np.sum((im_raw[ring & (S >= 0)]) - mu_bg_ref)) if np.any(ring & (S >= 0)) else 0.0
                asym_ratio = (abs(right_flux) + 1e-12) / (abs(left_flux) + 1e-12)
                ring_snr_list.append(ring_snr)
                ring_asym_list.append(asym_ratio)
            else:
                ring_snr_list.append(np.nan); ring_asym_list.append(np.nan)
        a_prev, b_prev = a_s, b_s

    if len(s_list) == 0:
        return a, b, (None, None, None, None, None, None)

    s_arr   = np.array(s_list, dtype=float)
    F_arr   = np.array(F_list, dtype=float)
    SNR_arr = np.array(SNR_list, dtype=float)
    NAP_arr = np.array(NAP_list, dtype=float)
    RING_SNR_arr  = np.array(ring_snr_list, dtype=float)
    RING_ASYM_arr = np.array(ring_asym_list, dtype=float)

    F_smooth = _movavg(F_arr, k=smooth_k)
    Fmax = np.nanmax(F_smooth)
    idx_plateau = np.where(F_smooth >= (1.0 - eps) * Fmax)[0]
    i_plateau = int(idx_plateau[0]) if idx_plateau.size > 0 else len(s_arr) - 1

    i_stop = len(s_arr) - 1
    if len(s_arr) >= 2:
        bad = (RING_SNR_arr < float(tau_ring)) | (RING_ASYM_arr > float(sym_ratio_max))
        run = 0
        for i in range(1, len(s_arr)):
            run = run + 1 if bad[i] else 0
            if run >= int(K_consec):
                i_stop = max(i - 1, 0)
                break

    i_star = min(i_plateau, i_stop)
    if not np.isfinite(F_arr[i_star]):
        i_star = int(np.nanargmax(SNR_arr))

    a_ref = a * s_arr[i_star]
    b_ref = b * s_arr[i_star]
    return a_ref, b_ref, (s_arr, F_arr, SNR_arr, NAP_arr, RING_SNR_arr, RING_ASYM_arr)

# --------------------------- Star masking helpers --------------------------
def _pixscale_arcsec_from_wcs(wcs, x_cen, y_cen):
    try:
        cd = wcs.pixel_scale_matrix  # deg/pix
        sx = np.hypot(cd[0,0], cd[1,0])
        sy = np.hypot(cd[0,1], cd[1,1])
        as_per_pix = float(0.5 * (sx + sy) * 3600.0)
        if not np.isfinite(as_per_pix) or as_per_pix <= 0:
            raise ValueError
        return as_per_pix
    except Exception:
        return 1.0

def estimate_window_fwhm(raw_win, max_peaks=30, min_distance=5, snr_k=3.0):
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian as sk_gaussian

    im = raw_win.astype(np.float32)
    med = np.nanmedian(im)
    sm = sk_gaussian(im - med, sigma=1.0, preserve_range=True)
    sig = np.nanstd(sm)
    coords = peak_local_max(sm, min_distance=min_distance, threshold_abs=snr_k*sig, num_peaks=max_peaks)
    if coords is None or len(coords) == 0:
        return 3.0
    fwhm = []
    H, W = im.shape
    for r, c in coords:
        r0, r1 = r-4, r+5; c0, c1 = c-4, c+5
        if r0 < 0 or c0 < 0 or r1 > H or c1 > W: continue
        patch = (im[r0:r1, c0:c1] - med).clip(0)
        S = patch.sum()
        if S <= 0: continue
        yy, xx = np.mgrid[r0:r1, c0:c1]
        yb = (yy*patch).sum()/S; xb = (xx*patch).sum()/S
        dy = yy - yb; dx = xx - xb
        var_x = ((dx*dx)*patch).sum()/S
        var_y = ((dy*dy)*patch).sum()/S
        sig_xy = np.sqrt(max(1e-12, 0.5*(var_x + var_y)))
        fwhm.append(2.355 * sig_xy)
    if len(fwhm) == 0:
        return 3.0
    fwhm = np.asarray(fwhm, float)
    med = np.median(fwhm); mad = np.median(np.abs(fwhm - med))
    keep = np.abs(fwhm - med) <= (3*1.4826*mad) if mad > 0 else np.ones_like(fwhm, bool)
    return float(np.median(fwhm[keep])) if np.any(keep) else 3.0

def gaia_mask_for_window(raw_win, wcs, x0, y0, ra_cen, dec_cen,
                         fwhm_star_pix=3.0,
                         search_pad_pix=20,
                         mag_range=(8.0, 18.5),
                         k_core=1.2, k_grow=1.8,
                         mag_soften=(14.0, 18.0),
                         mag_slope=0.25,
                         r_cap_pix=8.0):
    H, W = raw_win.shape
    as_per_pix = _pixscale_arcsec_from_wcs(wcs, x0 + W/2, y0 + H/2)
    half_diag_pix = 0.5 * np.hypot(H, W)
    radius_arcmin = (half_diag_pix + float(search_pad_pix)) * as_per_pix / 60.0

    center = SkyCoord(ra_cen, dec_cen, unit='deg')
    Vizier.ROW_LIMIT = 30000
    tab = None
    for cat in ["I/355/gaiadr3", "I/350/gaiaedr3"]:
        try:
            v = Vizier(columns=["RA_ICRS","DE_ICRS","Gmag"])
            res = v.query_region(center, radius=radius_arcmin*u.arcmin, catalog=cat)
            if len(res) and ("Gmag" in res[0].colnames):
                tab = res[0]; break
        except Exception:
            continue

    mask_win = np.zeros_like(raw_win, dtype=bool)
    if tab is None or len(tab) == 0:
        return mask_win

    gx, gy = wcs.all_world2pix(tab["RA_ICRS"], tab["DE_ICRS"], 0)
    mG = np.asarray(tab["Gmag"], float)
    gx = np.asarray(gx); gy = np.asarray(gy)

    in_win = (gx >= x0-1) & (gx < x0+W+1) & (gy >= y0-1) & (gy < y0+H+1)
    mag_ok = np.isfinite(mG) & (mG >= mag_range[0]) & (mG <= mag_range[1])
    keep = in_win & mag_ok
    if not np.any(keep):
        return mask_win

    gx, gy, mG = gx[keep], gy[keep], mG[keep]
    xw = gx - x0
    yw = gy - y0

    yy, xx = np.mgrid[0:H, 0:W]
    F = float(fwhm_star_pix)

    m0, m1 = mag_soften
    dm = np.clip(m0 - mG, 0.0, m0 - m1) / max(1e-6, (m0 - m1))
# ∈[0,1]
    dr = (dm * mag_slope) * F

    r_core = k_core * F
    r_max  = np.minimum(k_grow * F + dr, float(r_cap_pix))

    for xi, yi, rc in zip(xw, yw, r_max):
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        rr = np.hypot(xx - xi, yy - yi)
        mask_win |= (rr <= max(r_core, rc))
    return mask_win

from skimage.feature import peak_local_max
from skimage.filters import gaussian as sk_gaussian

def robust_sigma(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return 0.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    sig = 1.4826*mad if mad > 0 else np.std(arr)
    return float(sig if np.isfinite(sig) else 0.0)

def detect_mask_bright_stars_window(raw_win, wcs, RA_DEG, DEC_DEG, x0, y0,
                                    exclude_radius_pix=10.0,
                                    k_sigma=4.0,        # 保留：回退分支用
                                    min_distance=4,
                                    fwhm_default=3.0,
                                    k_core=1.2, k_grow=1.8,
                                    bright_scale=0.30,  # 将用于星等→半径的放大项
                                    r_cap_pix=10.0,
                                    show_preview=False):
    """

    """
    import numpy as np
    from astropy.wcs.utils import proj_plane_pixel_scales

    H, W = raw_win.shape

    # 目标像素（保护圈），与原版一致
    xt, yt = wcs.all_world2pix(RA_DEG, DEC_DEG, 0)
    cx_t, cy_t = xt - x0, yt - y0

    # 像素尺度（arcsec/pix），用于给出合理的 FWHM 下限
    try:
        pixscale_arcsec = float(np.mean(proj_plane_pixel_scales(wcs) * 3600.0))
    except Exception:
        pixscale_arcsec = None

    # ============ 1) 主路径：Gaia DR2 查询 ============ #
    gaia_ok = False
    coords_xy_r = None  # 将填充 [(x_pix_in_win, y_pix_in_win, radius_pix), ...]

    try:
        from astroquery.gaia import Gaia
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        # （a）用子窗四角估计查询圆心与半径（弧分），并加一点裕度
        corners = np.array([[x0,       y0      ],
                            [x0+W-1.0, y0      ],
                            [x0+W-1.0, y0+H-1.0],
                            [x0,       y0+H-1.0]])
        ra_c, dec_c = wcs.all_pix2world(corners[:,0], corners[:,1], 0)
        sky_corners = SkyCoord(ra_c*u.deg, dec_c*u.deg, frame='icrs')
        sky_center  = sky_corners.mean()
# 近似
        sep_max = sky_center.separation(sky_corners).max().to(u.arcmin).value
        search_radius_arcmin = float(sep_max + 2.0)
# +2' 裕度

        # （b）Gaia DR2 查询（限制 G<=16.5，避免星太多；可按需调低/提高）
        gmag_limit = 19
        max_results = 5000
        query = f"""
        SELECT TOP {int(max_results)}
            source_id, ra, dec, phot_g_mean_mag
        FROM gaiadr2.gaia_source
        WHERE phot_g_mean_mag <= {float(gmag_limit)}
          AND CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {sky_center.ra.deg}, {sky_center.dec.deg}, {search_radius_arcmin/60.0})
          )=1
        ORDER BY phot_g_mean_mag ASC
        """
        job = Gaia.launch_job_async(query)
        tab = job.get_results()

        if len(tab) > 0:
            ra_s  = np.array(tab["ra"], dtype=float)
            dec_s = np.array(tab["dec"], dtype=float)
            Gmag  = np.array(tab["phot_g_mean_mag"], dtype=float)

            # （c）世界坐标→像素；再转为子窗坐标
            x_pix, y_pix = wcs.all_world2pix(ra_s, dec_s, 0)
            cx = x_pix - x0
            cy = y_pix - y0
            inside = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)

            cx = cx[inside]; cy = cy[inside]; Gmag = Gmag[inside]
            if len(cx) > 0:
                # （d）星等→遮罩半径（像素）
                # 经验式：更亮（G小）→ 半径更大；再与 r_cap_pix 截顶
                # 以 fwhm_default 为下限；k_core/k_grow/bright_scale 参与尺度控制
                F = float(fwhm_default) if np.isfinite(fwhm_default) and fwhm_default > 0 else 3.0
                G0 = 12.0
                # 基础半径（类似 "core"）
                r_core = k_core * F
                # 亮度项（指数缩放更平滑）
                grow_term = (k_grow * F) + (bright_scale * F * (10.0 ** (-0.4*(Gmag - G0))))
                rad = np.maximum(r_core, grow_term)
                rad = np.minimum(rad, float(r_cap_pix))

                # 保护目标（exclude_radius_pix）
                d_to_target = np.hypot(cy - cy_t, cx - cx_t)
                keep = d_to_target > float(exclude_radius_pix)
                cx = cx[keep]; cy = cy[keep]; rad = rad[keep]

                if len(cx) > 0:
                    coords_xy_r = list(zip(cx, cy, rad))
                    gaia_ok = True

    except Exception as e:
        # 可按需打印：print(f"[mask] Gaia query failed: {e}")
        gaia_ok = False

    # ============ 2) 生成遮罩：Gaia 成功路径 ============ #
    mask_star = np.zeros_like(raw_win, dtype=bool)
    if gaia_ok and coords_xy_r:
        yy, xx = np.mgrid[0:H, 0:W]
        for (x, y, r) in coords_xy_r:
            mask_star |= (np.hypot(yy - y, xx - x) <= r)

        if show_preview:
            import matplotlib.pyplot as plt
            vmin, vmax = np.nanpercentile(raw_win, [5, 99.5])
            plt.figure(figsize=(6,6))
            plt.imshow(raw_win, cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
            ys, xs = np.where(mask_star)
            plt.scatter(xs, ys, s=1, alpha=0.35, label='Gaia star mask')
            plt.plot([cx_t], [cy_t], 'yx', ms=10, mew=2, label='target (RA/Dec)')
            # 把用于遮罩的星标出来
            if len(coords_xy_r) > 0:
                cxv = np.array([p[0] for p in coords_xy_r])
                cyv = np.array([p[1] for p in coords_xy_r])
                plt.scatter(cxv, cyv, s=10, facecolors='none', edgecolors='r', label='Gaia stars')
            plt.legend(); plt.title("Mask from Gaia DR2"); plt.axis('off'); plt.tight_layout(); plt.show()

        return mask_star

    # ============ 3) 回退路径：原先的局部峰值检出（签名要求不变） ============ #
    # 与你给出的实现一致（做了极简整理），以在无网络/查询失败时仍能工作
    try:
        import numpy as np
        from skimage.feature import peak_local_max
        from skimage.filters import gaussian as sk_gaussian
        from astropy.stats import mad_std as robust_sigma

        im = raw_win.astype(np.float32)
        med = np.nanmedian(im)
        sm  = sk_gaussian(im - med, sigma=1.0, preserve_range=True)
        sig = robust_sigma(sm)

        coords = peak_local_max(sm, min_distance=int(min_distance),
                                threshold_abs=float(k_sigma)*sig, exclude_border=False)
        if coords is None or len(coords) == 0:
            if show_preview:
                print("[mask] Gaia")
            return mask_star

        # 局部 FWHM 估计（简化版；边界返回默认值）
        def local_fwhm(yc, xc, box=11):
            half = box//2
            y0_, y1 = yc-half, yc+half+1
            x0_, x1 = xc-half, xc+half+1
            if y0_ < 0 or x0_ < 0 or y1 > H or x1 > W:
                return fwhm_default
            patch = (im[y0_:y1, x0_:x1] - med).clip(0)
            S = patch.sum()
            if S <= 0: return fwhm_default
            yy, xx = np.mgrid[y0_:y1, x0_:x1]
            yb = (yy*patch).sum()/S; xb = (xx*patch).sum()/S
            dy = yy - yb; dx = xx - xb
            var_x = ((dx*dx)*patch).sum()/S
            var_y = ((dy*dy)*patch).sum()/S
            sig_xy = np.sqrt(max(1e-12, 0.5*(var_x + var_y)))
            return float(2.355*sig_xy)

        # 亮度归一（用于增大半径）
        peaks_val = sm[coords[:,0], coords[:,1]]
        if np.all(~np.isfinite(peaks_val)) or np.nanmax(peaks_val) <= 0:
            peaks_norm = np.zeros_like(peaks_val, float)
        else:
            vmin = np.nanpercentile(peaks_val, 10)
            vmax = np.nanpercentile(peaks_val, 90)
            if vmax <= vmin: vmax = vmin + 1e-6
            peaks_norm = np.clip((peaks_val - vmin)/(vmax - vmin), 0.0, 1.0)

        yy, xx = np.mgrid[0:H, 0:W]
        for (r, c), bn in zip(coords, peaks_norm):
            # 保护目标
            if np.hypot(r - cy_t, c - cx_t) <= float(exclude_radius_pix):
                continue
            F = local_fwhm(int(r), int(c))
            if not (np.isfinite(F) and F > 0):
                F = float(fwhm_default)
            r_core = k_core * F
            r_grow = k_grow * F + bright_scale * F * bn
            rad = min(max(r_core, r_grow), float(r_cap_pix))
            mask_star |= (np.hypot(yy - r, xx - c) <= rad)

        if show_preview:
            import matplotlib.pyplot as plt
            vmin, vmax = np.nanpercentile(raw_win, [5, 99.5])
            plt.figure(figsize=(6,6))
            plt.imshow(raw_win, cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
            ys, xs = np.where(mask_star)
            plt.scatter(xs, ys, s=1, c='r', alpha=0.4, label='auto star mask (fallback)')
            plt.plot([cx_t], [cy_t], 'yx', ms=10, mew=2, label='target (RA/Dec)')
            plt.legend(); plt.title("Auto mask (fallback: local peaks)")
            plt.axis('off'); plt.tight_layout(); plt.show()

    except Exception as e:
        if show_preview:
            print(f"[mask] 回退也失败：{e}")

    return mask_star


def refine_angle_by_minor_triplet(
    im_win, cx, cy, a, b, angle_deg,
    *, sigma_prior_pix,
    bg_in_scale=1.3, bg_out_scale=1.8, min_bg_pix=80, fallback_ring_thick=8,
    delta_deg=2.0, step_deg=0.05,   # 角度微扫范围与步长
    s_frac=0.40,                    # 三段中心位置：±s_frac*a 与 0
    s_half_frac=0.35,               # 每段在 s 方向取样半宽
    t_span_frac=0.60,               # fit_t_center_psf 的 t 扫描范围系数
    sigma_tol_frac=0.12,            # fit_t_center_psf 的 sigma 容差
    binw=1.0
):
    """
    在当前 (cx,cy,a,b,angle_deg) 基础上，围绕 angle_deg 做小范围扫描。
    对 s ∈ {-s_frac*a, 0, +s_frac*a} 三段分别构建短轴条带并做 PSF 拟合，得到三段 t0。
    以 cost = sum(t0_i^2) 作为目标，选使 cost 最小的角度。
    返回 (angle_refined_deg, t0_center, diag)；若失败返回原角度。
    """
    import numpy as np

    # 角度候选
    angs = np.arange(angle_deg - float(delta_deg), angle_deg + float(delta_deg) + 1e-9, float(step_deg))
    if angs.size == 0:
        return angle_deg, 0.0, {"ok": False, "reason": "empty angle grid"}

    s_centers = np.array([-s_frac * a, 0.0, +s_frac * a], float)
    s_half = float(s_half_frac * a)
    best = {"cost": np.inf, "ang": angle_deg, "t0s": None}

    for ang in angs:
        t0_list = []
        ok_any = False
        for s0 in s_centers:
            # 构建短轴条带（沿 t 方向积分），中心在 s=s0
            t_c, F_t = build_minor_axis_profile(
                im_win, cy, cx, a, b, ang,
                s_center=s0, s_halfwidth=s_half,
                bg_in_scale=bg_in_scale, bg_out_scale=bg_out_scale,
                min_bg_pix=min_bg_pix, fallback_ring_thick=fallback_ring_thick,
                tspan_pix=max(12.0, 4.0 * max(b, sigma_prior_pix*2.355)), binw=binw
            )
            # 对该条带做短轴 PSF 拟合，得到 t0
            try:
                best_t, _ = fit_t_center_psf(
                    t_c, F_t, sigma_prior=sigma_prior_pix,
                    t_span_frac=t_span_frac, sigma_tol_frac=sigma_tol_frac, binw=binw
                )
                t0_list.append(float(best_t["t0"]))
                ok_any = True
            except Exception:
                t0_list.append(np.nan)

        if not ok_any:
            continue

        t0_arr = np.asarray(t0_list, float)
        finite = np.isfinite(t0_arr)
        if not np.any(finite):
            continue

        # 代价：三段 t0 的平方和（只用有限值）
        cost = float(np.sum((t0_arr[finite])**2))
        if cost < best["cost"]:
            best.update({"cost": cost, "ang": float(ang), "t0s": t0_arr})

    if not np.isfinite(best["cost"]):
        return angle_deg, 0.0, {"ok": False, "reason": "no valid angle candidate"}

    # 返回最优角度及中央条带的 t0（用于后续一次中心微调）
    t0_center = float(best["t0s"][1]) if best["t0s"] is not None and np.isfinite(best["t0s"][1]) else 0.0
    return float(best["ang"]), t0_center, {"ok": True, "cost": best["cost"], "t0s": best["t0s"]}

# --------------------------------- Main ------------------------------------
def smooth_index(x, a=0.5, b=10):
    return 1 + 1/(1 + np.exp(a*(x - b)))

    
# ================== Bi-axial (major/minor) strip-fitting helpers ==================
def erf_approx(x):
    x = np.asarray(x, dtype=float)
    sign = np.sign(x); ax = np.abs(x)
    p = 0.3275911
    a1,a2,a3,a4,a5 = 0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429
    t = 1.0/(1.0 + p*ax)
    poly = (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t
    return sign * (1.0 - poly*np.exp(-ax*ax))

def build_major_axis_profile(
    im_raw, cy, cx, a, b, angle_deg,
    bg_in_scale=1.3, bg_out_scale=1.8, min_bg_pix=80, fallback_ring_thick=8,
    xspan_pix=None, binw=1.0
):
    binw = max(1e-6, float(binw))
    H, W = im_raw.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dy = yy - cy; dx = xx - cx
    th = np.deg2rad(angle_deg); ct, st = np.cos(th), np.sin(th)
    s =  dx*ct + dy*st   # Along major axis
    t = -dx*st + dy*ct   # Along minor axis

    # Background from pill annulus (fallback to circular ring)
    try:
        bg_mask = pill_annulus_mask(im_raw.shape, cy, cx, a, b, angle_deg, bg_in_scale, bg_out_scale)
        if bg_mask.sum() < min_bg_pix:
            r0 = max(a, b)*bg_in_scale
            bg_mask = circular_annulus(im_raw.shape, cy, cx, r0, r0 + float(fallback_ring_thick))
        mu_bg, _ = robust_bg_stats(im_raw[bg_mask])
    except Exception:
        mu_bg = np.nanmedian(im_raw)

    half = (xspan_pix/2.0) if xspan_pix is not None else max(20.0, 1.2*a)
    bins = np.arange(-half, half + binw, binw)
    centers = 0.5*(bins[:-1] + bins[1:])
    F_s = np.full_like(centers, np.nan, dtype=float)

    band_t = (np.abs(t) <= b)
    for i, (s0_, s1_) in enumerate(zip(bins[:-1], bins[1:])):
        sel = band_t & (s >= s0_) & (s < s1_)
        if not np.any(sel): 
            continue
        F_s[i] = float(np.sum(im_raw[sel] - mu_bg))
    return centers, F_s

def build_minor_axis_profile(
    im_raw, cy, cx, a, b, angle_deg,
    s_center=0.0, s_halfwidth=None,
    bg_in_scale=1.3, bg_out_scale=1.8, min_bg_pix=80, fallback_ring_thick=8,
    tspan_pix=None, binw=1.0
):
    binw = max(1e-6, float(binw))
    H, W = im_raw.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dy = yy - cy; dx = xx - cx
    th = np.deg2rad(angle_deg); ct, st = np.cos(th), np.sin(th)
    s =  dx*ct + dy*st
    t = -dx*st + dy*ct

    try:
        bg_mask = pill_annulus_mask(im_raw.shape, cy, cx, a, b, angle_deg, bg_in_scale, bg_out_scale)
        if bg_mask.sum() < min_bg_pix:
            r0 = max(a, b)*bg_in_scale
            bg_mask = circular_annulus(im_raw.shape, cy, cx, r0, r0 + float(fallback_ring_thick))
        mu_bg, _ = robust_bg_stats(im_raw[bg_mask])
    except Exception:
        mu_bg = np.nanmedian(im_raw)

    if s_halfwidth is None:
        s_halfwidth = float(a)
    band_s = (np.abs(s - float(s_center)) <= float(s_halfwidth))

    if tspan_pix is None:
        tspan_pix = max(12.0, 4.0 * float(b))
    half = float(tspan_pix)/2.0
    bins = np.arange(-half, half + binw, binw)
    centers = 0.5*(bins[:-1] + bins[1:])
    F_t = np.full_like(centers, np.nan, dtype=float)

    core_mask = band_s
    for i, (t0_, t1_) in enumerate(zip(bins[:-1], bins[1:])):
        sel = core_mask & (t >= t0_) & (t < t1_)
        if not np.any(sel): 
            continue
        F_t[i] = float(np.sum(im_raw[sel] - mu_bg))
    return centers, F_t

def rect_gauss_model(s, s0, a, sigma):
    s = np.asarray(s, dtype=float).reshape(-1)
    sigma = float(sigma) if sigma > 0 else 1e-6
    rt2 = np.sqrt(2.0) * sigma
    return 0.5 * (erf_approx((s - s0 + a) / rt2) - erf_approx((s - s0 - a) / rt2))

def fit_midpoint_from_profile(s_c, F_s, a, b, smooth=False):
    s = np.asarray(s_c, dtype=float).reshape(-1)
    y = np.asarray(F_s, dtype=float).reshape(-1)
    mask = np.isfinite(s) & np.isfinite(y)
    s, y = s[mask], y[mask]
    if s.size < 10: 
        raise ValueError("Too few valid strip points for fitting (<10).")
    if smooth:
        k = 3
        y = np.convolve(np.pad(y, (k//2, k//2), mode="edge"), np.ones(k)/k, mode="valid")
    s0_grid    = np.linspace(-0.5*a, 0.5*a, 81)
    sig_min    = max(0.6, 0.5*b)
    sig_max    = max(sig_min*1.05, 2.5*b)
    sigma_grid = np.linspace(sig_min, sig_max, 25)
    best = {"sse": np.inf}
    ones = np.ones_like(s)
    for s0 in s0_grid:
        for sigma in sigma_grid:
            K = rect_gauss_model(s, s0, a, sigma)
            M = np.column_stack([K, ones])
            x, *_ = np.linalg.lstsq(M, y, rcond=None)
            y_pred = M @ x
            sse = float(np.sum((y - y_pred)**2))
            if sse < best["sse"]:
                best.update({"s0": float(s0), "sigma": float(sigma),
                             "A": float(x[0]), "C": float(x[1]), "sse": sse})
    K_full = rect_gauss_model(s_c, best["s0"], a, best["sigma"])
    F_fit0 = best["A"] * K_full + best["C"]
    return best["s0"], best["sigma"], best["A"], best["C"], F_fit0

def robust_sigma(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return 0.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    sig = 1.4826*mad if mad > 0 else np.std(arr)
    return float(sig if np.isfinite(sig) else 0.0)

from skimage.feature import peak_local_max
from skimage.filters import gaussian as sk_gaussian

def sigma_clip_star_overlap(s_c, F_s, F_model, s0, a, k_sigma=3.0, max_iter=3, core_keep_frac=1.05):
    s = np.asarray(s_c, float).copy()
    y = np.asarray(F_s, float).copy()
    m = np.asarray(F_model, float).copy()
    finite = np.isfinite(s) & np.isfinite(y) & np.isfinite(m)
    keep = finite.copy()
    core = np.abs(s - float(s0)) < (float(core_keep_frac) * float(a))
    scales = []
    for _ in range(int(max_iter)):
        idx = keep & (~core)
        if idx.sum() < 8: break
        resid = y[idx] - m[idx]
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        sig = 1.4826*mad if mad > 0 else np.std(resid)
        if not np.isfinite(sig) or sig <= 0: break
        scales.append(sig)
        bad = (y - m) > (k_sigma * sig)
# only clip positive residuals
        new_keep = keep & (~bad | core)
        if new_keep.sum() == keep.sum(): break
        keep = new_keep
    return keep, (scales[-1] if len(scales) else np.nan)

def estimate_field_fwhm(raw, cy, cx, a, b, angle_deg,
                        exclude_scale=1.6, max_peaks=30, min_distance=5,
                        snr_k=3.0):
    img = raw.astype(np.float32)
    med = np.nanmedian(img)
    norm = img - med
    sm = sk_gaussian(norm, sigma=1.0, preserve_range=True)
    sig = np.nanstd(sm)
    mask_ex = pill_mask(img.shape, cy, cx, a*exclude_scale, b*exclude_scale, angle_deg)
    thr = float(snr_k) * sig
    coords = peak_local_max(sm, min_distance=min_distance, threshold_abs=thr, num_peaks=max_peaks)
    fwhm_vals = []
    H, W = img.shape
    for r, c in coords:
        if mask_ex[r, c]: continue
        r0, r1 = r-4, r+5; c0, c1 = c-4, c+5
        if r0 < 0 or c0 < 0 or r1 > H or c1 > W: continue
        patch = norm[r0:r1, c0:c1]
        w = np.clip(patch, 0, None)
        S = w.sum()
        if S <= 0: continue
        yy, xx = np.mgrid[r0:r1, c0:c1]
        yb = (yy*w).sum()/S; xb = (xx*w).sum()/S
        dy = yy - yb; dx = xx - xb
        var_x = ((dx*dx)*w).sum()/S
        var_y = ((dy*dy)*w).sum()/S
        sig_xy = np.sqrt(max(1e-12, 0.5*(var_x + var_y)))
        fwhm_vals.append(2.355 * sig_xy)
    if len(fwhm_vals) == 0: 
        return np.nan
    fwhm_vals = np.asarray(fwhm_vals, float)
    med = np.median(fwhm_vals); mad = np.median(np.abs(fwhm_vals - med))
    keep = np.abs(fwhm_vals - med) <= (3*1.4826*mad) if mad > 0 else np.ones_like(fwhm_vals, bool)
    return float(np.median(fwhm_vals[keep]))

def psf_line_band_model(s_centers, s0, a, sigma, b, binw=1.0):
    binw = max(1e-6, float(binw))
    s = np.asarray(s_centers, dtype=float).reshape(-1)
    rt2 = np.sqrt(2.0) * float(sigma if sigma > 0 else 1e-6)
    Ks = 0.5 * (erf_approx((s - s0 + a)/rt2) - erf_approx((s - s0 - a)/rt2))
    Kt = erf_approx(b/rt2)
    return Ks * Kt * binw

def joint_fit_psf_line(
    s_c, F_s, a0, b, sigma_star_pix,
    a_scales=np.linspace(0.70, 1.20, 41),
    sigma_tol_frac=0.10,
    s0_span_frac=0.5, s0_steps=81,
    binw=1.0, smooth=False, reg_a_strength=1e-4
):
    s = np.asarray(s_c, dtype=float).reshape(-1)
    y = np.asarray(F_s, dtype=float).reshape(-1)
    mask = np.isfinite(s) & np.isfinite(y)
    s, y = s[mask], y[mask]
    if s.size < 12:
        raise ValueError("Insufficient valid strip samples (<12).")
    if smooth:
        k = 3
        y = np.convolve(np.pad(y, (k//2, k//2), mode="edge"), np.ones(k)/k, mode="valid")

    s0_grid    = np.linspace(-s0_span_frac*float(a0), +s0_span_frac*float(a0), int(s0_steps))
    sigma_grid = np.linspace((1.0 - sigma_tol_frac)*float(sigma_star_pix),
                             (1.0 + sigma_tol_frac)*float(sigma_star_pix), 9)
    ones = np.ones_like(s)
    best = {"sse": np.inf}

    for a in (float(a0) * np.asarray(a_scales, float)):
        for s0 in s0_grid:
            base_all = np.array([psf_line_band_model(s, s0, a, sig, b, binw=binw) for sig in sigma_grid])
# [nσ, N]
            for j, base in enumerate(base_all):
                M = np.column_stack([base, ones])
# linear in A,C
                x, *_ = np.linalg.lstsq(M, y, rcond=None)
                y_pred = M @ x
                reg = float(reg_a_strength) * (a/a0 - 1.0)**2 * y.size * np.var(y) if y.size > 1 else 0.0
                sse = float(np.sum((y - y_pred)**2)) + reg
                if sse < best["sse"]:
                    best.update({"a": float(a), "s0": float(s0), "sigma": float(sigma_grid[j]),
                                 "A": float(x[0]), "C": float(x[1]), "sse": float(sse)})

    base_full = psf_line_band_model(s_c, best["s0"], best["a"], best["sigma"], b, binw=binw)
    F_fit = best["A"] * base_full + best["C"]
    return best, F_fit

def fit_t_center_psf(t_c, F_t, sigma_prior, t_span_frac=0.6, sigma_tol_frac=0.12, binw=1.0):
    t = np.asarray(t_c, dtype=float).reshape(-1)
    y = np.asarray(F_t, dtype=float).reshape(-1)
    finite = np.isfinite(t) & np.isfinite(y)
    t, y = t[finite], y[finite]
    if t.size < 10:
        raise ValueError("Insufficient valid minor-axis strip samples (<10).")
    amp_span = max(3.0, float(np.nanmax(np.abs(t))))
    t0_grid = np.linspace(-t_span_frac*amp_span, +t_span_frac*amp_span, 61)
    sig0 = max(1e-3, float(sigma_prior))
    sig_grid = np.linspace((1.0 - sigma_tol_frac)*sig0, (1.0 + sigma_tol_frac)*sig0, 11)
    ones = np.ones_like(t)
    best = {"sse": np.inf}
    for t0 in t0_grid:
        for sig in sig_grid:
            if not np.isfinite(sig) or sig <= 0: 
                continue
            base = np.exp(-0.5*((t - t0)/sig)**2) * float(binw)
            M = np.column_stack([base, ones])
            x, *_ = np.linalg.lstsq(M, y, rcond=None)
            y_pred = M @ x
            sse = float(np.sum((y - y_pred)**2))
            if sse < best["sse"]:
                best.update({"t0": float(t0), "sigma": float(sig), "A": float(x[0]), "C": float(x[1]), "sse": sse})
    base_full = np.exp(-0.5*((t_c - best["t0"])/best["sigma"])**2) * float(binw)
    F_fit_t = best["A"] * base_full + best["C"]
    return best, F_fit_t

def update_center_from_st(cx, cy, s0, t0, angle_deg):
    th = np.deg2rad(angle_deg); ct, st = np.cos(th), np.sin(th)
    dx = ct*float(s0) - st*float(t0)
    dy = st*float(s0) + ct*float(t0)
    return cx + dx, cy + dy

# --------- Recursive orchestration (integrated version of previous module) ---------
def one_biaxial_update_step(
    raw_win, cx, cy, a, b, angle_deg,
    binw=1.0,
    bg_in_scale=1.3, bg_out_scale=1.8, min_bg_pix=80, fallback_ring_thick=8,
    a_scales=np.linspace(0.70, 1.20, 41),
    sigma_tol_frac=0.10, s0_span_frac=0.5, s0_steps=81,
    minor_k_sigma=2.5, b_floor=1.0, b_soft_limits=(0.4, 2.5),
    exclude_scale_for_stars=1.6,
    k_sigma_clip=3.0, core_keep_frac=1.05,
    reg_a_strength=1e-4
):
    # 1) major strip & initial fit
    s_c, F_s = build_major_axis_profile(
        raw_win, cy, cx, a, b, angle_deg,
        bg_in_scale=bg_in_scale, bg_out_scale=bg_out_scale,
        min_bg_pix=min_bg_pix, fallback_ring_thick=fallback_ring_thick,
        xspan_pix=max(80.0, 2.0*a*1.1), binw=binw
    )
    s0_init, sig_along0, A0, C0, F_fit0 = fit_midpoint_from_profile(s_c, F_s, a, b, smooth=False)
    keep_mask, _ = sigma_clip_star_overlap(
        s_c, F_s, F_fit0, s0_init, a, k_sigma=k_sigma_clip,
        max_iter=3, core_keep_frac=core_keep_frac
    )
    F_s_clean = F_s.copy(); F_s_clean[~keep_mask] = np.nan

    # 2) field-star FWHM prior
    fwhm_star_pix = estimate_field_fwhm(raw_win, cy, cx, a, b, angle_deg, exclude_scale=exclude_scale_for_stars)
    if not np.isfinite(fwhm_star_pix):
        fwhm_star_pix = 2.355 * float(sig_along0) if np.isfinite(sig_along0) else 3.0
    sigma_star_pix = fwhm_star_pix / 2.355

    # 3) major PSF⊗line joint fit
    joint_psf, F_fit_psf = joint_fit_psf_line(
        s_c, F_s_clean, a, b, sigma_star_pix,
        a_scales=np.asarray(a_scales, float),
        sigma_tol_frac=sigma_tol_frac,
        s0_span_frac=s0_span_frac, s0_steps=s0_steps,
        binw=binw, smooth=False, reg_a_strength=reg_a_strength
    )
    a_psf   = joint_psf["a"]; s0_psf = joint_psf["s0"]
    sigma_p = joint_psf["sigma"]; A_p = joint_psf["A"]; C_p = joint_psf["C"]

    # 4) minor strip @ s_center = s0_psf
    t_c, F_t = build_minor_axis_profile(
        raw_win, cy, cx, a_psf, b, angle_deg,
        s_center=s0_psf, s_halfwidth=a_psf*1.05,
        bg_in_scale=bg_in_scale, bg_out_scale=bg_out_scale,
        min_bg_pix=min_bg_pix, fallback_ring_thick=fallback_ring_thick,
        tspan_pix=max(12.0, 4.0*max(b, sigma_p*2.355)), binw=binw
    )
    best_t, F_fit_t = fit_t_center_psf(
        t_c, F_t, sigma_prior=sigma_p, t_span_frac=0.6, sigma_tol_frac=0.12, binw=binw
    )
    t0_psf  = best_t["t0"]; sigma_t = best_t["sigma"]

    # 5) center update
    cx_new, cy_new = update_center_from_st(cx, cy, s0_psf, t0_psf, angle_deg)

    # 6) a,b update
    a_new = float(a_psf)
    b_new = max(float(b_floor), float(minor_k_sigma) * float(sigma_t)) * 0.7
    b_lo = float(b_soft_limits[0]) * float(b)
    b_hi = float(b_soft_limits[1]) * float(b)
    b_new = float(np.clip(b_new, b_lo, b_hi))

    info = {
        "cx": cx_new, "cy": cy_new, "a": a_new, "b": b_new,
        "s0": s0_psf, "t0": t0_psf,
        "sigma_along": sigma_p, "sigma_across": sigma_t,
        "A_long": A_p, "C_long": C_p,
        "FWHM*": float(fwhm_star_pix),
    }
    return (cx_new, cy_new, a_new, b_new, info)

def refine_midpoint_biaxial_psf_recursive(
    raw_win, cx_init, cy_init, a_init, b_init, angle_deg,
    *, max_depth=6, stop_tol_pix=0.05, monotonic=True, **kwargs
):
    history = []
    def _recur(cx, cy, a, b, depth, prev_shift=np.inf):
        cx1, cy1, a1, b1, info = one_biaxial_update_step(
            raw_win, cx, cy, a, b, angle_deg, **kwargs
        )
        shift = float(np.hypot(cx1 - cx, cy1 - cy))
        info["iter"] = depth + 1
        info["shift"] = shift
        history.append(info)

        if shift < float(stop_tol_pix) or depth + 1 >= int(max_depth):
            return cx1, cy1, a1, b1

        if monotonic and shift > prev_shift:
            cx_mid = 0.5*(cx + cx1); cy_mid = 0.5*(cy + cy1)
            return _recur(cx_mid, cy_mid, a1, b1, depth+1, prev_shift=prev_shift)
        else:
            return _recur(cx1, cy1, a1, b1, depth+1, prev_shift=shift)

    cx_fin, cy_fin, a_fin, b_fin = _recur(cx_init, cy_init, a_init, b_init, depth=0)
    return (cx_fin, cy_fin, a_fin, b_fin), history

# ================= Gaia ZP helpers (ported) =================
def circular_ap_phot(im, yx_list, fwhm_pix, bg_in=2.5, bg_out=4.0, k_ap=1.5, sample=0.5):
    H, W = im.shape
    rad = max(1.0, k_ap * float(fwhm_pix))
    r_in, r_out = bg_in*float(fwhm_pix), bg_out*float(fwhm_pix)
    ys = np.arange(-r_out-1.0, r_out+1.0+sample, sample)
    xs = np.arange(-r_out-1.0, r_out+1.0+sample, sample)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    RR = np.hypot(XX, YY)
    m_ap = RR <= rad
    m_bg = (RR >= r_in) & (RR <= r_out)
    area_pix = sample * sample
    fluxes, ferrs = [], []
    for (y0, x0) in yx_list:
        X = x0 + XX; Y = y0 + YY
        m_in = (X >= 0) & (X <= W - 1) & (Y >= 0) & (Y <= H - 1)
        mA = m_in & m_ap; mB = m_in & m_bg
        if not np.any(mA):
            fluxes.append(np.nan); ferrs.append(np.nan); continue
        I_ap = map_coordinates(im, [Y[mA], X[mA]], order=1, mode='nearest')
        I_bg = map_coordinates(im, [Y[mB], X[mB]], order=1, mode='nearest') if np.any(mB) else np.array([])
        if I_bg.size < 20:
            bg_med = np.median(I_ap); bg_sig = 1.4826 * np.median(np.abs(I_ap - bg_med)) + 1e-9
        else:
            bg_med = float(np.median(I_bg)); bg_sig = 1.4826 * np.median(np.abs(I_bg - bg_med)) + 1e-9
        flux = float(np.sum(I_ap - bg_med) * area_pix)
        N_eff = max(I_ap.size, 1)
        ferr = float(np.sqrt(N_eff) * bg_sig * area_pix)
        fluxes.append(flux); ferrs.append(ferr)
    return np.array(fluxes), np.array(ferrs)

def compute_gaia_zero_point(im, hdr, fwhm_pix=3.0, pixscale_arcsec=None,
                            c0=0.0, c1=0.0, gmin=10.5, gmax=19.5, ruwe_max=2.5,
                            max_sources=20000, margin_px=None, min_stars=5,
                            debug=False, out_prefix="gaia_zp_debug"):
    """
    Frame zero-point via Gaia DR2/DR3-like reference; robust median with MAD-clip.
    Returns dict(zp, zp_err, n_ref).
    """
    import numpy as np
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as uu
    from astroquery.vizier import Vizier

    H, W = im.shape
    w = WCS(hdr, relax=True)

    # pixel scale
    def est_pixscale_arcsec(wcs_obj):
        try:
            cd = wcs_obj.wcs.cd
            if cd is not None and cd.shape == (2, 2):
                sx = np.hypot(cd[0,0], cd[1,0]) * 3600.0
                sy = np.hypot(cd[0,1], cd[1,1]) * 3600.0
                if np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0:
                    return float(0.5*(abs(sx)+abs(sy)))
        except Exception:
            pass
        try:
            cdx = abs(wcs_obj.wcs.cdelt[0]) * 3600.0
            cdy = abs(wcs_obj.wcs.cdelt[1]) * 3600.0
            if np.isfinite(cdx) and np.isfinite(cdy) and cdx > 0 and cdy > 0:
                return float(0.5*(cdx+cdy))
        except Exception:
            pass
        return float(pixscale_arcsec) if (pixscale_arcsec is not None) else 1.3

    s_arcsec = est_pixscale_arcsec(w)
    ra0 = float(hdr["CRVAL1"]); dec0 = float(hdr["CRVAL2"])
    c_center = SkyCoord(ra0 * uu.deg, dec0 * uu.deg)
    r_max = 0.5 * np.hypot(W, H) * s_arcsec / 60.0 + 2.0  # arcmin
    Vizier.ROW_LIMIT = int(max_sources)
    viz = Vizier(columns=["RA_ICRS","DE_ICRS","Gmag","bp_rp","ruwe"], row_limit=int(max_sources))
    res = viz.query_region(c_center, radius=r_max * uu.arcmin, catalog="I/345/gaia2")
    if (res is None) or (len(res) == 0) or (len(res[0]) == 0):
        raise RuntimeError("Gaia ZP: empty query result")
    tab = res[0]

    def col_fallback(name, default=np.nan):
        return np.array(tab[name], float) if (name in tab.colnames) else np.full(len(tab), default, float)

    RA = col_fallback("RA_ICRS"); DEC = col_fallback("DE_ICRS")
    G  = col_fallback("Gmag");    color = col_fallback("bp_rp"); ruwe = col_fallback("ruwe")
    m_sel = np.isfinite(G) & (G > float(gmin)) & (G < float(gmax))
    if np.any(np.isfinite(ruwe)): m_sel &= (np.where(np.isfinite(ruwe), ruwe, 9.9) <= float(ruwe_max))
    if np.sum(m_sel) < int(min_stars): raise RuntimeError("Gaia ZP: too few filtered stars.")
    ra = RA[m_sel]; dec = DEC[m_sel]; col = np.where(np.isfinite(color[m_sel]), color[m_sel], 0.0)

    x, y = w.all_world2pix(ra, dec, 1); x -= 1.0; y -= 1.0
    finite = np.isfinite(x) & np.isfinite(y)
    margin = float(margin_px) if (margin_px is not None) else max(8.0, 4.0*float(fwhm_pix))
    inside = finite & (x > margin) & (x < W-1-margin) & (y > margin) & (y < H-1-margin)
    if np.sum(inside) < int(min_stars): raise RuntimeError("Gaia ZP: too few stars inside frame.")
    x_in, y_in = x[inside], y[inside]; G_in, col_in = G[m_sel][inside], col[inside]

    flux, ferr = circular_ap_phot(im, np.column_stack([y_in, x_in]), fwhm_pix=fwhm_pix,
                                  bg_in=2.5, bg_out=4.0, k_ap=1.5, sample=0.5)
    good = (flux > 0) & np.isfinite(flux)
    if np.sum(good) < int(min_stars): raise RuntimeError("Gaia ZP: too few positive-flux stars.")

    m_pred = G_in[good] + float(c0) + float(c1) * col_in[good]
    zp_i = m_pred + 2.5 * np.log10(flux[good])
    med = np.nanmedian(zp_i); mad = 1.4826*np.nanmedian(np.abs(zp_i - med))
    sel = np.abs(zp_i - med) <= max(2.5*mad, 0.05)
    zp_used = zp_i[sel] if np.sum(sel) >= 3 else zp_i
    zp = float(np.nanmedian(zp_used))
    zp_err = float(np.nanstd(zp_used, ddof=1)/np.sqrt(len(zp_used))) if zp_used.size > 1 else np.nan
    return dict(zp=zp, zp_err=zp_err, n_ref=int(zp_used.size))
def _run_photometry(args):
    FILE_PATH = args.fits
    RA_DEG, DEC_DEG = args.ra, args.dec

    # ---------- Read Data and Crop Window ----------
    hdul = fits.open(FILE_PATH)
    hdu, wcs = pick_image_hdu(hdul)
    image = hdu.data.astype(np.float32)
    if wcs is None:
        raise RuntimeError("This HDU is missing a usable WCS.")

    x_pix, y_pix = wcs.all_world2pix(RA_DEG, DEC_DEG, 0)
    H, W = image.shape
    if not (0 <= x_pix < W and 0 <= y_pix < H):
        raise ValueError("WCS result is outside image bounds.")

    raw_win, x0, y0 = safe_crop(image, x_pix, y_pix, args.half_size)
    proc_win = preprocess_window(raw_win, sigma=args.smooth, norm=args.norm)

    # --- Window-only: Generate Limited Gaia Mask ---
    fwhm_est_pix = estimate_window_fwhm(raw_win)
# returns 3.0 on failure
    mask_bad_win = gaia_mask_for_window(raw_win, wcs, x0, y0, RA_DEG, DEC_DEG,
                                        fwhm_star_pix=fwhm_est_pix,
                                        search_pad_pix=20,
                                        k_core=1.2, k_grow=1.8,
                                        mag_soften=(10.0, 19.0),
                                        mag_slope=0.25,
                                        r_cap_pix=6.0)

    mask_star_win = detect_mask_bright_stars_window(
        raw_win, wcs, RA_DEG, DEC_DEG, x0, y0,
        exclude_radius_pix=15.0,
        k_sigma=4.0, min_distance=4,
        fwhm_default=3.0,
        k_core=1.2, k_grow=1.8,
        bright_scale=0.30, r_cap_pix=6.0,
        show_preview=False
    )

#    mask_bad_win = mask_bad_win | mask_star_win
    mask_bad_win = mask_bad_win 

    # Initial center: window center
    cy, cx = proc_win.shape[0]/2.0, proc_win.shape[1]/2.0

    best_overall = {"snr": -np.inf}
    history = []

    # ---------- Iteration: Shape Search -> Center Refinement ----------
    for it in range(min(MAX_ITERS, len(ANGLE_GRID), len(L_GRID), len(W_GRID), len(REFINE_RADIUS))):
        angles = ANGLE_GRID[it]
        Ls     = L_GRID[it]
        Ws     = W_GRID[it]
        rmax   = REFINE_RADIUS[it]
        step   = REFINE_STEP[it]

        # (A) Shape parameter search
        best_shape = {"snr": -np.inf}
        for ang in angles:
            for L in Ls:
                a_try = L/2.0
                for Ww in Ws:
                    b_try = max(1.0, Ww/2.0)
                    _, _, snr, _, _ = forced_photometry(
                        proc_win, cy, cx, a_try, b_try, ang,
                        BG_IN_SCALE, BG_OUT_SCALE, MIN_AP_PIX, MIN_BG_PIX, FALLBACK_RING_THICK,
                        bad_mask=mask_bad_win, trim_frac=TRIM_FRAC_AP
                    )
                    if np.isfinite(snr) and snr > best_shape["snr"]:
                        best_shape.update({"snr": snr, "angle": float(ang),
                                           "a": float(a_try), "b": float(b_try),
                                           "L": int(L), "W": float(Ww)})

        if best_shape["snr"] <= -np.inf/2:
            raise RuntimeError("Shape search failed: invalid photometry at current center.")

        # (B) Center refinement (grid search) + subpixel quadratic fit
        deltas = np.arange(-rmax, rmax + 1e-9, step)
        best_pos = {"snr": -np.inf}
        for dy in deltas:
            for dx in deltas:
                cy_try, cx_try = cy + dy, cx + dx
                _, _, snr, _, _ = forced_photometry(
                    proc_win, cy_try, cx_try, best_shape["a"], best_shape["b"], best_shape["angle"],
                    BG_IN_SCALE, BG_OUT_SCALE, MIN_AP_PIX, MIN_BG_PIX, FALLBACK_RING_THICK,
                    bad_mask=mask_bad_win, trim_frac=TRIM_FRAC_AP
                )
                if np.isfinite(snr) and snr > best_pos["snr"]:
                    best_pos.update({"snr": snr, "cy": float(cy_try), "cx": float(cx_try)})

        gy, gx = int(round(best_pos["cy"])), int(round(best_pos["cx"]))
        if 1 <= gy < proc_win.shape[0]-1 and 1 <= gx < proc_win.shape[1]-1:
            Z = np.empty((3,3), dtype=float)
            for j, yy_ in enumerate([gy-1, gy, gy+1]):
                for i, xx_ in enumerate([gx-1, gx, gx+1]):
                    _, _, s_, _, _ = forced_photometry(
                        proc_win, yy_, xx_, best_shape["a"], best_shape["b"], best_shape["angle"],
                        BG_IN_SCALE, BG_OUT_SCALE, MIN_AP_PIX, MIN_BG_PIX, FALLBACK_RING_THICK,
                        bad_mask=mask_bad_win, trim_frac=TRIM_FRAC_AP
                    )
                    Z[j,i] = s_ if np.isfinite(s_) else -np.inf
            dyy, dxx = subpixel_peak_quadratic(Z)
            cy_ref, cx_ref = gy + dyy, gx + dxx
        else:
            cy_ref, cx_ref = best_pos["cy"], best_pos["cx"]

        snr_gain = (best_pos["snr"] - best_overall.get("snr", -np.inf)) / max(abs(best_overall.get("snr", 1.0)), 1.0)
        move = np.hypot(cy_ref - cy, cx_ref - cx)
        history.append({
            "iter": it+1,
            "angle": best_shape["angle"], "a": best_shape["a"], "b": best_shape["b"],
            "snr_shape": best_shape["snr"], "snr_pos": best_pos["snr"],
            "cy": cy_ref, "cx": cx_ref, "move": move, "snr_gain": snr_gain,
        })

        cy, cx = cy_ref, cx_ref
        if best_pos["snr"] > best_overall.get("snr", -np.inf):
            best_overall = {
                "snr": best_pos["snr"],
                "angle": best_shape["angle"],
                "a": best_shape["a"], "b": best_shape["b"],
                "cy": cy, "cx": cx
            }

        if move < POS_TOL or (snr_gain is not None and snr_gain < SNR_TOL):
            break

    # ---------- Growth-curve refinement ----------
    margin_min = min(best_overall["cy"],
                     raw_win.shape[0]-1 - best_overall["cy"],
                     best_overall["cx"],
                     raw_win.shape[1]-1 - best_overall["cx"])

    a_ref, b_ref, growth = refine_scale_by_growth_curve(
        raw_win, best_overall["cy"], best_overall["cx"],
        best_overall["a"], best_overall["b"], best_overall["angle"],
        scales=C_SCALES, eps=C_EPS, smooth_k=C_SMOOTH_K,
        bg_in_scale=BG_IN_SCALE, bg_out_scale=BG_OUT_SCALE,
        min_ap_pix=MIN_AP_PIX, min_bg_pix=MIN_BG_PIX, fallback_ring_thick=FALLBACK_RING_THICK,
        geom_margin_min=margin_min,
        tau_ring=TAU_RING, K_consec=K_CONSEC,
        sym_ratio_max=SYM_RATIO_MAX,
        phys_Lmax_pix=PHYS_LMAX_PIX
    )

    best_overall["a"], best_overall["b"] = a_ref, b_ref

    # ---------- Recursive bi-axial midpoint correction ----------
    try:
        (cx_rec, cy_rec, a_rec, b_rec), hist_rec = refine_midpoint_biaxial_psf_recursive(
            raw_win, best_overall["cx"], best_overall["cy"],
            best_overall["a"], best_overall["b"], best_overall["angle"],
            max_depth=4, stop_tol_pix=0.05, monotonic=True
        )
        best_overall["cx"], best_overall["cy"] = cx_rec, cy_rec
        best_overall["a"], best_overall["b"] = a_rec, b_rec
    except Exception as e:
        print("[warn] recursive midpoint correction skipped:", e)
    # --- Angle micro-refinement by 3-segment minor-axis PSF ---
    try:
        sigma_prior = float(max(1e-3, best_overall.get("b", 1.0))) / 1.5  # prior for across-PSF sigma
        angle_new, t0_c, diag = refine_angle_by_minor_triplet(
            raw_win,
            best_overall["cx"], best_overall["cy"],
            best_overall["a"],  best_overall["b"],
            best_overall["angle"],
            sigma_prior_pix=sigma_prior,
            delta_deg=30.0, step_deg=1.0, s_frac=0.40,
        )
        best_overall["angle"] = angle_new
        cx2, cy2 = update_center_from_st(
            best_overall["cx"], best_overall["cy"], 0.0, t0_c, angle_new
        )
        best_overall["cx"], best_overall["cy"] = cx2, cy2
    except Exception as e:
        if args.debug:
            print("[warn] angle micro-refine skipped:", e)
    
    # --- Final photometry ---
    flux, err, snr_final, n_ap, n_bg = forced_photometry(
        raw_win,
        best_overall["cy"], best_overall["cx"],
        best_overall["a"],  best_overall["b"],
        best_overall["angle"],
        BG_IN_SCALE, BG_OUT_SCALE,
        MIN_AP_PIX, MIN_BG_PIX, FALLBACK_RING_THICK,
        bad_mask=mask_bad_win, trim_frac=TRIM_FRAC_AP,
    )
    
    # --- World coordinates ---
    global_x = x0 + best_overall["cx"]
    global_y = y0 + best_overall["cy"]
    ra_fin, dec_fin = wcs.all_pix2world(global_x, global_y, 0)
    
    # --- Zero-point & magnitude ---
    zp_used = None
    zp_err_used = None
    zp_method = None
    
    if args.zero_point is not None:
        zp_used = float(args.zero_point)
        zp_method = "user"
    elif getattr(args, "gaia_zp", False):
        try:
            zpd = compute_gaia_zero_point(
                image, hdu.header,
                fwhm_pix=args.fwhm_pix, pixscale_arcsec=args.pixscale_arcsec,
                c0=args.color_c0, c1=args.color_c1,
                gmin=args.gaia_gmin, gmax=args.gaia_gmax,
                ruwe_max=args.gaia_ruwe_max, max_sources=args.gaia_max_sources,
                margin_px=args.gaia_margin, min_stars=args.gaia_min_stars,
                debug=args.debug,
            )
            zp_used = zpd["zp"]
            zp_err_used = zpd.get("zp_err")
            zp_method = f"gaia (n_ref={zpd.get('n_ref','?')})"
        except Exception as e:
            if args.debug:
                print("[WARN] Gaia ZP failed:", e)
    
    # --- Aperture (pill) throughput correction ---
    mag = None
    mag_err = None
    mag_corr = None
    E_pill = 1.0
    dm = 0.0  # 初始化 dm
    try:
        fwhm_val = args.fwhm_pix
        print(fwhm_val)
        E_pill, dm = epill_throughput(best_overall["a"], best_overall["b"], fwhm_pix=fwhm_val)
    except Exception as e:
        if args.debug:
            print(f"[WARN] epill_throughput failed: {e}, using E_pill=1.0, dm=0.0")
        # E_pill = 1.0
        # dm = 0.0
    
    print(f'[Debug]E_pill = {E_pill}, dm = {dm}')
    flux_corr = float(flux) / float(E_pill) if flux is not None else np.nan
    if (zp_used is not None) and np.isfinite(zp_used) and (flux_corr > 0):
        mag = float(zp_used - 2.5 * np.log10(flux_corr))
        mag_corr = mag-dm
        mag_err = float(1.0857362047581294 * (err / flux)) if (err > 0 and np.isfinite(err)) else np.nan
    
    # --- Debug visualization: window + final pill ---
    if args.debug:
        # 创建PSF拟合剖面图
        debug_fig = create_psf_profile_debug_plot(
            raw_win, 
            best_overall["cy"], best_overall["cx"],
            best_overall["a"], best_overall["b"], 
            best_overall["angle"],
            out_path=None,  # 如果要保存: "debug_psf_profile.png"
            title_prefix=""
        )
        plt.show()
        plt.close(debug_fig)

    # ---------- Growth-curve diagnostics (optional plots are commented) ----------
    if growth[0] is not None:
        s_arr, F_arr, SNR_arr, NAP_arr, RING_SNR_arr, RING_ASYM_arr = growth
        # Choose s* near a_ref
        idx_s = int(np.nanargmin(np.abs((s_arr * (best_overall["a"]/s_arr[-1])) - best_overall["a"]))) if np.isfinite(best_overall["a"]) else np.argmax(s_arr)
        s_star = s_arr[idx_s]
        # (Add plotting here if desired)

    # ---------- Log ----------
    if args.debug:
        print("=== Iteration log ===")
        for rec in history:
            print(
                f"(iter {rec['iter']}) angle={rec['angle']:.1f}°, a={rec['a']:.1f}, b={rec['b']:.1f}, "
                f"SNR(shape)={rec['snr_shape']:.2f}, SNR(pos)={rec['snr_pos']:.2f}, "
                f"center=({rec['cx']:.3f},{rec['cy']:.3f}), move={rec['move']:.3f}, "
                f"SNR gain={rec['snr_gain']:.3f}"
            )

    if args.debug:
        print("\n=== Final (with masks & robust trim) ===")
        print(f"[CENTER(win)] ({best_overall['cx']:.3f}, {best_overall['cy']:.3f}); "
          f"[CENTER(global)] ({global_x:.2f},{global_y:.2f}); "
          f"[SKY] ({ra_fin:.6f},{dec_fin:.6f})")
        print(f"[SHAPE] angle={best_overall['angle']:.2f}°, L={2*best_overall['a']:.2f}, W={2*best_overall['b']:.2f}")
        print(f"[FLUX] {flux:.2f} ± {err:.2f}  (SNR={snr_final:.2f}, ap_pix={n_ap}, bg_pix={n_bg})")

    # --- Result summary (simple) ---
    summary = (
        f"RA={ra_fin:.6f} Dec={dec_fin:.6f} "
        f"SNR={snr_final:.2f} flux={flux:.2f}±{err:.2f} "
        f"ap_px={n_ap} bg_px={n_bg}"
    )
    
    if zp_used is not None:
        summary += f"  ZP={zp_used:.3f}"
        if zp_err_used is not None:
            summary += f"±{zp_err_used:.3f}"
        if zp_method:
            summary += f" ({zp_method})"
    
    if mag is not None:
        if mag_err is not None:
            summary += f"  mag={mag:.3f}±{mag_err:.3f}"
        else:
            summary += f"  mag={mag:.3f}"

    if args.debug or getattr(args, "print_result", False):
        print("[RESULT]", summary)

    # Try to extract observation time from headers
    obs_time = None
    try:
        for ext in hdul:
            for key in ("DATE-MID", "DATE-OBS", "DATE", "TIME-OBS", "MJD-OBS"):
                val = ext.header.get(key)
                if val not in (None, ""):
                    obs_time = str(val)
                    break
            if obs_time:
                break
    except Exception:
        obs_time = None

    return {
        "flux": float(flux),
        "flux_err": float(err),
        "snr": float(snr_final),
        "aperture_pixels": int(n_ap),
        "background_pixels": int(n_bg),
        "magnitude": float(mag_corr) if mag is not None else None,
        "magnitude_err": float(mag_err) if mag_err is not None else None,
        "zero_point": float(zp_used) if zp_used is not None else None,
        "zero_point_err": float(zp_err_used) if zp_err_used is not None else None,
        "zero_point_method": zp_method,
        "ra": float(ra_fin),
        "dec": float(dec_fin),
        "history": history,
        "growth": growth,
        "obs_time":obs_time,
        "a": float(best_overall.get("a", float('nan'))),
        "b": float(best_overall.get("b", float('nan'))),
        "angle": float(best_overall.get("angle", float('nan'))),
        "cx": float(best_overall.get("cx", float('nan'))),
        "cy": float(best_overall.get("cy", float('nan'))),
    }


def run_pill_photometry(
    fits_path,
    ra_deg,
    dec_deg,
    *,
    half_size=HALF_SIZE,
    smooth=SIGMA_SMOOTH,
    norm=NORM_MODE,
    out_path="result.png",
    debug=False,
    show=False,
    zero_point=None,
    gaia_zp=False,
    fwhm_pix=3.0,
    pixscale_arcsec=None,
    color_c0=0.5,
    color_c1=0.0,
    gaia_gmin=10.5,
    gaia_gmax=19.5,
    gaia_ruwe_max=2.5,
    gaia_margin=None,
    gaia_min_stars=5,
    gaia_max_sources=20000,
):
    """Run pill-aperture forced photometry from Python code (e.g., notebooks).

    Parameters mirror the CLI options; see :func:`main` for descriptions.
    Set ``out_path=None`` to skip writing the summary PNG and ``show=True`` to
    display it inline inside Jupyter.
    """

    args = SimpleNamespace(
        fits=fits_path,
        ra=ra_deg,
        dec=dec_deg,
        half_size=half_size,
        smooth=smooth,
        norm=norm,
        out=out_path,
        debug=debug,
        show=show,
        zero_point=zero_point,
        gaia_zp=gaia_zp,
        fwhm_pix=fwhm_pix,
        pixscale_arcsec=pixscale_arcsec,
        color_c0=color_c0,
        color_c1=color_c1,
        gaia_gmin=gaia_gmin,
        gaia_gmax=gaia_gmax,
        gaia_ruwe_max=gaia_ruwe_max,
        gaia_margin=gaia_margin,
        gaia_min_stars=gaia_min_stars,
        gaia_max_sources=gaia_max_sources,
    )
    return _run_photometry(args)



def main():
    parser = argparse.ArgumentParser(description="Pill-aperture forced photometry with growth-curve refinement.")
    # --- Single-image (original) ---
    parser.add_argument("fits", nargs="?", default=DEF_FILE_PATH, help="Path to FITS image")
    parser.add_argument("--ra", type=float, default=DEF_RA_DEG, help="Target RA (deg)")
    parser.add_argument("--dec", type=float, default=DEF_DEC_DEG, help="Target Dec (deg)")
    parser.add_argument("--half-size", type=int, default=HALF_SIZE, help="Crop half-size (pixels)")
    parser.add_argument("--smooth", type=float, default=SIGMA_SMOOTH, help="Gaussian sigma for preprocessing")
    parser.add_argument("--norm", type=str, default=NORM_MODE, choices=["zscore","minmax","none"], help="Normalization mode")
    parser.add_argument("--out", type=str, default="result.png", help="Output PNG for summary plot")
    parser.add_argument("--debug", action="store_true", help="Show debug logs and debug plots")
    parser.add_argument("--show", action="store_true", help="Display the final summary figure")

    # --- Zero-point / photometry options ---
    parser.add_argument("--zero-point", type=float, default=None, help="Photometric ZP (mag); if given, used directly.")
    parser.add_argument("--gaia-zp", action="store_true", help="Compute frame ZP from Gaia (single-image mode only).")
    parser.add_argument("--fwhm-pix", type=float, default=3.0, help="FWHM (pix) for Gaia circular aperture phot.")
    parser.add_argument("--pixscale-arcsec", type=float, default=None, help="Override pixel scale in arcsec (auto from WCS if omitted).")
    parser.add_argument("--color-c0", type=float, default=0.5, help="Color term constant for Gaia G -> inst. system")
    parser.add_argument("--color-c1", type=float, default=0.0, help="Color term slope for Gaia G -> inst. system")

    # --- Four-image mode ---
    parser.add_argument("--fits4", nargs=4, metavar=("FITS1","FITS2","FITS3","FITS4"),
                        help="Run 4-image mode with these four FITS files")
    parser.add_argument("--ra-list", type=str, help="Comma-separated RAs (deg) for 4 images, e.g. 'ra1,ra2,ra3,ra4'")
    parser.add_argument("--dec-list", type=str, help="Comma-separated Decs (deg) for 4 images, e.g. 'dec1,dec2,dec3,dec4'")
    parser.add_argument("--zp-list", type=str, help="Comma-separated Zero-Points (mag) for 4 images, e.g. 'zp1,zp2,zp3,zp4'")
    parser.add_argument("--out4", type=str, default="result_4panel.png", help="Output PNG for 4-image panel")
    parser.add_argument("--no-template", action="store_true", help="Disable median template subtraction for 4-image mode")

    args = parser.parse_args()

    # 4-image branch
    if args.fits4 is not None:
        if not args.ra_list or not args.dec_list:
            raise SystemExit("In 4-image mode, please provide --ra-list and --dec-list (comma-separated, 4 each).")
        try:
            ra_list = [float(x.strip()) for x in args.ra_list.split(",")]
            dec_list = [float(x.strip()) for x in args.dec_list.split(",")]
        except Exception:
            raise SystemExit("Failed to parse --ra-list/--dec-list; please provide numeric comma-separated values.")
        if len(ra_list) != 4 or len(dec_list) != 4:
            raise SystemExit("Exactly 4 RA and 4 Dec values are required for 4-image mode.")

        zp_list = None
        if args.zp_list:
            try:
                zp_list = [float(x.strip()) for x in args.zp_list.split(",")]
            except Exception:
                raise SystemExit("Failed to parse --zp-list; please provide 4 numeric values (mag)")
            if len(zp_list) != 4:
                raise SystemExit("--zp-list must contain exactly 4 values")

        if zp_list is None and args.zero_point is None:
            raise SystemExit("In 4-image mode, provide either --zp-list (recommended) or a scalar --zero-point.")

        _ = run_pill_photometry_4images(
            args.fits4,
            ra_list,
            dec_list,
            half_size=args.half_size,
            out4_path=args.out4,
            use_template=(not args.no_template),
            debug=args.debug,
            zero_point_list=zp_list,
            zero_point=args.zero_point,
            fwhm_pix=args.fwhm_pix,
            pixscale_arcsec=args.pixscale_arcsec,
            color_c0=args.color_c0,
            color_c1=args.color_c1,
        )
        return

    # Default: single-image original path
    return {
            "flux": float(flux),
            "flux_err": float(err),
            "snr": float(snr_final),
            "aperture_pixels": int(n_ap),
            "background_pixels": int(n_bg),
            "magnitude": float(mag) if mag is not None else None,
            "magnitude_err": float(mag_err) if mag_err is not None else None,
            "zero_point": float(zp_used) if zp_used is not None else None,
            "zero_point_err": float(zp_err_used) if zp_err_used is not None else None,
            "zero_point_method": zp_method,
            "ra_fin": float(ra_fin),
            "dec_fin": float(dec_fin),
            "history": history,
            "growth": growth,
            "obs_time":obs_time
        }

# ====================== Four-image template-subtraction mode (aligned) ======================

def _write_temp_fits_with_window(orig_path, data_full, header_like):
    """
    Write a temporary FITS file that contains a single image HDU with the provided data and header.
    Returns the temp file path.
    """
    from astropy.io import fits as _fits
    hdul_new = _fits.HDUList([_fits.PrimaryHDU(data=data_full, header=header_like)])
    tmp = tempfile.NamedTemporaryFile(prefix="tmplsub_", suffix=".fits", delete=False)
    hdul_new.writeto(tmp.name, overwrite=True)
    tmp_path = tmp.name
    tmp.close()
    return tmp_path

def run_pill_photometry_4images(
    fits_paths,
    ra_list,
    dec_list,
    *,
    half_size=30,
    out4_path="result_4panel.png",
    use_template=True,
    debug=False,
    zero_point_list=None,
    zero_point=None,
    fwhm_pix=3.0,
    pixscale_arcsec=None,
    color_c0=0.5,
    color_c1=0.0,
):
    """
    Build a median template from an ALIGNED window, subtract it from each frame,
    then run photometry on each temp FITS.
    
    Parameters:
    -----------
    fits_paths : list of str
        List of 4 FITS file paths
    ra_list, dec_list : list of float
        Lists of 4 RA and Dec coordinates (degrees)
    half_size : int
        Half-size of the window (pixels)
    out4_path : str
        Output path for the 4-panel visualization
    use_template : bool
        Whether to use template subtraction (currently always enabled)
    debug : bool
        Enable debug output
    zero_point_list : list of float or None
        List of 4 zero points (one per frame). If provided, overrides zero_point.
    zero_point : float or None
        Single zero point to use for all frames (if zero_point_list is None)
    fwhm_pix : float or list of float
        FWHM in pixels. Can be a single value (applied to all frames) or 
        a list of 4 values (one per frame).
    pixscale_arcsec : float or None
        Pixel scale in arcseconds/pixel
    color_c0, color_c1 : float
        Color correction coefficients
    
    Returns:
    --------
    results : list of dict
        List of 4 photometry result dictionaries
    """
    import numpy as np
    import matplotlib.pyplot as plt

    assert len(fits_paths) == 4 and len(ra_list) == 4 and len(dec_list) == 4, \
        "Need 4 images and 4 coords"

    # ---- 第一步: 加载所有4个图像和窗口 ----
    hduls = []
    hdus = []
    wcss = []
    full_images = []
    raw_wins = []
    x0s, y0s = [], []

    print("[4-image mode] Loading images...")
    for i in range(4):
        hdul = fits.open(fits_paths[i])
        hdu, wcs = pick_image_hdu(hdul)
        if wcs is None or getattr(hdu, "data", None) is None:
            raise RuntimeError(f"WCS or data missing in {fits_paths[i]}")
        
        img = hdu.data.astype(np.float32)
        H, W = img.shape
        
        # Aligned center window
        xc, yc = W / 2.0, H / 2.0
        win, x0, y0 = safe_crop(img, xc, yc, half_size)
        
        hduls.append(hdul)
        hdus.append(hdu)
        wcss.append(wcs)
        full_images.append(img)
        raw_wins.append(win.copy())
        x0s.append(x0)
        y0s.append(y0)
        
        if debug:
            print(f"  Frame {i+1}: shape={img.shape}, window at ({x0},{y0})")

    H_win, W_win = raw_wins[0].shape
    
    # ---- 第二步: 创建目标masks（用于对齐和模板生成）----
    print("[4-image mode] Creating target masks...")
    target_masks = []
    for i in range(4):
        H_w, W_w = raw_wins[i].shape
        
        # 获取目标在窗口中的位置
        x_t_full, y_t_full = wcss[i].all_world2pix(float(ra_list[i]), 
                                                     float(dec_list[i]), 0)
        x_t_win = x_t_full - x0s[i]
        y_t_win = y_t_full - y0s[i]
        
        # 创建目标mask（比实际目标稍大一些）
        mask_radius = 15  # 像素
        yy, xx = np.mgrid[0:H_w, 0:W_w]
        target_mask = np.hypot(yy - y_t_win, xx - x_t_win) <= mask_radius
        target_masks.append(target_mask)
        
        if debug:
            masked_pix = np.sum(target_mask)
            print(f"  Frame {i+1}: target at ({x_t_win:.1f},{y_t_win:.1f}), "
                  f"masked {masked_pix} pixels")
    
    # ---- 第三步: 对齐窗口（排除目标区域）----
    print("[4-image mode] Aligning windows...")
    aligned_wins, shifts = _align_windows_robust(raw_wins, 
                                                  target_masks=target_masks,
                                                  search_radius=20.0, 
                                                  debug=debug)
    
    # ---- 第四步: 生成模板（排除目标像素）----
    print("[4-image mode] Generating template...")
    # template_ref = generate_template_with_mask(aligned_wins, 
    #                                        target_masks, 
    #                                        method='median')
    if use_template:
        template_ref = generate_template_with_mask(aligned_wins, 
                                                   target_masks, 
                                                   method='median')
        if debug:
            non_nan = np.sum(np.isfinite(template_ref))
            print(f"  Template: {non_nan}/{template_ref.size} valid pixels")
    else:
        template_ref = np.zeros_like(aligned_wins[0], dtype=np.float32)
        print("  Template subtraction disabled")
    
    # ---- 第五步: 每帧减去模板并写入临时FITS ----
    print("[4-image mode] Subtracting template from each frame...")
    temp_paths = []
    for i in range(4):
        data_full = full_images[i].copy()
        y0, x0 = y0s[i], x0s[i]
        dx, dy = shifts[i]
        
        # 把模板从参考系移回该帧
        tmpl_i = _fourier_shift2d_smooth(template_ref, dx, dy, edge_taper=5)
        
        # 减法
        sub_win = raw_wins[i] - tmpl_i
        
        # 替换到完整图像中
        data_full[y0:y0+H_win, x0:x0+W_win] = sub_win
        
        # 写临时FITS
        temp_path = _write_temp_fits_with_window(fits_paths[i], data_full, 
                                                 hdus[i].header)
        temp_paths.append(temp_path)
        
        if debug:
            sub_median = np.nanmedian(sub_win)
            sub_std = np.nanstd(sub_win)
            print(f"  Frame {i+1}: subtracted, median={sub_median:.2f}, "
                  f"std={sub_std:.2f}")

    # ---- 第六步: 对每个临时FITS运行测光 ----
    print("[4-image mode] Running photometry on each frame...")
    results = []
    for i in range(4):
        # 计算安全的half-size
        H, W = full_images[i].shape
        x_t, y_t = wcss[i].all_world2pix(float(ra_list[i]), float(dec_list[i]), 0)
        min_margin = float(min(x_t, y_t, W - x_t, H - y_t))
        safe_half = int(max(5, min(half_size, np.floor(min_margin - 2))))

        # 处理 zero_point
        zp_i = None
        if zero_point_list is not None:
            if len(zero_point_list) != 4:
                raise RuntimeError("zero_point_list must have 4 values")
            zp_i = float(zero_point_list[i])
        elif zero_point is not None:
            zp_i = float(zero_point)

        # 处理 fwhm_pix（支持单个值或列表）
        fwhm_i = fwhm_pix
        if isinstance(fwhm_pix, (list, tuple, np.ndarray)):
            if len(fwhm_pix) != 4:
                raise RuntimeError("fwhm_pix list must have 4 values")
            fwhm_i = float(fwhm_pix[i])
        elif fwhm_pix is not None:
            fwhm_i = float(fwhm_pix)

        res = run_pill_photometry(
            temp_paths[i],
            float(ra_list[i]),
            float(dec_list[i]),
            half_size=safe_half,
            smooth=0.2,  # SIGMA_SMOOTH
            norm="zscore",  # NORM_MODE
            out_path=None,
            debug=False,  # 单独的debug在这里关闭，用主debug控制
            show=False,
            zero_point=zp_i,
            gaia_zp=False,
            fwhm_pix=fwhm_i,
            pixscale_arcsec=pixscale_arcsec,
            color_c0=color_c0,
            color_c1=color_c1,
        )
        results.append(res)
        
        if debug:
            mag = res.get('magnitude') or np.nan
            mag_err = res.get('magnitude_err') or np.nan  
            snr = res.get('snr') or np.nan
            print(f"  Frame {i+1}: mag={mag:.3f}±{mag_err:.3f}, SNR={snr:.2f}")

    # ---- 第七步: 生成4面板可视化 ----
    print("[4-image mode] Creating 4-panel visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)
    
    for i, ax in enumerate(axes):
        # 显示减法后的窗口
        dx, dy = shifts[i]
        tmpl_i = _fourier_shift2d_smooth(template_ref, dx, dy, edge_taper=5)
        disp_win = raw_wins[i] - tmpl_i
        H_disp, W_disp = disp_win.shape
        
        # Asinh stretch
        disp_arr, vmin, vmax = _asinh_stretch(disp_win)
        im = ax.imshow(disp_arr, cmap="gray", vmin=vmin, vmax=vmax, origin='upper')
        ax.set_title(f"Image {i+1}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    
        res = results[i]
    
        # 获取目标位置（使用最终的世界坐标）
        ra_cen = res.get("ra", float(ra_list[i])) if isinstance(res, dict) else float(ra_list[i])
        dec_cen = res.get("dec", float(dec_list[i])) if isinstance(res, dict) else float(dec_list[i])
        x_full, y_full = wcss[i].all_world2pix(ra_cen, dec_cen, 0)
        x_disp = x_full - x0s[i]
        y_disp = y_full - y0s[i]
    
        if res is not None and isinstance(res, dict) and np.isfinite(res.get("a", np.nan)):
            a_i = float(res["a"])
            b_i = float(res["b"])
            ang = float(res["angle"])
    
            # 使用统一的孔径绘制函数
            draw_aperture_overlay(ax, x_disp, y_disp, a_i, b_i, ang, 
                                 disp_win.shape, color='cyan', linewidth=0.8, alpha=0.95)
    
            # Epill - 使用对应的 fwhm_i
            try:
                # 获取当前帧的 fwhm
                fwhm_i = fwhm_pix
                if isinstance(fwhm_pix, (list, tuple, np.ndarray)):
                    fwhm_i = float(fwhm_pix[i])
                elif fwhm_pix is not None:
                    fwhm_i = float(fwhm_pix)
                    
                E, dm = epill_throughput(a_i, b_i, fwhm_pix=fwhm_i)
                epill_text = f"Epill={E:.2f} (Δm={dm:.2f})"
            except Exception:
                epill_text = "Epill=N/A"
    
            # 文本标签
            mag = res.get("magnitude", None)
            magerr = res.get("magnitude_err", None)
            if mag is not None and np.isfinite(mag):
                txt = f"RA {ra_cen:.6f}\nDec {dec_cen:.6f}\nmag {mag:.3f} ± {magerr:.3f}\n{epill_text}"
            else:
                txt = f"RA {ra_cen:.6f}\nDec {dec_cen:.6f}\nmag N/A\n{epill_text}"
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, va="bottom", ha="left",
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"), fontsize=10)
    
        # 色条
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    
    plt.suptitle("4 images (template-subtracted, asinh stretch)", 
                fontsize=14, fontweight="bold")
    plt.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.9)
    
    if out4_path:
        plt.savefig(out4_path, dpi=150, bbox_inches="tight")
        print(f"[4-image mode] Saved output to: {out4_path}")
    
    plt.close(fig)

    # ---- 清理临时文件 ----
    for p in temp_paths:
        try:
            os.unlink(p)
        except Exception:
            pass

    return results


    # # Cleanup temp files
    # for p in temp_paths:
    #     try:
    #         os.unlink(p)
    #     except Exception:
    #         pass

    # return results


if __name__ == "__main__":
    main()