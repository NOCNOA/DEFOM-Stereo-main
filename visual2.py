import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap_norm8_and_l2curve(volume: torch.Tensor,
                                  h: int,
                                  w: int,
                                  gt: float = None,
                                  save_heatmap_path: str = None,
                                  save_curve_path: str = None,
                                  cmap: str = "viridis",
                                  dpi: int = 200):
    """
    volume: torch.Tensor [32, D, H, W]
      - 前 8 个通道：做全局 min-max 归一化后，在 (h,w) 处取 [8, D] 画热力图
      - 后 24 个通道：分成两组各 12 通道，计算每个 d 的 L2 距离，画折线图

    gt: 正确视差(浮点)，在两张图里都用 axvline 标注。
    """

    if volume.ndim != 4:
        raise ValueError(f"Expect volume shape [32, D, H, W], got {tuple(volume.shape)}")

    vol = volume.detach().float().cpu()
    C, D, H, W = vol.shape
    if C != 32:
        raise ValueError(f"Expect C=32, got C={C}")

    if not (0 <= h < H and 0 <= w < W):
        raise ValueError(f"(h,w)=({h},{w}) out of range. H={H}, W={W}")

    # -----------------------------
    # Part 1) 前 8 通道：全局 min-max 归一化 + 热力图
    # -----------------------------
    vol8 = vol[:8]  # [8, D, H, W]
    vmin8 = float(vol8.min().item())
    vmax8 = float(vol8.max().item())
    denom = vmax8 - vmin8
    if denom < 1e-12:
        # 避免除零：所有值几乎一样时，归一化为全 0
        vol8_norm = torch.zeros_like(vol8)
    else:
        vol8_norm = (vol8 - vmin8) / denom

    sl8 = vol8_norm[:, :, h, w]         # [8, D]
    sl8_np = sl8.numpy()
    # heatmap 范围：归一化后通常是 [0,1]，仍然按切片 min/max 更稳健
    hm_vmin = float(sl8_np.min())
    hm_vmax = float(sl8_np.max())

    gt_str = "None" if gt is None else f"{float(gt):.4f}"

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    im = ax1.imshow(
        sl8_np,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=hm_vmin,
        vmax=hm_vmax,
    )
    ax1.set_xlabel("D (disparity index)")
    ax1.set_ylabel("C (channel index, 0..7)")
    ax1.set_title(
        f"Norm Heatmap (C0-7) at (h={h}, w={w}) | gtD={gt_str} | "
        f"global_min={vmin8:.6g}, global_max={vmax8:.6g}"
    )

    if gt is not None:
        gt_val = float(gt)
        if gt_val < 0 or gt_val > (D - 1):
            print(f"[Warning] gt={gt_val} is out of D range [0, {D-1}].", flush=True)
        ax1.axvline(x=gt_val, color="white", linestyle="--", linewidth=1.8, alpha=0.95)

    cb = fig1.colorbar(im, ax=ax1)
    cb.set_label("normalized value")

    fig1.tight_layout()
    if save_heatmap_path is not None:
        os.makedirs(os.path.dirname(save_heatmap_path) or ".", exist_ok=True)
        fig1.savefig(save_heatmap_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig1)

    # -----------------------------
    # Part 2) 后 24 通道：两组各 12，按 d 计算 L2 距离并画折线
    # -----------------------------
    A = vol[8:20, :, h, w]   # [12, D]
    B = vol[20:32, :, h, w]  # [12, D]
    # dist[d] = ||A[:,d] - B[:,d]||_2
    diff = A - B             # [12, D]
    dist = torch.sqrt(torch.sum(diff * diff, dim=0) + 1e-12)  # [D]
    dist_np = dist.numpy()

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    x = np.arange(D)
    ax2.plot(x, dist_np)
    ax2.set_xlabel("D (disparity index)")
    ax2.set_ylabel("L2 distance (12ch vs 12ch)")
    ax2.set_title(f"L2 Curve (C8-19 vs C20-31) at (h={h}, w={w}) | gtD={gt_str}")

    if gt is not None:
        gt_val = float(gt)
        ax2.axvline(x=gt_val, color="black", linestyle="--", linewidth=1.6, alpha=0.9)
        # 可选：在最近整数 d 上点一下，便于读数（不影响主逻辑）
        gt_int = int(round(gt_val))
        if 0 <= gt_int < D:
            ax2.scatter([gt_int], [dist_np[gt_int]])

    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    if save_curve_path is not None:
        os.makedirs(os.path.dirname(save_curve_path) or ".", exist_ok=True)
        fig2.savefig(save_curve_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig2)

    return {
        "heatmap_slice_vmin": hm_vmin,
        "heatmap_slice_vmax": hm_vmax,
        "norm_global_min": vmin8,
        "norm_global_max": vmax8,
        "l2_curve": dist_np,
    }
