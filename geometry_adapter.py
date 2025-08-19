import os, numpy as np, open3d as o3d, torch

def compute_bounding_sphere_params(verts_t: torch.Tensor):
    """
    Compute center (mean) and scale (max L2 radius) exactly like your Normalizer.
    Returns (center_np:(3,), scale_float).
    """
    center = torch.mean(verts_t, dim=0)
    scale  = torch.max(torch.norm(verts_t - center, p=2, dim=1))
    return center.detach().cpu().numpy(), float(scale.detach().cpu().item())

def project_field_to_pointcloud(mlp,
                                ply_path: str,
                                center_np,
                                scale_float,
                                out_dir: str,
                                device: torch.device,
                                highlighter_rgb=(204,255,0),
                                gray_rgb=(180,180,180)):
    """
    Evaluate the learned field F(x) on the *original* (non-normalized) point cloud:
    - normalize points with same (center,scale)
    - run MLP → probs → argmax
    - colorize and save PLY + raw scores
    """
    os.makedirs(out_dir, exist_ok=True)
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Empty point cloud: {ply_path}")

    pts = np.asarray(pcd.points, dtype=np.float32)
    pts_norm = (pts - center_np) / (scale_float + 1e-8)

    mlp.eval()
    with torch.no_grad():
        pts_t = torch.from_numpy(pts_norm).to(device)
        probs = mlp(pts_t)                         # (N,2) softmax
        labels = torch.argmax(probs, dim=1).cpu().numpy()
        p_high = probs[:,0].detach().cpu().numpy()

    hi = np.array(highlighter_rgb, dtype=np.uint8)
    gr = np.array(gray_rgb, dtype=np.uint8)
    colors = np.where(labels[:,None] == 0, hi, gr).astype(np.float32) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_ply = os.path.join(out_dir, "pointcloud_highlighted.ply")
    o3d.io.write_point_cloud(out_ply, pcd)
    np.save(os.path.join(out_dir, "pointcloud_prob_highlight.npy"), p_high)
    np.save(os.path.join(out_dir, "pointcloud_label.npy"), labels)
    return out_ply
