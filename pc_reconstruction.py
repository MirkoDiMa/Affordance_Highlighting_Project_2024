# pc_reconstruction.py
# Clean, single-implementation version.

import os
import numpy as np
import open3d as o3d


# --------------------------- Point cloud sampling --------------------------- #
def sample_pointcloud_from_obj(
    obj_path: str,
    n_points: int = 150_000,
    method: str = "poisson_disk",
    remove_outliers: bool = True,
    nb_neighbors: int = 30,
    std_ratio: float = 2.5,
    estimate_normals: bool = True,
    out_ply_path: str = None,
) -> str:
    """
    Load an .obj mesh, sample a robust point cloud, clean it, estimate normals,
    and save to PLY. Returns the saved path.

    Notes:
    - Poisson-disk oversampling -> better coverage before cleaning.
    - Statistical + radius outlier removal -> remove isolated points.
    - Hybrid-normal estimation with radius relative to model size -> stable normals.
    - Voxel downsample -> bring back to ~n_points.
    """
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if mesh.is_empty():
        raise RuntimeError(f"Empty mesh: {obj_path}")

    bbox = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    if diag <= 0:
        diag = 1.0

    # 1) Sampling (oversample so we can clean and still have enough points)
    oversample = 1.6
    if method == "poisson_disk":
        pcd = mesh.sample_points_poisson_disk(number_of_points=int(n_points * oversample))
    elif method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=int(n_points * oversample))
    else:
        raise ValueError("Unknown sampling method. Use 'poisson_disk' or 'uniform'.")

    # 2) Outlier removal (more conservative to keep thin parts)
    if remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd, _ = pcd.remove_radius_outlier(nb_points=6, radius=0.008 * diag)

    # 3) Normal estimation (scale-invariant radius + stronger orientation)
    if estimate_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04 * diag, max_nn=60)
        )
        pcd.orient_normals_consistent_tangent_plane(200)

    # 4) Downsample to ~n_points
    if len(pcd.points) > n_points:
        vox = 0.002 * diag
        ds = pcd.voxel_down_sample(voxel_size=vox)
        if 0 < len(ds.points) < len(pcd.points):
            pcd = ds
    if len(pcd.points) > n_points:
        idx = np.random.choice(len(pcd.points), size=n_points, replace=False)
        pcd = pcd.select_by_index(idx)

    # 5) Save
    if out_ply_path is None:
        base = os.path.splitext(os.path.basename(obj_path))[0]
        out_ply_path = f"./pointcloud/{base}.ply"
    os.makedirs(os.path.dirname(out_ply_path), exist_ok=True)
    o3d.io.write_point_cloud(out_ply_path, pcd)
    return out_ply_path


# ------------------------------ Mesh utilities ----------------------------- #
def _largest_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Keep only the largest connected triangle component."""
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if counts.size == 0:
        return mesh
    largest = counts.argmax()
    mesh.remove_triangles_by_mask(labels != largest)
    mesh.remove_unreferenced_vertices()
    return mesh


def _remove_tiny_triangles(mesh: o3d.geometry.TriangleMesh, area_thresh_rel: float = 1e-7
                           ) -> o3d.geometry.TriangleMesh:
    """
    Remove very small triangles by area threshold relative to bbox diagonal.
    Conservative default to avoid punching holes in the surface.
    """
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)
    if tris.size == 0 or verts.size == 0:
        return mesh
    aabb = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
    if diag <= 0:
        return mesh
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    keep = areas > (area_thresh_rel * (diag ** 2))
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    return mesh


def _safe_taubin(mesh: o3d.geometry.TriangleMesh, iterations: int) -> o3d.geometry.TriangleMesh:
    """Run Taubin smoothing if available, otherwise simple smoothing."""
    if hasattr(mesh, "filter_smooth_taubin"):
        mesh = mesh.filter_smooth_taubin(number_of_iterations=int(iterations))
    else:
        mesh = mesh.filter_smooth_simple(number_of_iterations=int(iterations))
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------- Surface reconstruction ----------------------- #
def reconstruct_mesh_from_ply(
    ply_path: str,
    recon_method: str = "poisson",     # "poisson" | "bpa" | "alpha"
    poisson_depth: int = 11,
    alpha: float = 0.03,
    bpa_ball_radius_rel: float = 0.03,
    density_quantile: float = 0.0,     # 0.0 = no trimming by density
    smooth_iters: int = 8,
    out_mesh_path: str = None,
) -> str:
    """
    Reconstruct a mesh from PLY using Open3D and save to OBJ.
    Pipeline:
      PCD -> (Poisson/BPA/Alpha) -> crop to bbox -> largest component ->
      cleanup -> tiny-triangle removal (very conservative) -> smoothing -> save.
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Empty point cloud: {ply_path}")

    # --- Reconstruction ---
    if recon_method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=int(poisson_depth), scale=1.03, linear_fit=True
        )
        densities = np.asarray(densities)
        if density_quantile and density_quantile > 0.0:
            keep = densities >= np.quantile(densities, float(density_quantile))
            mesh = mesh.select_by_index(np.where(keep)[0])
    elif recon_method == "bpa":
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        radii = [bpa_ball_radius_rel * diag, 2 * bpa_ball_radius_rel * diag]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    elif recon_method == "alpha":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    else:
        raise ValueError("Unknown recon_method.")

    # --- Crop to point cloud bbox (slightly expanded to keep border triangles) ---
    aabb = pcd.get_axis_aligned_bounding_box().scale(1.02, pcd.get_center())
    mesh = mesh.crop(aabb)

    # --- Keep largest connected component ---
    mesh = _largest_component(mesh)

    # --- Basic cleanup ---
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    # --- Conservative tiny-triangle removal with fallback ---
    V0 = np.asarray(mesh.vertices).shape[0]
    F0 = np.asarray(mesh.triangles).shape[0]
    if V0 == 0 or F0 == 0:
        if recon_method != "bpa":
            return reconstruct_mesh_from_ply(
                ply_path, recon_method="bpa",
                bpa_ball_radius_rel=bpa_ball_radius_rel,
                out_mesh_path=out_mesh_path
            )
        raise RuntimeError("Reconstruction produced empty mesh.")

    mesh_before = mesh.clone()
    mesh = _remove_tiny_triangles(mesh, area_thresh_rel=1e-7)
    mesh.compute_vertex_normals()

    F1 = np.asarray(mesh.triangles).shape[0]
    # If we removed too much, relax the threshold strongly
    if F1 < max(1000, int(0.8 * F0)):
        mesh = _remove_tiny_triangles(mesh_before, area_thresh_rel=1e-8)
        mesh.compute_vertex_normals()

    # --- Smoothing (slightly conservative) ---
    mesh = _safe_taubin(mesh, iterations=int(max(5, smooth_iters)))

    # --- Final sanity and save to OBJ ---
    V = np.asarray(mesh.vertices).shape[0]
    F = np.asarray(mesh.triangles).shape[0]
    if V == 0 or F == 0:
        raise RuntimeError("Mesh empty after cleaning; relax thresholds or try BPA.")

    if out_mesh_path is None:
        base = os.path.splitext(os.path.basename(ply_path))[0]
        out_mesh_path = f"./data/{base}_reconstructed.obj"
    else:
        root, _ = os.path.splitext(out_mesh_path)
        out_mesh_path = root + ".obj"

    os.makedirs(os.path.dirname(out_mesh_path), exist_ok=True)
    o3d.io.write_triangle_mesh(out_mesh_path, mesh)
    print(f"[reconstruct_mesh_from_ply] Saved mesh: {out_mesh_path} | V={V} F={F}")
    return out_mesh_path
