import os, numpy as np, open3d as o3d

def sample_pointcloud_from_obj(obj_path: str,
                               n_points: int = 100_000,
                               method: str = "poisson_disk",
                               remove_outliers: bool = True,
                               nb_neighbors: int = 25,
                               std_ratio: float = 2.5,
                               estimate_normals: bool = True,
                               out_ply_path: str = None) -> str:
    """
    Load a triangle mesh (.obj), sample a *robust* point cloud, clean, estimate normals, save as PLY.
    - Poisson-disk with oversampling -> better coverage
    - Statistical + radius outlier removal -> fewer spurious points
    - Hybrid radius for normals -> scale-invariant normals
    - Voxel downsample -> bring back close to n_points
    """
    import os, numpy as np, open3d as o3d

    mesh = o3d.io.read_triangle_mesh(obj_path)
    if mesh.is_empty():
        raise RuntimeError(f"Empty mesh: {obj_path}")

    # --- scene scale (diagonal length) for scale-invariant parameters ---
    bbox = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    if diag <= 0:
        diag = 1.0  # fallback

    # --- 1) sampling (oversample so we can clean and still have enough points) ---
    oversample = 1.5
    n_init = int(n_points * oversample)
    if method == "poisson_disk":
        pcd = mesh.sample_points_poisson_disk(number_of_points=n_init)
    elif method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=n_init)
    else:
        raise ValueError("Unknown sampling method. Use 'poisson_disk' or 'uniform'.")

    # --- 2) outlier removal (statistical + radius) ---
    if remove_outliers:
        # statistical: good general clean-up
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        # radius: remove isolated points based on a radius relative to model size
        rad = 0.01 * diag            # 1% of diag
        min_nn = 8                   # at least 8 neighbors in that radius
        pcd, _ = pcd.remove_radius_outlier(nb_points=min_nn, radius=rad)

    # --- 3) normals (Hybrid = radius + max_nn, scale-invariant wrt model size) ---
    if estimate_normals:
        n_rad   = 0.04 * diag        # 4% of diag
        max_nn  = 50
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=n_rad, max_nn=max_nn))
        pcd.orient_normals_consistent_tangent_plane(100)

    # --- 4) downsample to ~n_points (voxel size relative to model size) ---
    if len(pcd.points) > n_points:
        # choose voxel so that we land near n_points
        # heuristic: volume scales with voxel^3; start with small fraction of diag
        voxel = 0.0025 * diag
        pcd_ds = pcd.voxel_down_sample(voxel)
        if 0 < len(pcd_ds.points) < len(pcd.points):  # accept only if it helps
            pcd = pcd_ds
        # if still too many, do a random pick
        if len(pcd.points) > n_points:
            idx = np.random.choice(len(pcd.points), size=n_points, replace=False)
            pcd = pcd.select_by_index(idx)

    if out_ply_path is None:
        base = os.path.splitext(os.path.basename(obj_path))[0]
        out_ply_path = f"./pointcloud/{base}.ply"
    os.makedirs(os.path.dirname(out_ply_path), exist_ok=True)
    o3d.io.write_point_cloud(out_ply_path, pcd)
    return out_ply_path



import numpy as np, open3d as o3d, os
def reconstruct_mesh_from_ply(ply_path: str,
                              recon_method: str = "poisson",
                              poisson_depth: int = 11,           # +1 dettaglio
                              alpha: float = 0.03,
                              bpa_ball_radius_rel: float = 0.03,
                              density_quantile: float = 0.002,   # meno aggressivo (0.2%)
                              smooth_iters: int = 5,             # smoothing leggero, come il compagno
                              out_mesh_path: str = None) -> str:

    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Empty point cloud: {ply_path}")

    # --- Ricostruzione ---
    if recon_method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=int(poisson_depth)
        )
        densities = np.asarray(densities)
        # Rimuovi SOLO la coda a bassissima densità (conservativo)
        keep = densities > np.quantile(densities, density_quantile)
        mesh = mesh.select_by_index(np.where(keep)[0])
    elif recon_method == "bpa":
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        radii = [bpa_ball_radius_rel * diag, 2*bpa_ball_radius_rel * diag]  # due raggi aiutano
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    elif recon_method == "alpha":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    else:
        raise ValueError("Unknown recon_method.")

    # --- Crop alla bbox della point cloud (evita gusci spuri Poisson) ---
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb = aabb.scale(1.02, aabb.get_center())  # un filo più grande
    mesh = mesh.crop(aabb)

    # --- Tieni SOLO il componente connesso più grande ---
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    largest = counts.argmax()
    mask = labels != largest
    mesh.remove_triangles_by_mask(mask)
    mesh.remove_unreferenced_vertices()

    # --- Pulizia e smoothing ---
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    if smooth_iters > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=int(smooth_iters))
        mesh.compute_vertex_normals()

    # --- Salvataggio ---
    if out_mesh_path is None:
        base = os.path.splitext(os.path.basename(ply_path))[0]
        out_mesh_path = f"./data/{base}_reconstructed.obj"
    os.makedirs(os.path.dirname(out_mesh_path), exist_ok=True)
    o3d.io.write_triangle_mesh(out_mesh_path, mesh)
    return out_mesh_path
