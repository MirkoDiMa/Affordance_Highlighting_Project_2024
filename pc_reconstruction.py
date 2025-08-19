import os, numpy as np, open3d as o3d

def sample_pointcloud_from_obj(obj_path: str,
                               n_points: int = 50000,
                               method: str = "uniform",
                               remove_outliers: bool = True,
                               nb_neighbors: int = 25,
                               std_ratio: float = 2.5,
                               estimate_normals: bool = True,
                               out_ply_path: str = None) -> str:
    """
    Load a triangle mesh (.obj), sample a point cloud, optional clean-up, save as PLY.
    Returns the saved .ply path.
    """
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    elif method == "poisson_disk":
        pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)
    else:
        raise ValueError("Unknown sampling method.")

    if remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    if estimate_normals:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_consistent_tangent_plane(50)

    if out_ply_path is None:
        base = os.path.splitext(os.path.basename(obj_path))[0]
        out_ply_path = f"./pointcloud/{base}.ply"
    os.makedirs(os.path.dirname(out_ply_path), exist_ok=True)
    o3d.io.write_point_cloud(out_ply_path, pcd)
    return out_ply_path


def reconstruct_mesh_from_ply(ply_path: str,
                              recon_method: str = "poisson",
                              poisson_depth: int = 10,
                              alpha: float = 0.03,
                              bpa_ball_radius_rel: float = 0.03,
                              density_quantile: float = 0.01,
                              smooth_iters: int = 0,
                              out_mesh_path: str = None) -> str:
    """
    Reconstruct a triangle mesh from PLY point cloud using Open3D:
    - 'poisson' | 'bpa' | 'alpha'
    Cleans low-density vertices for Poisson, optional smoothing.
    Returns saved mesh path (.obj).
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Empty point cloud: {ply_path}")

    if recon_method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(poisson_depth))
        densities = np.asarray(densities)
        keep = densities > np.quantile(densities, 1.0 - (1.0 - density_quantile))
        mesh = mesh.select_by_index(np.where(keep)[0])
    elif recon_method == "bpa":
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        radii = [bpa_ball_radius_rel * diag]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    elif recon_method == "alpha":
        # For speed, you can pass tetra_mesh/pt_map if computing multiple alphas
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    else:
        raise ValueError("Unknown recon_method.")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    if smooth_iters > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=int(smooth_iters))
        mesh.compute_vertex_normals()

    if out_mesh_path is None:
        base = os.path.splitext(os.path.basename(ply_path))[0]
        out_mesh_path = f"./data/{base}_reconstructed.obj"
    os.makedirs(os.path.dirname(out_mesh_path), exist_ok=True)
    o3d.io.write_triangle_mesh(out_mesh_path, mesh)
    return out_mesh_path
