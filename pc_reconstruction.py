import os, numpy as np, open3d as o3d

def sample_pointcloud_from_obj(obj_path: str,
                               n_points: int = 150_000,          # ↑ più punti
                               method: str = "poisson_disk",
                               remove_outliers: bool = True,
                               nb_neighbors: int = 30,            # un filo più alto
                               std_ratio: float = 2.5,
                               estimate_normals: bool = True,
                               out_ply_path: str = None) -> str:
    import os, numpy as np, open3d as o3d
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if mesh.is_empty():
        raise RuntimeError(f"Empty mesh: {obj_path}")

    # scala relativa per parametri robusti
    bbox = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    if diag <= 0: diag = 1.0

    # 1) oversampling -> cleaning -> downsample
    oversample = 1.6
    pcd = mesh.sample_points_poisson_disk(number_of_points=int(n_points * oversample))

    if remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        # raggio relativo: elimina isolati residui
        pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.01 * diag)

    if estimate_normals:
        # raggio più ampio = normali più stabili per Poisson
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05 * diag, max_nn=60
        ))
        pcd.orient_normals_consistent_tangent_plane(150)

    # voxel downsample verso ~n_points
    if len(pcd.points) > n_points:
        vox = 0.002 * diag
        ds = pcd.voxel_down_sample(voxel_size=vox)
        if 0 < len(ds.points) < len(pcd.points):
            pcd = ds
    if len(pcd.points) > n_points:
        import numpy as np
        idx = np.random.choice(len(pcd.points), size=n_points, replace=False)
        pcd = pcd.select_by_index(idx)

    if out_ply_path is None:
        base = os.path.splitext(os.path.basename(obj_path))[0]
        out_ply_path = f"./pointcloud/{base}.ply"
    os.makedirs(os.path.dirname(out_ply_path), exist_ok=True)
    o3d.io.write_point_cloud(out_ply_path, pcd)
    return out_ply_path


import numpy as np, open3d as o3d
def _remove_tiny_triangles(mesh, area_thresh_rel=1e-5):
    # soglia in funzione della bbox -> indipendente dalla scala
    aabb = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)
    v0, v1, v2 = verts[tris[:,0]], verts[tris[:,1]], verts[tris[:,2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    keep = areas > (area_thresh_rel * (diag**2))
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    return mesh

def _taubin_smooth(mesh, iters=15):
    # disponibile in Open3D 0.18: smoothing che non rimpicciolisce il modello
    mesh = mesh.filter_smooth_taubin(number_of_iterations=int(iters))
    mesh.compute_vertex_normals()
    return mesh


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
            pcd, depth=int(poisson_depth), scale=1.03, linear_fit=True
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
    
    # elimina triangoli minuscoli residui
    mesh = _remove_tiny_triangles(mesh, area_thresh_rel=5e-6)

    # smoothing Taubin (più efficace del simple smoothing per chiudere micro-fori)
    mesh = _taubin_smooth(mesh, iters=15)

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
