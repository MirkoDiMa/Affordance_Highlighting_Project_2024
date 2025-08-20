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
                              poisson_depth: int = 11,
                              alpha: float = 0.03,
                              bpa_ball_radius_rel: float = 0.03,
                              density_quantile: float = 0.002,
                              smooth_iters: int = 5,
                              out_mesh_path: str = None) -> str:
    import numpy as np, open3d as o3d, os, copy

    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Empty point cloud: {ply_path}")

    # --- Ricostruzione di base ---
    if recon_method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=int(poisson_depth), scale=1.03, linear_fit=True
        )
        densities = np.asarray(densities)
        keep = densities > np.quantile(densities, density_quantile)
        mesh = mesh.select_by_index(np.where(keep)[0])
    elif recon_method == "bpa":
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        radii = [bpa_ball_radius_rel * diag, 2*bpa_ball_radius_rel * diag]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    elif recon_method == "alpha":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    else:
        raise ValueError("Unknown recon_method.")

    # --- Crop alla bbox (leggermente espansa) ---
    aabb = pcd.get_axis_aligned_bounding_box().scale(1.02, pcd.get_center())
    mesh = mesh.crop(aabb)

    # --- Largest connected component ---
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    if counts.size > 0:
        largest = counts.argmax()
        mesh.remove_triangles_by_mask(labels != largest)
        mesh.remove_unreferenced_vertices()

    # --- Pulizia base ---
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    # --- Sanity prima del filtro "tiny" ---
    V0 = np.asarray(mesh.vertices).shape[0]
    F0 = np.asarray(mesh.triangles).shape[0]
    if V0 == 0 or F0 == 0:
        # Fallback: prova BPA se Poisson ha collassato, o viceversa
        if recon_method != "bpa":
            return reconstruct_mesh_from_ply(ply_path, recon_method="bpa",
                                             bpa_ball_radius_rel=bpa_ball_radius_rel,
                                             out_mesh_path=out_mesh_path)
        else:
            raise RuntimeError("Reconstruction produced empty mesh.")

    # --- Rimuovi triangoli minuscoli (con guard rail) ---
    def _remove_tiny_triangles(mesh, area_thresh_rel=5e-6):
        aabb = mesh.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
        tris = np.asarray(mesh.triangles)
        verts = np.asarray(mesh.vertices)
        if tris.size == 0 or verts.size == 0:
            return mesh
        v0, v1, v2 = verts[tris[:,0]], verts[tris[:,1]], verts[tris[:,2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        keep = areas > (area_thresh_rel * (diag**2))
        mesh.remove_triangles_by_mask(~keep)
        mesh.remove_unreferenced_vertices()
        return mesh

    mesh_before = copy.deepcopy(mesh)
    mesh = _remove_tiny_triangles(mesh, area_thresh_rel=5e-6)
    mesh.compute_vertex_normals()

    # Se abbiamo “ucciso” quasi tutto, allenta la soglia
    F1 = np.asarray(mesh.triangles).shape[0]
    if F1 < max(1000, 0.4 * F0):
        mesh = _remove_tiny_triangles(mesh_before, area_thresh_rel=1e-6)
        mesh.compute_vertex_normals()

    # --- Smoothing (Taubin se presente, altrimenti simple) ---
    if hasattr(mesh, "filter_smooth_taubin"):
        mesh = mesh.filter_smooth_taubin(number_of_iterations=int(smooth_iters))
    else:
        mesh = mesh.filter_smooth_simple(number_of_iterations=int(smooth_iters))
    mesh.compute_vertex_normals()

    # --- Ultimo sanity + salvataggio in OFF (più “safe” per Kaolin) ---
    V = np.asarray(mesh.vertices).shape[0]
    F = np.asarray(mesh.triangles).shape[0]
    if V == 0 or F == 0:
        raise RuntimeError("Mesh empty after cleaning; relax thresholds or try BPA.")

    # --- Salvataggio in OBJ (come il compagno) ---
    if out_mesh_path is None:
        base = os.path.splitext(os.path.basename(ply_path))[0]
        out_mesh_path = f"./data/{base}_reconstructed.obj"
    else:
        root, _ = os.path.splitext(out_mesh_path)
        out_mesh_path = root + ".obj"

    # (facoltativo) rimuovi i colori per restare “pulito”
    if mesh.has_vertex_colors():
        mesh.remove_vertex_colors()

    o3d.io.write_triangle_mesh(out_mesh_path, mesh)
    print(f"[reconstruct_mesh_from_ply] Saved mesh: {out_mesh_path} | V={V} F={F}")
    return out_mesh_path

