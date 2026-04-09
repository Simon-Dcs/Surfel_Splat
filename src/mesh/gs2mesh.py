import os
import time

import numpy as np
import open3d as o3d
from plyfile import PlyData
from tqdm import tqdm


class HighQualityMeshReconstructor:
    def __init__(
        self,
        points,
        f_dc,
        voxel_size=0.001,
        outlier_nb_neighbors=50,
        outlier_std_ratio=2.5,
        normal_nb_neighbors=50,
        depth=11,
        smooth_iterations=2,
        min_density=0.1,
        convert_to_cm=True,
    ):
        if convert_to_cm:
            self.points = points
            print("Converting coordinates from meters to centimeters")
        else:
            self.points = points

        self.f_dc = f_dc
        self.voxel_size = voxel_size * (1.0 if convert_to_cm else 1.0)
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        self.normal_nb_neighbors = normal_nb_neighbors
        self.depth = depth
        self.smooth_iterations = smooth_iterations
        self.min_density = min_density
        self.pcd = None
        self.mesh = None
        self.scene_center = None

    def estimate_scene_center(self):
        """Estimate the scene center."""
        print("Estimating scene center...")
        points = np.asarray(self.pcd.points)

        tree = o3d.geometry.KDTreeFlann(self.pcd)
        weights = np.zeros(len(points))

        for i, point in enumerate(points):
            k, _, _ = tree.search_radius_vector_3d(point, radius=self.voxel_size * 2)
            weights[i] = k

        weights = weights / np.sum(weights)
        self.scene_center = np.average(points, weights=weights, axis=0)
        print(f"Scene center estimated at: {self.scene_center}")
        return self.scene_center

    def orient_normals_to_center(self):
        """Orient normals toward the scene center."""
        print("Orienting normals towards scene center...")
        points = np.asarray(self.pcd.points)
        normals = np.asarray(self.pcd.normals)

        if self.scene_center is None:
            self.estimate_scene_center()

        vectors_to_center = self.scene_center - points
        vectors_to_center = vectors_to_center / np.linalg.norm(
            vectors_to_center, axis=1
        )[:, np.newaxis]

        for i in range(len(normals)):
            if np.dot(normals[i], vectors_to_center[i]) < 0:
                normals[i] = -normals[i]

        self.pcd.normals = o3d.utility.Vector3dVector(normals)
        return self.pcd

    def process_colors(self, f_dc):
        """Convert spherical-harmonic DC coefficients into RGB colors."""
        min_vals = np.min(f_dc, axis=0)
        max_vals = np.max(f_dc, axis=0)

        normalized_colors = np.zeros_like(f_dc)
        for i in range(3):
            channel_range = max_vals[i] - min_vals[i]
            if channel_range > 0:
                normalized_colors[:, i] = (f_dc[:, i] - min_vals[i]) / channel_range
            else:
                normalized_colors[:, i] = 0.5

        return np.clip(normalized_colors, 0, 1)

    def create_point_cloud(self):
        """Create a colored point cloud."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.process_colors(self.f_dc))
        return pcd

    def downsample(self):
        """Downsample the point cloud while preserving colors."""
        print("Downsampling point cloud...")
        self.pcd = self.create_point_cloud()

        original_points = len(self.points)
        self.pcd = self.pcd.voxel_down_sample(voxel_size=self.voxel_size)
        downsampled_points = len(np.asarray(self.pcd.points))

        print(f"Downsampled from {original_points} to {downsampled_points} points")
        return self.pcd

    def remove_outliers(self):
        """Remove outliers."""
        print("Removing outliers...")
        with tqdm(total=1) as pbar:
            cl, _ = self.pcd.remove_statistical_outlier(
                nb_neighbors=self.outlier_nb_neighbors,
                std_ratio=self.outlier_std_ratio,
            )
            self.pcd = cl
            pbar.update(1)
        return self.pcd

    def estimate_normals(self):
        """Estimate normals from the local point cloud structure."""
        print("Estimating normals using local point cloud structure...")
        with tqdm(total=2) as pbar:
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(
                    knn=self.normal_nb_neighbors
                )
            )
            pbar.update(1)

            self.pcd.orient_normals_consistent_tangent_plane(
                k=self.normal_nb_neighbors
            )
            pbar.update(1)

        return self.pcd

    def poisson_reconstruction(self):
        """Run Poisson reconstruction with high-quality defaults."""
        print("Performing high-quality Poisson reconstruction...")
        start_time = time.time()

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.pcd,
            depth=self.depth,
            width=0,
            scale=1.1,
            linear_fit=False,
        )

        vertices_to_remove = densities < np.quantile(densities, self.min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        mesh.orient_triangles()

        vertex_colors = self._transfer_colors(mesh, self.pcd)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        elapsed_time = time.time() - start_time
        print(f"High-quality reconstruction completed in {elapsed_time:.2f} seconds")

        self.mesh = mesh
        return mesh

    def _transfer_colors(self, mesh, pcd):
        """Transfer point-cloud colors to mesh vertices with weighted averaging."""
        vertices = np.asarray(mesh.vertices)
        vertex_colors = np.zeros((len(vertices), 3))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        k = 5
        for i, vertex in enumerate(vertices):
            _, idx, dist = pcd_tree.search_knn_vector_3d(vertex, k)
            weights = 1 / (np.array(dist) + 1e-10)
            weights = weights / np.sum(weights)
            colors = np.asarray(pcd.colors)[idx]
            vertex_colors[i] = np.average(colors, weights=weights, axis=0)

        return vertex_colors

    def optimize_mesh(self):
        """Optimize the mesh after reconstruction."""
        print("Optimizing mesh...")
        mesh = self.mesh

        with tqdm(total=4) as pbar:
            mesh.remove_duplicated_triangles()
            pbar.update(1)
            mesh.remove_duplicated_vertices()
            pbar.update(1)
            mesh.remove_degenerate_triangles()
            pbar.update(1)

            mesh.compute_vertex_normals()
            mesh.orient_triangles()

            if self.smooth_iterations > 0:
                for _ in range(self.smooth_iterations):
                    mesh = mesh.filter_smooth_laplacian(
                        number_of_iterations=1,
                        lambda_filter=0.3,
                    )
            pbar.update(1)

        self.mesh = mesh
        return mesh

    def reconstruct(self):
        """Run the full reconstruction pipeline."""
        total_start_time = time.time()

        print("\nStarting mesh reconstruction pipeline...")
        self.downsample()
        self.remove_outliers()
        self.estimate_normals()
        self.poisson_reconstruction()
        self.optimize_mesh()

        total_time = time.time() - total_start_time
        print(f"\nTotal reconstruction time: {total_time:.2f} seconds")
        print(
            f"Final mesh contains {len(self.mesh.vertices)} vertices and "
            f"{len(self.mesh.triangles)} faces"
        )

        return self.mesh

    def save_mesh(self, output_path):
        """Save the mesh as OBJ or PLY."""
        if self.mesh is None:
            raise ValueError("No mesh to save. Please run reconstruct() first.")

        if output_path.lower().endswith(".obj"):
            mesh_format = "obj"
        elif output_path.lower().endswith(".ply"):
            mesh_format = "ply"
        else:
            mesh_format = "ply"
            output_path = output_path + ".ply"

        print(
            f"Saving high-quality mesh to {output_path} as "
            f"{mesh_format.upper()} format..."
        )
        o3d.io.write_triangle_mesh(
            output_path, self.mesh, write_ascii=(mesh_format == "ply")
        )
        print("Mesh saved successfully!")


def reconstruct_high_quality_mesh(
    points, f_dc, output_path=None, convert_to_cm=True, **kwargs
):
    """Convenience wrapper for high-quality mesh reconstruction."""
    reconstructor = HighQualityMeshReconstructor(
        points, f_dc, convert_to_cm=convert_to_cm, **kwargs
    )
    mesh = reconstructor.reconstruct()

    if output_path is not None:
        reconstructor.save_mesh(output_path)

    return mesh


def post_process_mesh(mesh, cluster_to_keep=500):
    """Post-process a mesh to filter out floaters and disconnected parts."""
    import copy

    print("post processing the mesh to have {} clusters".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_0.cluster_connected_triangles()
        )

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    total_clusters = len(cluster_n_triangles)
    if total_clusters <= cluster_to_keep:
        n_cluster = np.sort(cluster_n_triangles.copy())[0] if total_clusters > 0 else 0
    else:
        n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]

    n_cluster = max(n_cluster, 50)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()

    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def convert_to_double_sided(input_path, output_path):
    """Convert a single-sided mesh into a double-sided mesh."""
    input_format = "obj" if input_path.lower().endswith(".obj") else "ply"
    output_format = "obj" if output_path.lower().endswith(".obj") else "ply"

    if not (
        output_path.lower().endswith(".obj")
        or output_path.lower().endswith(".ply")
    ):
        output_path = output_path + "." + output_format

    mesh = o3d.io.read_triangle_mesh(input_path)
    mesh = post_process_mesh(mesh)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    vertex_normals = (
        np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
    )

    triangles_reversed = np.copy(triangles)
    triangles_reversed = triangles_reversed[:, ::-1]
    triangles_double_sided = np.vstack((triangles, triangles_reversed))

    double_sided_mesh = o3d.geometry.TriangleMesh()
    double_sided_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    double_sided_mesh.triangles = o3d.utility.Vector3iVector(triangles_double_sided)

    if vertex_colors is not None and len(vertex_colors) > 0:
        double_sided_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    if vertex_normals is not None and len(vertex_normals) > 0:
        double_sided_mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)

    o3d.io.write_triangle_mesh(
        output_path, double_sided_mesh, write_ascii=(output_format == "ply")
    )
    print(f"Double-sided mesh saved to {output_path}")
    return double_sided_mesh


def load_gaussians_from_ply(ply_path: str) -> tuple[np.ndarray, np.ndarray]:
    plydata = PlyData.read(ply_path)
    vertex_data = plydata["vertex"]

    points = np.column_stack(
        (vertex_data["x"], vertex_data["y"], vertex_data["z"])
    ).astype(np.float32)

    f_dc = np.column_stack(
        (vertex_data["f_dc_0"], vertex_data["f_dc_1"], vertex_data["f_dc_2"])
    ).astype(np.float32)

    return points, f_dc


def env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    input_ply = os.environ.get(
        "GS2MESH_INPUT_PLY",
        "point_clouds/data/000000_scan24_train/gaussians.ply",
    )
    output_mesh = os.environ.get(
        "GS2MESH_OUTPUT_MESH",
        "point_clouds/data/000000_scan24_train/scene24_mesh.ply",
    )
    temp_mesh = os.environ.get("GS2MESH_TEMP_MESH", "./gs2mesh/temp.ply")
    voxel_size = float(os.environ.get("GS2MESH_VOXEL_SIZE", "0.0001"))
    depth = int(os.environ.get("GS2MESH_DEPTH", "10"))
    smooth_iterations = int(os.environ.get("GS2MESH_SMOOTH_ITERATIONS", "2"))
    min_density = float(os.environ.get("GS2MESH_MIN_DENSITY", "0.03"))
    normal_nb_neighbors = int(os.environ.get("GS2MESH_NORMAL_NB_NEIGHBORS", "50"))
    convert_to_cm = env_flag("GS2MESH_CONVERT_TO_CM", False)
    make_double_sided = env_flag("GS2MESH_DOUBLE_SIDED", True)
    visualize = env_flag("GS2MESH_VISUALIZE", True)

    os.makedirs(os.path.dirname(temp_mesh) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_mesh) or ".", exist_ok=True)

    points, f_dc = load_gaussians_from_ply(input_ply)

    mesh = reconstruct_high_quality_mesh(
        points,
        f_dc,
        output_path=temp_mesh,
        voxel_size=voxel_size,
        depth=depth,
        smooth_iterations=smooth_iterations,
        min_density=min_density,
        normal_nb_neighbors=normal_nb_neighbors,
        convert_to_cm=convert_to_cm,
    )

    if make_double_sided:
        convert_to_double_sided(temp_mesh, output_mesh)
    else:
        o3d.io.write_triangle_mesh(
            output_mesh, mesh, write_ascii=output_mesh.lower().endswith(".ply")
        )

    if visualize:
        mesh = o3d.io.read_triangle_mesh(output_mesh)
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1024, height=1024)
        vis.add_geometry(mesh)
        vis.run()
