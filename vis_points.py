import numpy as np
import open3d as o3d

# 加载 npz 文件，设置 allow_pickle=True
data = np.load("predictions.npz", allow_pickle=True)
print("Keys in npz file:", data.files)

# 假设点云数据存储在 "points" 键中，如果不存在，则使用 "arr_0"
if "points" in data.files:
    points = data["points"]
elif "arr_0" in data.files:
    points = data["arr_0"]
else:
    raise ValueError("找不到存储点云数据的键，请检查 npz 文件内容。")

# 如果点云数据不是 (N, 3) 的数组，请根据实际情况进行调整
if points.ndim != 2 or points.shape[1] != 3:
    raise ValueError("点云数据的形状应为 (N, 3)")

# 创建 Open3D 点云对象，并加载点云数据
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 保存为 PLY 文件
output_path = "output.ply"
o3d.io.write_point_cloud(output_path, pcd)
print(f"点云已保存为 {output_path}")