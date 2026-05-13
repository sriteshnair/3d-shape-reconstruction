import matplotlib.image as mpimg  # Used to read image files (.pgm)
import numpy as np                # Matrix math, arrays, and high-performance vector operations
from skimage import measure       # Contains 'marching_cubes' to turn voxels into a mesh
import trimesh                    # Used to save the final mesh as a standard file (.off)

"""
Voxel Carving 3D Reconstruction

This script performs 3D reconstruction of an object using voxel carving
based on silhouette images taken from multiple camera angles.
The process involves projecting a 3D voxel grid into each silhouette image,
and carving away voxels that do not align with the silhouettes.
The final 3D model is extracted using the Marching Cubes algorithm and saved as an .off file.
"""
# ==========================================
# 1. CAMERA CALIBRATION
# ==========================================
# This matrix contains the projection parameters for 12 different camera angles.
# Each row represents a flattened 3x4 Camera Matrix (P)= K[R|t].
# P combines intrinsic (focal length, center) and extrinsic (rotation, translation) parameters.
calib = np.array([
    [-78.8596, -178.763, -127.597, 300, -230.924, 0, -33.6163, 300,
     -0.525731, 0, -0.85065, 2],
    [0, -221.578, 73.2053, 300, -178.763, -127.597, -78.8596, 300,
     0, -0.85065, -0.525731, 2],
    [ 78.8596, -178.763, -127.597, 300, -73.2053, 0, -221.578, 300,
     0.525731, 0, -0.85065, 2],
    [0, 33.6163, -230.924, 300, -178.763, 127.597, -78.8596, 300,
     0, 0.85065, -0.525731, 2],
    [-78.8596, -178.763, 127.597, 300, 73.2053, 0, 221.578, 300,
     -0.525731, 0, 0.85065, 2],
    [78.8596, -178.763, 127.597, 300, 230.924, 0, 33.6163, 300,
     0.525731, 0, 0.85065, 2],
    [0, -221.578, -73.2053, 300, 178.763, -127.597, 78.8596, 300,
     0, -0.85065, 0.525731, 2],
    [0, 33.6163, 230.924, 300, 178.763, 127.597, 78.8596, 300,
     0, 0.85065, 0.525731, 2],
    [-33.6163, -230.924, 0, 300, -127.597, -78.8596, 178.763, 300,
     -0.85065, -0.525731, 0, 2],
    [-221.578, -73.2053, 0, 300, -127.597, 78.8596, 178.763, 300,
     -0.85065, 0.525731, 0, 2],
    [221.578, -73.2053, 0, 300, 127.597, 78.8596, -178.763, 300,
     0.85065, 0.525731, 0, 2],
    [33.6163, -230.924, 0, 300, 127.597, -78.8596, -178.763, 300,
     0.85065, -0.525731, 0, 2]
])


# Build 3D grids
# Resolution: How detailed the grid is. Higher = smoother model but slower.
# 300 means the grid will be 300 voxels wide.
resolution = 300
# Step: The physical size of one voxel.
# The world is defined from -1 to 1 (size 2). 
# So step = Total Size (2) / 300 = 0.0066...
step = 2 / resolution

# Voxel coordinates
# X, Y, Z = np.mgrid[...]
# Input: Ranges (-1 to 1).
# Operation: Creates three 3D grids. 
#   - X contains the x-coordinate for every voxel.
#   - Y contains the y-coordinate.
#   - Z contains the z-coordinate.
# Output: X, Y, Z are all shape (300, 300, 150).
X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]

# Voxel occupancy
# Input: The shape of our grid.
# Operation: Create a 3D array filled with 1s (True).
# Meaning: We start assuming the ENTIRE box is solid object.
# Output: 'occupancy' is shape (300, 300, 150) type int.
occupancy = np.ndarray((resolution, resolution, resolution // 2), dtype=int)
# Voxels are initially occupied then carved with silhouette information
occupancy.fill(1)
 

# ---------- MAIN ----------
if __name__ == "__main__":
    # ---- PREPARATION ----
    # We flatten the 3D grids into 1D arrays to do matrix multiplication.
    # We verify the stack adds a row of '1's. This is for Homogeneous Coordinates.
    # (x, y, z) -> (x, y, z, 1). This allows the matrix to handle translation.
    # Input: X, Y, Z arrays.
    # Output: XYZ matrix of shape (4, 13,500,000). (4 rows, Total Voxels columns)
    XYZ = np.vstack((
        X.ravel(),
        Y.ravel(),
        Z.ravel(),
        np.ones_like(X.ravel())
    ))
    for i in range(12):
        # 1. READ IMAGE
        # read the input silhouettes
        myFile = "images/image{0}.pgm".format(i)
        print(myFile)
        img = mpimg.imread(myFile)
        # Normalization: Ensure 0 is background, 255 is object.
        if img.dtype == np.float32:  # if not integer
            img = (img * 255).astype(np.uint8)

        # Compute grid projection in images
        # Get image dimensions (Height h, Width w)
        h, w = img.shape

        # ---- Projection ----
        # 2. PROJECTION MATRIX SETUP
        # Reshape the flat row from 'calib' into a 3x4 matrix
        P = calib[i].reshape(3,4)

        # 3. PROJECT 3D POINTS TO 2D
        # Operation: Matrix Multiply (3x4) @ (4xN) -> (3xN)
        # Result 'proj' contains [x', y', w'] for every voxel.
        # These are the horizontal and vertical positions on the image plane before they are scaled for depth.
        # w' (the 3rd row) represents depth/scale.
        proj = P @ XYZ
        
        # 4. PERSPECTIVE DIVISION
        # To get actual pixel coordinates (u, v), we divide by the depth (w').
        # u = column (horizontal), v = row (vertical)
        u = (proj[0] / proj[2]).astype(int)
        v = (proj[1] / proj[2]).astype(int)

        # 5. CHECK BOUNDS
        # Find which voxels project to valid pixels inside the image.
        # Voxels behind the camera or off-screen are ignored (or treated as empty).
        inside = (
            (u >= 0) & (u < w) &
            (v >= 0) & (v < h)
        )
        #print(inside)
        
        # 6. SILHOUETTE CHECK (The Carving)
        # Create a boolean array representing this specific camera view.
        sil = np.zeros_like(u, dtype=bool)
        
        # CRITICAL LINE: Look up the pixel value for every projected point.
        # img index is [row, col] -> [u,v].
        # If img[u,v] > 0, it's part of the object (True).
        # If img[u,v] == 0, it's background (False).
        sil[inside] = img[u[inside], v[inside]] > 0

        # Reshape the flat array back to 3D grid shape (300, 300, 150)
        occ = sil.reshape(occupancy.shape)

        # Update grid occupancy
        # 7. INTERSECTION
        # Logic: New_Occupancy = Old_Occupancy AND Current_View
        # If 'occ' is False (background) for a voxel, 'occupancy' becomes False forever.
        # The voxel is "carved" away
        occupancy &= occ 

    # Voxel visualization
    np.save("myoccupancy.npy", occupancy)
    
    # 8. MARCHING CUBES
    # Input: The 3D boolean grid 'occupancy'.
    # Operation: Finds the surface boundary between 0s and 1s and creates triangles.
    # Output: 'verts' (list of points), 'faces' (list of triangles connecting points).
    # Vertices are returned in "grid coordinates" (e.g., x=150, y=150), not world coords.
    verts, faces, normals, values = measure.marching_cubes(occupancy, 0.25)

    # # 9. TRANSFORM BACK TO WORLD
    # # We must scale the mesh back to the -1 to 1 range.
    # # Formula: Vertex_World = (Vertex_Grid * Step_Size) + Origin
    # verts = verts * step + np.array([-1, -1, -0.5])

    # 10. EXPORT
    # Create a Trimesh object and save it as an .off file.
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export('alvoxels.off')
    
    # Debug info
    data = np.load("occupancy.npy") # Loading a pre-existing file (comparison)
    print(data.shape)
    data2 = np.load("myoccupancy.npy") # Loading the one we just made
    print(data2.shape)
