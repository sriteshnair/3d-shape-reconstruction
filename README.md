# 3D Shape Modeling

This repository contains Python implementations for reconstructing 3D shapes from a set of 2D binary silhouette images. It explores two distinct approaches: a geometric approach (Voxel Carving) and a machine learning approach (Neural Implicit Representations).

## Project Structure

* **`voxcarv3D.py`**: Estimates the 3D visual hull by projecting a grid of voxels onto the silhouette images and carving away empty space.
* **`MLPimplicit3D.py`**: Trains a Multi-Layer Perceptron (MLP) to implicitly learn the 3D shape occupancy directly from coordinate data.
* **`show_mesh.py / show_mesh_trimesh.py`**: Helper scripts to open and view the generated 3D meshes.
* **`images/`**: Contains the 12 binary silhouette projection images of the object.