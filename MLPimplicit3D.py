import math as m
import matplotlib.image as mpimg
import numpy as np
import skimage
from skimage import measure
import torch
from torch import nn
import trimesh

# Camera Calibration for Al's image[1..12].pgm
calib = np.array([
    [-78.8596, -178.763, -127.597, 300, -230.924, 0, -33.6163, 300,
     -0.525731, 0, -0.85065, 2],
    [0, -221.578, 73.2053, 300, -178.763, -127.597, -78.8596, 300,
     0, -0.85065, -0.525731, 2],
    [78.8596, -178.763, -127.597, 300, -73.2053, 0, -221.578, 300,
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

# ============================
# 1. CONFIGURATION
# ============================
MAX_EPOCH = 10     # Number of times the AI sees the full dataset
BATCH_SIZE = 100   # How many points the AI learns from at once (Note: 100 is very small for this data size)

# ============================
# 2. GRID GENERATION
# ============================
# We define the 3D space we want to learn.
resolution = 300
step = 2 / resolution

# Voxel coordinates
# X, Y, Z are 3D grids containing the coordinates for every voxel.
# INPUT: Grid parameters (-1 to 1)
# OUTPUT: Three arrays of shape (300, 300, 150)
X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]

# Voxel occupancy
occupancy = np.ndarray((resolution, resolution, resolution // 2), dtype=int)

# Voxels are initially occupied then carved with silhouette information
occupancy.fill(1)

# ============================
# 3. THE NEURAL NETWORK (MLP)
# ============================
# MLP class
class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self):
        super().__init__()
        # The network consists of Linear layers (matrix multiplication) and Activation functions.
        self.layers = nn.Sequential(
            nn.Linear(3, 60),    # Layer 1: Takes (x,y,z), expands to 60 features
            nn.Tanh(),           # Activation: Tanh squashes values between -1 and 1
            nn.Linear(60, 120),  # Layer 2: Expands to 120 features
            nn.ReLU(),           # Activation: ReLU (keeps positives, zeros negatives)
            nn.Linear(120, 60),  # Layer 3: Compresses back to 60
            nn.ReLU(),
            nn.Linear(60, 30),   # Layer 4: Compresses to 30
            nn.ReLU(),
            nn.Linear(30, 1),    # Output Layer: Compresses to 1 value (The "Logit")
            # nn.Sigmoid()
            # Note: No Sigmoid here. The output is a raw score (-inf to +inf).
        )
    def forward(self, x):
        """ Forward pass """
        # INPUT: x is a batch of points, shape (Batch_Size, 3)
        # OUTPUT: Prediction scores, shape (Batch_Size, 1)
        return self.layers(x)


# GPU or not GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

    
# MLP Training
def nif_train(data_in, data_out, batch_size):
    # Initialize the MLP
    mlp = MLP()
    mlp = mlp.float()
    mlp.to(device)

    # --- BALANCING THE DATA ---
    # In 3D space, most voxels are empty (0). If 95% is empty, the AI 
    # could just guess "0" everywhere and get 95% accuracy without learning the shape.
    # We calculate a weight to tell the AI: "Being wrong about an object (1) 
    # is much worse than being wrong about empty air (0)."
    
    n_one = (data_out == 1).sum() # Count total "Object" voxels

    # loss for positives will be multiplied by this factor in the loss function
    # Formula: Weight = Count_of_Empty / Count_of_Objects
    p_weight = (data_out.size()[0] - n_one) / n_one
    print("Pos. Weight: ", p_weight)

    # Define the loss function and optimizer
    # loss_function = nn.CrossEntropyLoss()

    # --- LOSS FUNCTION ---
    # BCEWithLogitsLoss combines Sigmoid + Binary Cross Entropy.
    # It takes raw scores (logits) and compares them to the True Labels (0 or 1).
    # pos_weight applies the balancing factor calculated above.
    loss_function = nn.BCEWithLogitsLoss(pos_weight=p_weight)
    
    # Optimizer: The math engine that updates the weights (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-2)

    # Run the training loop
    for epoch in range(0, MAX_EPOCH):

        print(f'Starting epoch {epoch + 1}/{MAX_EPOCH}')

        # Creating batch indices
        # 1. Shuffle the data indices so batches are random
        permutation = torch.randperm(data_in.size()[0])

        # Set current loss value
        current_loss = 0.0
        accuracy = 0

        # 2. Batch Loop: Process the data in small chunks
        for i in range(0, data_in.size()[0], batch_size):

            # Extract a batch of indices
            indices = permutation[i:i + batch_size]

            # Get the actual data points and labels for this batch
            # batch_x shape: (100, 3) -> 100 coordinates
            # batch_y shape: (100, 1) -> 100 labels (0 or 1)
            batch_x, batch_y = data_in[indices], data_out[indices]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Zero the gradient
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(batch_x.float())

            # Compute loss
            loss = loss_function(outputs, batch_y.float())

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print current loss so far
            current_loss += loss.item()
            if (i/batch_size) % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      ((i/batch_size) + 1, current_loss / (i/batch_size) + 1))

        # End of Epoch: Check accuracy on the whole dataset
        # We run the whole dataset through Sigmoid to get probabilities (0.0 to 1.0)
        outputs = torch.sigmoid(mlp(data_in.float()))
        acc = binary_acc(outputs, data_out)
        print("Binary accuracy: ", acc)

        # Training is complete.
    print('MLP trained.')
    return mlp

# ============================
# 5. ACCURACY METRIC
# ============================
# IOU evaluation between binary grids
def binary_acc(y_pred, y_test):
    # This function is for HUMAN monitoring only. It does not train the AI.
    # y_pred: Probabilities (0.0 to 1.0)
    # y_test: Ground Truth (0 or 1)
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    accuracy = correct_results_sum / y_test.shape[0]
    accuracy = torch.round(accuracy * 100)
    return accuracy


def main():
    # Generate X,Y,Z and occupancy
    occupancy = np.load("myoccupancy.npy")

    # Format data for PyTorch
    # The grid is currently 3 separate 3D arrays (X, Y, Z).
    # We stack them to get shape (300, 300, 150, 3).
    data_in = np.stack((X, Y, Z), axis=-1)
    
    # Total number of voxels = 300 * 300 * 150 = 13,500,000
    resolution_cube = resolution * resolution * resolution
    
    # FLATTENING: The MLP cannot take a 3D grid. It needs a flat list of points.
    # data_in becomes shape (13500000, 3) -> A list of coordinates
    data_in = np.reshape(data_in, (resolution_cube // 2, 3))
    
    # data_out becomes shape (13500000, 1) -> A list of labels
    data_out = np.reshape(occupancy, (resolution_cube // 2, 1))

    # Convert Numpy arrays to PyTorch Tensors
    data_in = torch.from_numpy(data_in).to(device)
    data_out = torch.from_numpy(data_out).to(device)

    # Train mlp
    mlp = nif_train(data_in, data_out, BATCH_SIZE)  # data_out.size()[0])

    # --- VISUALIZATION ---
    # Now that the MLP is trained, we ask it to recreate the shape.
    # 1. Ask MLP to predict the occupancy for EVERY point in the grid.
    outputs = mlp(data_in.float())
    
    # 2. Detach from GPU and convert to Numpy
    occ = outputs.detach().cpu().numpy()

    # 3. Reshape the flat list back into a 3D block (300, 300, 150)
    newocc = np.reshape(occ, (resolution, resolution, resolution // 2))
    
    # 4. Round to 0 or 1 (Note: Ideally should apply sigmoid first, but works roughly without)
    newocc = np.around(newocc)

    # 5. Use Marching Cubes to generate a mesh from the 3D block
    verts, faces, normals, values = measure.marching_cubes(newocc, 0.25)
    
    # 6. Save to file
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export('alimplicit.off')

"""
Assignment 1: Random Sampling
Instead of training on the full grid, we randomly sample points within the 3D space.
This is more efficient and can yield better results with fewer points.
"""
def main_assignment_1():
    # 1. Load the Ground Truth Grid (The "Answer Key")
    occupancy = np.load("myoccupancy.npy")
    
    # 2. Define Training Size
    # Instead of 13.5 million points, we can pick a smaller number (e.g., 100,000)
    # because random sampling is very efficient.
    N_SAMPLES = 100000 
    
    # 3. Generate Random Coordinates (World Space)
    # We create random float numbers between the bounds of your box.
    # X and Y are between -1 and 1. Z is between -0.5 and 0.5.
    rand_x = np.random.uniform(-1, 1, N_SAMPLES)
    rand_y = np.random.uniform(-1, 1, N_SAMPLES)
    rand_z = np.random.uniform(-0.5, 0.5, N_SAMPLES)
    
    # Stack them into the input format (N, 3)
    data_in_numpy = np.stack((rand_x, rand_y, rand_z), axis=-1)
    
    # 4. Get Labels (The Lookup Step)
    # We have a random float like x = -0.45. We need to know: Is this point inside?
    # We calculate which voxel index corresponds to this float.
    # Formula: index = (Coordinate - Start) / Step_Size
    
    idx_x = ((rand_x - (-1)) / step).astype(int)
    idx_y = ((rand_y - (-1)) / step).astype(int)
    idx_z = ((rand_z - (-0.5)) / step).astype(int)
    
    # Safety Check: Ensure we don't try to access index 300 (out of bounds)
    idx_x = np.clip(idx_x, 0, resolution - 1)
    idx_y = np.clip(idx_y, 0, resolution - 1)
    idx_z = np.clip(idx_z, 0, (resolution // 2) - 1)
    
    # Look up the answer in the occupancy grid
    data_out_numpy = occupancy[idx_x, idx_y, idx_z].reshape(-1, 1)

    # 5. Convert to PyTorch
    data_in = torch.from_numpy(data_in_numpy).float().to(device)
    data_out = torch.from_numpy(data_out_numpy).float().to(device)
    
    print(f"Training on {N_SAMPLES} random points...")
    
    # 6. Train
    # Note: We use a larger batch size because the dataset is smaller/cleaner
    mlp = nif_train(data_in, data_out, BATCH_SIZE=5000)
    
    # --- VISUALIZATION ---
    # Important: To DRAW the mesh, we still need to query the regular grid 
    # because Marching Cubes requires a grid structure.
    # We reuse the X, Y, Z grids from the top of your script.
    
    print("Generating mesh from regular grid...")
    grid_in = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    grid_in = torch.from_numpy(grid_in).float().to(device)
    
    # Ask the MLP (which learned from random points) to predict the grid points
    outputs = mlp(grid_in)
    
    # ... (Rest of visualization code is identical) ...

"""
Assignment 2: Balanced Sampling
Instead of training on the full grid, we create a balanced dataset with equal object and background points.
"""
def main_assignment_2():
    # 1. Load the massive grid
    occupancy = np.load("myoccupancy.npy")
    
    # Flatten it into lists
    # data_in: (13500000, 3), data_out: (13500000, 1)
    data_in = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    data_out = occupancy.reshape(-1, 1)

    print(f"Original Dataset Size: {len(data_out)}")

    # 2. SEPARATE indices for "Object" (1) and "Background" (0)
    # np.where returns the indices where the condition is true
    idx_object = np.where(data_out == 1)[0]
    idx_background = np.where(data_out == 0)[0]

    n_object = len(idx_object)
    print(f"Number of Object Voxels: {n_object}")
    print(f"Number of Background Voxels: {len(idx_background)}")

    # 3. UNDERSAMPLE the Background
    # We want the same number of background points as object points (50/50 split)
    # replace=False means we don't pick the same point twice
    idx_background_subset = np.random.choice(idx_background, size=n_object, replace=False)

    # 4. COMBINE them to create the new Training Set
    # Concatenate the lists of indices
    final_indices = np.concatenate([idx_object, idx_background_subset])
    
    # Create the smaller, balanced arrays
    data_in_balanced = data_in[final_indices]
    data_out_balanced = data_out[final_indices]

    print(f"New Balanced Dataset Size: {len(data_out_balanced)}")
    # Result: Instead of 13.5 million, you might have only ~1 million points!

    # 5. Convert to PyTorch tensors and Train
    t_in = torch.from_numpy(data_in_balanced).to(device)
    t_out = torch.from_numpy(data_out_balanced).to(device)

    # IMPORTANT: When calling nif_train, you NO LONGER need weighting.
    # The data is naturally balanced (50% ones, 50% zeros).
    # You must modify nif_train to remove 'pos_weight' or set it to 1.0.
    mlp = nif_train(t_in, t_out, BATCH_SIZE=10000)
    # ... (Rest of visualization code is identical) ...

# --------- MAIN ---------
if __name__ == "__main__":
    main()
