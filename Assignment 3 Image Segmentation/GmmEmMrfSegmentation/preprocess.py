import os
import numpy as np
import h5py
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    os.makedirs('initialise', exist_ok=True)

    # Path to the MATLAB v7.3 file
    mat_path = os.path.join('data', 'assignmentSegmentBrainGmmEmMrf.mat')
    
    # 1. Read .mat file with h5py
    with h5py.File(mat_path, 'r') as f:
        # Adjust keys if different in your file
        image_data_h5 = f['imageData']
        image_mask_h5 = f['imageMask']
        
        # Convert HDF5 datasets to NumPy arrays; transpose if needed
        image_data = np.array(image_data_h5).T
        image_mask = np.array(image_mask_h5).T

    print("image_data shape:", image_data.shape)
    print("image_mask shape:", image_mask.shape)

    # 2. Save arrays as .npy
    np.save(os.path.join('data', 'imageData.npy'), image_data)
    np.save(os.path.join('data', 'imageMask.npy'), image_mask)

    # Visualize the image
    plt.imshow(image_data, cmap='gray')
    plt.savefig(os.path.join('images', 'original_image.png'))
    plt.close()

    # Visualize the mask    
    plt.imshow(image_mask, cmap='gray')
    plt.savefig(os.path.join('images', 'original_mask.png'))
    plt.close()

    # 3. K-means initialization on the brain region only
    #    (mask == 1 indicates inside-brain pixels)
    brain_pixels = image_data[image_mask == 1].reshape(-1, 1)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(brain_pixels)
    labels_1d = kmeans.labels_  # in {0, 1, 2}

    # Create discrete label image (0=background, 1..3=brain classes)
    label_image = np.zeros_like(image_data, dtype=int)
    label_image[image_mask == 1] = labels_1d + 1  # shift to {1, 2, 3}

    # 4. Compute initial means (µ_k) and stds (σ_k)
    init_means = np.zeros(k)
    init_stds = np.zeros(k)
    for cluster_id in range(k):
        cluster_vals = brain_pixels[labels_1d == cluster_id].flatten()
        init_means[cluster_id] = cluster_vals.mean()
        init_stds[cluster_id] = cluster_vals.std(ddof=1)

    # Save initial parameters
    np.save(os.path.join('initialise', 'initial_means.npy'), init_means)
    np.save(os.path.join('initialise', 'initial_stds.npy'), init_stds)

    # 5. Create a 3-level grayscale image (plus background):
    #    - background => intensity 0
    #    - cluster i => intensity = µ_i
    seg_image = np.zeros_like(image_data, dtype=float)  # float for mean intensities
    for cluster_id in range(k):
        # label_image == cluster_id+1 => cluster i
        seg_image[label_image == cluster_id + 1] = init_means[cluster_id]

    # 6. Save the segmentation image as a NumPy array if desired
    np.save(os.path.join('initialise', 'initial_label_image_means.npy'), seg_image)

    # 7. Visualization: 
    #    - Only 4 distinct intensities in seg_image: 0 (BG), µ1, µ2, µ3.
    #    - Grayscale colormap, with a discrete colorbar labeling each tick.
    if image_data.ndim == 2:
        plt.figure(figsize=(6, 5))
        plt.imshow(seg_image, vmin=0, vmax=np.max(init_means))
        plt.title("K-means Initialization \n(0=Background, Means for 3 Clusters)")
        plt.axis('off')

        # Build a sorted list of all intensities for colorbar ticks:
        # Start with 0 for background, then each mean in cluster order
        # If you want them strictly ascending, you could sort the means,
        # but then the cluster labels might not match. We'll keep the order as is.
        tick_vals = [0] + list(init_means)
        tick_labels = ["Background"] + [f"Class {i+1}\n(Mean={m:.2f})" 
                                        for i, m in enumerate(init_means)]
        
        # Create colorbar with discrete ticks at these intensities
        cbar = plt.colorbar(ticks=tick_vals)
        cbar.ax.set_yticklabels(tick_labels)

        plt.savefig(os.path.join('images', 'initial_label_image.png'),
                    bbox_inches='tight')
        plt.close()

    # 8. Plot histogram + Gaussians
    plt.figure()
    brain_vals = brain_pixels.flatten()
    plt.hist(brain_vals, bins=100, density=True, alpha=0.5, label="Brain Intensities")
    x_vals = np.linspace(brain_vals.min(), brain_vals.max(), 300)

    for cluster_id in range(k):
        mean_k = init_means[cluster_id]
        std_k = init_stds[cluster_id] if init_stds[cluster_id] >= 1e-6 else 1e-6
        gauss_curve = (
            1.0 / (np.sqrt(2.0 * np.pi) * std_k)
            * np.exp(-0.5 * ((x_vals - mean_k) / std_k) ** 2)
        )
        plt.plot(x_vals, gauss_curve, linewidth=2, label=f"Class {cluster_id+1} (µ={mean_k:.2f})")

    plt.title("Histogram + Initial Gaussians (K-means)")
    plt.legend()
    plt.savefig(os.path.join('images', 'initial_gaussians.png'), bbox_inches='tight')
    plt.close()

    print("Initial Means:", init_means)
    print("Initial Stds:", init_stds)
    print("Initialization complete. Outputs in 'data/', 'images/', and 'initialise/' folders.")

if __name__ == "__main__":
    main()
