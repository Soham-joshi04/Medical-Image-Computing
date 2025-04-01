import os
import numpy as np
import matplotlib.pyplot as plt

# Import your update functions from update.py, e.g.
# from update import update_memberships, update_means, update_stds, icm_label_optimization

def compute_log_posterior(image_data, image_mask, label_image, means, stds, beta):
    """
    Compute the log posterior = sum of log-likelihood + MRF prior term.
    Using 8-neighborhood and 1-based class labels in 'label_image'.
    """
    H, W = image_data.shape
    K = len(means)

    log_post = 0.0
    gauss_const = -0.5 * np.log(2.0 * np.pi)

    def get_8_neighbors(i, j):
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    neighbors.append((ni, nj))
        return neighbors

    # Sum of log-likelihood terms
    for i in range(H):
        for j in range(W):
            if image_mask[i, j] == 1:
                k_label = label_image[i, j] - 1  # convert 1-based -> 0-based
                x_ij = image_data[i, j]
                mu = means[k_label]
                std = max(stds[k_label], 1e-6)  # safeguard from zero

                diff = (x_ij - mu)/std
                log_likelihood = gauss_const - np.log(std) - 0.5*(diff**2)
                log_post += log_likelihood

    # MRF prior term: penalty for label discontinuities => - beta * (#discontinuities)
    # Here we do a half counting approach. Alternatively, you can do full adjacency but watch duplicates.
    visited = np.zeros_like(label_image, dtype=bool)
    for i in range(H):
        for j in range(W):
            if image_mask[i, j] == 1 and not visited[i, j]:
                neighbors = get_8_neighbors(i, j)
                for (ni, nj) in neighbors:
                    if image_mask[ni, nj] == 1:
                        if label_image[i, j] != label_image[ni, nj]:
                            log_post -= beta
                visited[i, j] = True

    return log_post

def main():
    # 1) Create output directories
    os.makedirs('optimal', exist_ok=True)
    os.makedirs('images', exist_ok=True)

    # 2) Load data, mask, and initial parameters
    image_data = np.load('data/imageData.npy')
    image_mask = np.load('data/imageMask.npy')

    # We'll assume we have an initial label image in terms of cluster means, or discrete labels.
    init_label_means_path = 'initialise/initial_label_image_means.npy'
    init_means_path = 'initialise/initial_means.npy'
    init_stds_path = 'initialise/initial_stds.npy'

    if not (os.path.exists(init_label_means_path) and 
            os.path.exists(init_means_path) and 
            os.path.exists(init_stds_path)):
        raise FileNotFoundError("Missing initial files in 'initialise/' folder.")

    init_label_image_means = np.load(init_label_means_path)  # shape: (H,W), intensities in {0, mu_1, mu_2, mu_3}
    init_means = np.load(init_means_path)   # shape: (K,)
    init_stds = np.load(init_stds_path)     # shape: (K,)

    # Convert from 3-level grayscale + 0 => discrete labels {0=BG, 1,2,3=classes}
    H, W = image_data.shape
    label_image = np.zeros((H, W), dtype=int)
    for i in range(H):
        for j in range(W):
            if image_mask[i, j] == 1:
                val = init_label_image_means[i, j]
                if val != 0.0:
                    # find which of init_means is closest
                    diffs = np.abs(init_means - val)
                    idx = np.argmin(diffs)
                    label_image[i, j] = idx + 1  # 1-based label
                # else remain 0 = background

    # 3) Select MRF parameter beta
    beta = 0  # example
    beta_str = str(beta).replace('.', '_')  # "0_2"

    # 4) EM + MRF iterations
    max_iter = 10
    log_post_before = []
    log_post_after = []

    # Load the module with the update functions
    import update

    means = init_means.copy()
    stds = init_stds.copy()

    for it in range(max_iter):
        print(f"\nIteration {it+1}/{max_iter}")

        # (a) Update memberships
        memberships = update.update_memberships(
            image_data, image_mask, label_image, means, stds, beta
        )

        # (b) Update means
        new_means = update.update_means(image_data, image_mask, memberships)

        # (c) Update stds
        new_stds = update.update_stds(image_data, image_mask, memberships, new_means)

        # Compute log posterior BEFORE ICM
        lp_before = compute_log_posterior(image_data, image_mask, label_image, new_means, new_stds, beta)
        log_post_before.append(lp_before)

        # (d) Label update => modified ICM
        new_label_image = update.icm_label_optimization(
            image_data, image_mask, label_image, new_means, new_stds, beta, max_iter=1
        )

        # Compute log posterior AFTER ICM
        lp_after = compute_log_posterior(image_data, image_mask, new_label_image, new_means, new_stds, beta)
        log_post_after.append(lp_after)

        # Check if labeling changed
        changed = (new_label_image != label_image).any()

        # Accept new label image + new parameters
        label_image = new_label_image
        means = new_means
        stds = new_stds

        print(f"  LogPosterior BEFORE ICM: {lp_before:.3f}")
        print(f"  LogPosterior AFTER  ICM: {lp_after:.3f}")
        print(f"  #LabelChanged = {changed}")

        if not changed:
            print("No label changes => labeling converged.")
            break

    # 5) Save final (optimal) parameters
    np.save(f'optimal/final_means_beta{beta_str}.npy', means)
    np.save(f'optimal/final_stds_beta{beta_str}.npy', stds)
    np.save(f'optimal/final_label_image_beta{beta_str}.npy', label_image)

    # 6) Save final membership images (one per class)
    # Recompute final memberships with final labels (some methods do it that way),
    # or just keep the 'memberships' from the last iteration above. We'll do the latter:
    final_memberships = memberships

    for k_idx in range(len(means)):
        plt.figure()
        # membership is in [0,1], so grayscale is typical
        plt.imshow(final_memberships[:, :, k_idx], cmap='gray', vmin=0, vmax=1)
        plt.title(f'Optimal Membership (Class {k_idx+1}) - beta={beta}')
        plt.colorbar()
        out_path = f'images/final_membership_class{k_idx+1}_beta{beta_str}.png'
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    H, W = image_data.shape
    seg_image = np.zeros((H, W), dtype=float)

    for i in range(H):
        for j in range(W):
            if image_mask[i, j] == 1:
                lab = label_image[i, j]
                if lab > 0:  # if it's one of the classes 1..3
                    seg_image[i, j] = means[lab - 1]
            # else 0 remains (background)

    # Now plot in grayscale with discrete ticks at [0, mu_1, mu_2, mu_3].
    plt.figure(figsize=(6, 5))
    plt.imshow(seg_image, vmin=0, vmax=np.max(means))
    plt.title(f"Final Segmentation\n(0=Background, Means for {len(means)} Clusters, β={beta})")
    plt.axis('off')

    # Build a list of tick positions: 0 (BG) + the three final means
    tick_vals = [0] + list(means)
    tick_labels = ["Background"] + [f"Class {i+1}\n(Mean={m:.2f})"
                                    for i, m in enumerate(means)]

    # Create colorbar with those discrete ticks
    cbar = plt.colorbar(ticks=tick_vals)
    cbar.ax.set_yticklabels(tick_labels)

    out_file = f'images/final_label_image_beta{beta_str}.png'
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

    print(f"Saved final segmentation with discrete means to: {out_file}")


    # 8) Plot log posterior sequences
    plt.figure()
    iters = np.arange(len(log_post_before)) + 1
    plt.plot(iters, log_post_before, 'ro-', label='Before ICM')
    plt.plot(iters, log_post_after, 'bo-', label='After ICM')
    plt.xlabel('Iteration')
    plt.ylabel('Log Posterior')
    plt.title(f'Log Posterior Before/After ICM (beta={beta})')
    plt.legend()
    plt.savefig(f'images/log_posterior_beta{beta_str}.png', bbox_inches='tight')
    plt.close()

    print("\nTraining complete. Final parameters and images saved.")

if __name__ == "__main__":
    main()
