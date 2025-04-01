#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Import the local update functions from update.py
from update import (
    update_memberships,
    update_bias,
    update_class_means
)

def create_gaussian_filter(fsize=3, sigma=1.0):
    """
    Create a 2D Gaussian filter of size (fsize x fsize) with standard deviation sigma.
    Ensures the sum of all values is 1.
    """
    # fsize should be odd
    ax = np.arange(-fsize//2 + 1, fsize//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def compute_objective(Y, b, U, c_means, filter_2d, q, M):
    """
    Compute the modified-FCM objective function:
    
    J = sum_{(r,c)} sum_{k=1..K} [ U(r,c,k)^q * d_{k}(r,c) ],
    
    where
      d_{k}(r,c) = sum over filter neighborhood => w * [ Y(r,c) - c_k * b(neighbor) ]^2.
    """
    H, W, K = U.shape
    f = filter_2d.shape[0]
    half = f // 2
    
    Uq = U**q
    cost = 0.0
    eps = 1e-12

    for r in range(H):
        for c in range(W):
            if M[r, c] == 1:
                for k_ in range(K):
                    # sum over local neighborhood for d_k(r,c)
                    d_val = 0.0
                    for dy in range(-half, half+1):
                        for dx in range(-half, half+1):
                            rr = r + dy
                            cc = c + dx
                            if 0 <= rr < H and 0 <= cc < W and M[rr, cc] == 1:
                                w_rc = filter_2d[dy+half, dx+half]
                                diff = Y[r,c] - c_means[k_] * b[rr, cc]
                                d_val += w_rc * (diff**2)
                    
                    cost += Uq[r,c,k_] * d_val

    return cost

def main():
    # 2. LOAD DATA
    # -----------------------------------------------------------------------
    data_folder = "data"
    membership_folder = "membership_init"
    
    image_path = os.path.join(data_folder, "imageData.npy")
    mask_path = os.path.join(data_folder, "imageMask.npy")
    membership_path = os.path.join(membership_folder, "membership.npy")
    
    # Load the brain MR image (H,W), brain mask (H,W), and initial membership (H,W,K)
    Y = np.load(image_path)      # 2D array, shape (H, W)
    M = np.load(mask_path)       # 2D array, shape (H, W), 1=brain,0=outside
    U = np.load(membership_path) # 3D array, shape (H, W, K)
    
    # Basic checks
    H, W = Y.shape
    if U.shape[0] != H or U.shape[1] != W:
        raise ValueError("Loaded membership shape does not match the image shape!")
    K = U.shape[2]
    print(f"Image shape: {Y.shape}, Mask shape: {M.shape}, Membership shape: {U.shape}")

    # 3. INITIALIZE BIAS FIELD AND CLASS MEANS
    # -----------------------------------------------------------------------
    # Set the bias to 1 for brain pixels, 0 otherwise
    b = np.zeros_like(Y, dtype=np.float32)
    b[M == 1] = 1.0

    # We have K classes (e.g., 3).
    # Provide an initial guess for c_means.
    c_means = np.array([0.22, 0.45, 0.63], dtype=np.float32)
    if K != len(c_means):
        raise ValueError("Number of classes (K) != length of initial c_means!")

    # 4. CREATE GAUSSIAN WEIGHT FILTER
    # -----------------------------------------------------------------------
    fsize = 5
    sigma = 3.25
    filter_2d = create_gaussian_filter(fsize, sigma)
    print("Filter:\n", filter_2d)

    # 5. ITERATIVE UPDATES
    # -----------------------------------------------------------------------
    q = 2       # Fuzziness exponent
    max_iters = 10
    tol = 1e-3

    # We'll store objective values to plot them later
    objective_values = []

    for it in range(max_iters):
        U_old = U.copy()
        b_old = b.copy()
        c_old = c_means.copy()

        # Update bias
        b = update_bias(Y, U, b, c_means, filter_2d, q, M)
        # Update memberships
        U = update_memberships(Y, b, U, c_means, filter_2d, q, M)
        # Update class means
        c_means = update_class_means(Y, U, b, c_means, filter_2d, q, M)

        # Compute the objective function value
        J_current = compute_objective(Y, b, U, c_means, filter_2d, q, M)

        # (i) Renormalize b, c
        # b_avg = np.mean(b)  # average of the bias field
        # if b_avg < 1e-12:
        #     b_avg = 1e-12  # protect from zero
        # b = b / b_avg      # divide each b_n by average
        # c = c * b_avg      # multiply each c_k by the average

        objective_values.append(J_current)

        # Check for convergence
        diff_U = np.max(np.abs(U - U_old))
        diff_b = np.max(np.abs(b - b_old))
        diff_c = np.max(np.abs(c_means - c_old))
        diff_max = max(diff_U, diff_b, diff_c)
        print(f"Iteration {it+1}: max change = {diff_max:.6f}, objective = {J_current:.6f}")
        
        if diff_max < tol:
            print(f"Converged at iteration {it+1} (max change < {tol})")
            break

    # 6. SAVE RESULTS & VISUALIZE
    # -----------------------------------------------------------------------
    # Create output folders if they don't exist
    os.makedirs("images", exist_ok=True)
    os.makedirs("optimal_parameter", exist_ok=True)

    # Save final parameters
    np.save(os.path.join("optimal_parameter", "final_membership.npy"), U)
    np.save(os.path.join("optimal_parameter", "final_bias.npy"), b)
    np.save(os.path.join("optimal_parameter", "final_class_means.npy"), c_means)
    print("Saved final results (membership, bias, class_means) in 'optimal_parameter' folder.")
    
    # Print final class means
    print("Optimal Class Means (final c_means) =", c_means)

    # Plot the objective function vs. iteration and save in images folder
    plt.figure()
    plt.plot(range(1, len(objective_values)+1), objective_values, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Modified-FCM Objective vs. Iteration")
    plt.grid(True)
    plt.savefig(os.path.join("images", "objective_plot.png"), dpi=120)
    plt.show()

    # Plot & save final bias in images folder
    plt.figure()
    plt.imshow(b, cmap='gray')
    plt.title("Final Bias Field")
    plt.colorbar()
    plt.savefig(os.path.join("images", "final_bias.png"), dpi=120)
    plt.show()

    classes = ["CSF", "GreyMatter", "WhiteMatter"]

    # Plot & save membership maps in images folder
    for k_ in range(K):
        plt.figure()
        plt.imshow(U[:,:,k_], cmap='gray')
        plt.title(f"Final Membership: Class {classes[k_]}")
        plt.colorbar()
        plt.savefig(os.path.join("images", f"final_membership_class{classes[k_]}.png"), dpi=120)
        plt.show()

if __name__ == "__main__":
    main()
