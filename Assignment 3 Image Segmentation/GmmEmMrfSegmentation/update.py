import numpy as np

def get_8_neighbors(i, j, mask):
    """
    Return the list of (r, c) of valid 8-neighbors around (i, j).
    This includes horizontal, vertical, and diagonal neighbors.
    """
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue  # skip the pixel itself
            ni = i + di
            nj = j + dj
            if  mask[ni][nj] == 1:
                neighbors.append((ni, nj))
    return neighbors


def update_memberships(image_data, image_mask, label_image, means, stds, beta):
    """
    Update the soft memberships alpha_{i,k} for each pixel i and class k,
    based on:
      - Gaussian likelihood
      - MRF prior from discrete label_image (8-neighbors)
      - Only update inside image_mask==1
    Returns an array 'memberships' with shape (H, W, K).
    """
    height, width = image_data.shape
    K = len(means)  # number of classes
    
    # Initialize memberships
    memberships = np.zeros((height, width, K), dtype=np.float64)
    
    # Precompute Gaussian factor for each class k at intensity x:
    # p(x|mu_k, std_k) = 1/(sqrt(2*pi)*std_k) * exp(-0.5*(x - mu_k)^2 / std_k^2)
    # We'll do that on-the-fly for each pixel to handle large images.

    # Loop over all pixels
    for i in range(height):
        for j in range(width):
            # Only update inside the mask
            if image_mask[i, j] == 1:
                x_ij = image_data[i, j]
                
                # count how many neighbors have label k
                neighbors = get_8_neighbors(i, j, image_mask)
                
                # Compute unnormalized alpha_{i,k} for each k
                unnorm = np.zeros(K, dtype=np.float64)
                for k in range(K):
                    mu_k = means[k]
                    std_k = stds[k]
                    if std_k < 1e-6:
                        std_k = 1e-6  # safeguard from zero-division
                    # Gaussian likelihood
                    gauss_val = (1.0 / (np.sqrt(2.0*np.pi)*std_k)) * \
                                np.exp(-0.5 * ((x_ij - mu_k)/std_k)**2)
                    
                    # MRF prior factor: exponent of beta * (#neighbors labeled k)
                    # label_image might store {1..K}, so compare with k+1 if needed
                    label_k = k+1  # if labels in image are 1-based
                    neighbor_count = 0
                    for (ni, nj) in neighbors:
                        if label_image[ni, nj] == label_k:
                            neighbor_count += 1
                    
                    mrf_factor = np.exp(beta * neighbor_count)
                    
                    unnorm[k] = gauss_val * mrf_factor
                
                # Normalize
                denom = np.sum(unnorm)
                if denom > 0:
                    memberships[i, j, :] = unnorm / denom
                else:
                    # if for some reason denom=0 (numerical underflow), fallback uniform
                    memberships[i, j, :] = 1.0 / K
            # else outside mask: keep memberships as all zeros

    return memberships


def update_means(image_data, image_mask, memberships):
    """
    Compute new means for each class k using the membership weights.
    memberships shape: (H, W, K)
    Return an array means[k].
    """
    height, width, K = memberships.shape
    
    new_means = np.zeros(K, dtype=np.float64)
    
    for k in range(K):
        num = 0.0
        den = 0.0
        for i in range(height):
            for j in range(width):
                if image_mask[i, j] == 1:
                    w = memberships[i, j, k]
                    x = image_data[i, j]
                    num += w * x
                    den += w
        if den > 1e-12:
            new_means[k] = num / den
        else:
            new_means[k] = 0.0  # or fallback
    
    return new_means

def update_stds(image_data, image_mask, memberships, means):
    """
    Compute new standard deviations for each class k using membership weights
    and the newly updated means.
    memberships shape: (H, W, K)
    Return an array stds[k].
    """
    height, width, K = memberships.shape
    
    new_stds = np.zeros(K, dtype=np.float64)
    
    for k in range(K):
        num = 0.0
        den = 0.0
        mu_k = means[k]
        for i in range(height):
            for j in range(width):
                if image_mask[i, j] == 1:
                    w = memberships[i, j, k]
                    x = image_data[i, j]
                    diff = x - mu_k
                    num += w * (diff * diff)
                    den += w
        if den > 1e-12:
            var = num / den
            new_stds[k] = np.sqrt(var)
        else:
            new_stds[k] = 1.0  # fallback
    
    return new_stds


import numpy as np

def update_labels_icm_once(
    image_data,   # 2D array of intensities
    image_mask,   # 2D array, 1=brain, 0=outside
    label_image,  # 2D array of current labels (1..K inside mask, 0 outside)
    means,        # array of shape (K,) => mu_k
    stds,         # array of shape (K,) => sigma_k
    beta,         # MRF parameter
    ):
    """
    Perform one 'modified ICM' update over the entire label field simultaneously.
    Returns a new label image (same shape) that potentially improves the posterior.

    - 8-neighborhood is used.
    - For each pixel i in the mask, we compute the 'score' for each class k 
      and pick argmax_k. 
    - All label updates happen simultaneously.
    """
    H, W = image_data.shape
    K = len(means)  # number of classes
    new_label_image = np.copy(label_image)  # to store updated labels
    
    # Loop over pixels in the mask
    for i in range(H):
        for j in range(W):
            if image_mask[i, j] == 1:
                x_ij = image_data[i, j]
                best_label = label_image[i, j]  # start with current label
                best_score = -np.inf
                
                # Pre-compute neighbor labels for MRF sum
                neighbors = get_8_neighbors(i, j, image_mask)
                
                for k_idx in range(K):
                    # k_idx is 0-based internally, actual label is (k_idx+1)
                    label_val = k_idx + 1
                    mu_k = means[k_idx]
                    std_k = stds[k_idx]
                    if std_k < 1e-6:
                        std_k = 1e-6  # avoid zero-division
                        
                    # log-likelihood
                    # p(x|mu,sigma) = (1/(sqrt(2*pi)*sigma)) * exp(-0.5*((x-mu)/sigma)^2)
                    # => log p(x|mu,sigma) = -log(sigma) - 0.5*((x-mu)/sigma)^2 - log(sqrt(2*pi))
                    log_likelihood = (-np.log(std_k) 
                                      - 0.5 * ((x_ij - mu_k)/std_k)**2 
                                      - 0.5 * np.log(2.0*np.pi))
                    
                    # sum of neighbors that have label = label_val
                    neighbor_count = 0
                    for (ni, nj) in neighbors:
                        if label_image[ni, nj] == label_val:
                            neighbor_count += 1
                    
                    # MRF prior term
                    # => beta * (#neighbors labeled k)
                    mrf_term = beta * neighbor_count
                    
                    score_k = log_likelihood + mrf_term
                    
                    # pick best label
                    if score_k > best_score:
                        best_score = score_k
                        best_label = label_val
                
                # After checking all classes, assign best_label
                new_label_image[i, j] = best_label
    
    return new_label_image

def icm_label_optimization(image_data, image_mask, label_image, means, stds, beta, max_iter=10):
    for iteration in range(max_iter):
        new_label_image = update_labels_icm_once(
            image_data, image_mask, label_image, means, stds, beta
        )
        # Check if labeling changed
        if np.array_equal(new_label_image, label_image):
            print(f"ICM converged at iteration {iteration+1}.")
            break
        label_image = new_label_image
    return label_image
