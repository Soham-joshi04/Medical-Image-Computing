import numpy as np

def update_memberships(Y, b, U, c_means, filter_2d, q, M):
    """
    Update memberships U(r,c,k) using local filter-based formula:
      d_{k}(r,c) = sum_{(rr,cc) in neighborhood of (r,c)} w_{(rr,cc),(r,c)} * [ Y(r,c) - c_k * b(rr,cc) ]^2
      Then:
      u(r,c,k) = ( (1 / d_{k}(r,c))^(1/(q-1)) ) / sum_{k'}( (1 / d_{k'}(r,c))^(1/(q-1)) ).
    
    Parameters
    ----------
    Y : 2D array (H, W)
    b : 2D array (H, W)
    U : 3D array (H, W, K)  [Will be overwritten with new memberships]
    c_means : 1D array (K,)
    filter_2d : 2D array (f, f)
    q : float
        Fuzziness exponent.
    M : 2D array (H, W) brain mask (1=brain, 0=background)
    
    Returns
    -------
    U_new : 3D array (H, W, K)
    """
    H, W, K = U.shape
    U_new = np.zeros_like(U)
    
    eps = 1e-12
    exponent = 1.0 / (q - 1.0)
    
    f = filter_2d.shape[0]
    half = f // 2
    
    for r in range(H):
        for c in range(W):
            if M[r,c] == 1:
                # Compute d_k for k in [0..K-1]
                dvals = np.zeros(K, dtype=np.float64)
                for k_ in range(K):
                    # sum over neighborhood
                    accum = 0.0
                    for dy in range(-half, half+1):
                        for dx in range(-half, half+1):
                            rr = r + dy
                            cc = c + dx
                            if 0 <= rr < H and 0 <= cc < W and M[rr, cc] == 1:
                                w_ij = filter_2d[dy+half, dx+half]
                                diff = Y[r,c] - c_means[k_] * b[rr, cc]
                                accum += w_ij * (diff**2)
                    if accum < eps:
                        accum = eps
                    dvals[k_] = accum
                
                # Now compute membership
                tmp = (1.0 / dvals)**exponent  # shape (K,)
                denom = np.sum(tmp)
                U_new[r,c,:] = tmp / denom
    
    return U_new


def update_bias(Y, U, b, c_means, filter_2d, q, M):
    """
    Update the bias field b(r,c) with local filter-based weighting.
    
    b(r,c) = [ sum_{(p,q) in neighborhood} w_{(r,c),(p,q)} * Y(p,q) * sum_k( U^q(p,q,k) * c_k ) ]
             -------------------------------------------------------------------------------
             [ sum_{(p,q) in neighborhood} w_{(r,c),(p,q)} * sum_k( U^q(p,q,k) * c_k^2 ) ]
    
    Parameters
    ----------
    Y : 2D array (H, W)
    U : 3D array (H, W, K)
    b : 2D array (H, W)  [will be replaced with new bias]
    c_means : 1D array (K,)
    filter_2d : 2D array (f, f)
    q : float
    M : 2D mask (H,W)
    
    Returns
    -------
    b_new : 2D array (H, W)
    """
    H, W, K = U.shape
    b_new = np.zeros_like(b)
    
    f = filter_2d.shape[0]
    half = f // 2
    
    Uq = U**q
    
    eps = 1e-12
    
    for r in range(H):
        for c in range(W):
            if M[r,c] == 1:
                numerator = 0.0
                denominator = 0.0
                # sum over local neighborhood
                for dy in range(-half, half+1):
                    for dx in range(-half, half+1):
                        rr = r + dy
                        cc = c + dx
                        if 0 <= rr < H and 0 <= cc < W and M[rr, cc] == 1:
                            w_ij = filter_2d[dy+half, dx+half]
                            
                            # sum_k(U^q(rr,cc,k) * c_means[k])
                            sum_uqc = 0.0
                            sum_uqc2 = 0.0
                            for k_ in range(K):
                                sum_uqc  += Uq[rr, cc, k_] * c_means[k_]
                                sum_uqc2 += Uq[rr, cc, k_] * (c_means[k_]**2)
                            
                            numerator   += w_ij * Y[rr, cc] * sum_uqc
                            denominator += w_ij * sum_uqc2
                if denominator < eps:
                    denominator = eps
                b_new[r,c] = numerator / denominator
    
    return b_new

def update_class_means(Y, U, b, c_means, filter_2d, q, M):
    """
    c_k = 
      [ sum_{(p,q)}( U^q(p,q,k) * Y(p,q) * sum_{(r,c) in nbr(p,q)}( w_{(r,c),(p,q)} * b(r,c) ) ] 
      --------------------------------------------------------------------------------------
      [ sum_{(p,q)}( U^q(p,q,k) * sum_{(r,c) in nbr(p,q)}( w_{(r,c),(p,q)} * b(r,c)^2 ) ]
    
    We do it in a double loop: over (p,q), then over the filter offsets (r,c).
    
    In practice, you might invert the logic or use convolution.  This direct approach
    is the "literal" interpretation of the formula.
    """
    H, W, K = U.shape
    Uq = U**q
    
    # We'll accumulate numerator_k and denominator_k for each k
    numerator = np.zeros(K, dtype=np.float64)
    denominator = np.zeros(K, dtype=np.float64)
    
    f = filter_2d.shape[0]
    half = f // 2
    
    eps = 1e-12
    
    for p in range(H):
        for q_ in range(W):
            if M[p, q_] == 1:
                # sum_k(U^q(p,q,k) * Y(p,q)) is easy:
                for k_ in range(K):
                    val_uq_y = Uq[p,q_,k_] * Y[p,q_]
                    
                    # now sum_{(r,c) in neighborhood of (p,q)} w_{(r,c),(p,q)} * b(r,c) (or b^2)
                    sum_wb = 0.0
                    sum_wb2 = 0.0
                    
                    for dy in range(-half, half+1):
                        for dx in range(-half, half+1):
                            rr = p + dy
                            cc = q_ + dx
                            if (0 <= rr < H) and (0 <= cc < W) and (M[rr, cc] == 1):
                                w_rc_pq = filter_2d[dy+half, dx+half]
                                sum_wb  += w_rc_pq * b[rr, cc]
                                sum_wb2 += w_rc_pq * (b[rr, cc]**2)
                    
                    numerator[k_]   += (val_uq_y * sum_wb)
                    denominator[k_] += (Uq[p,q_,k_] * sum_wb2)
    
    # Now c_k = numerator[k] / denominator[k]
    for k_ in range(K):
        if denominator[k_] < eps:
            denominator[k_] = eps
        c_means[k_] = numerator[k_] / denominator[k_]
    
    return c_means

