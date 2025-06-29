import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

#
# Entropy-regularised optimal transport layer
#
# y = argmin_u f(x, u; mu)
#     subject to h(u) = 0
#            and u_i >= 0
#
# where f(x, u) = \sum_{i=1}^n (x_i * u_i + mu * u_i * (log(u_i) - 1))
# and h(u) = Au - b
# for A = [
#          e_1 x n, ..., e_m x n,
#          I_n, ..., I_n
#         ]_{-1,.}
# where [.]_{-1,.} denotes removing the first row of the matrix
#
# Dylan Campbell, Liu Liu, Stephen Gould, 2020,
# "Solving the Blind Perspective-n-Point Problem End-To-End With
# Robust Differentiable Geometric Optimization"
#
# Dylan Campbell <dylan.campbell@anu.edu.au>
#
# v0: 20191018
#

class RegularisedTransportFn(torch.autograd.Function):
    """ Class for solving the entropy-regularised transport problem
    
    Finds the transport (or joint probability) matrix P that is the
    smallest Sinkhorn distance from cost/distance matrix D
    
    Using:
    [1] Sinkhorn distances: Lightspeed computation of optimal transport
        Marco Cuturi, 2013
        Advances in Neural Information Processing Systems
        https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport.pdf
    """
    def objectiveFn(m, p, mu=0.1):
        """ Vectorised objective function

        Using:
            Equation (2) from [1]
        """
        logw = torch.where(p > 0.0, p.log(), torch.full_like(p, -1e9))
        return (p * m).sum(-1) + mu * (p * (logw - 1.0)).sum(-1)

    def sinkhorn(M, r=None, c=None, mu=0.1, tolerance=1e-9, iterations=None):
        """ Compute transport matrix P, given cost matrix M
        
        Using:
            Algorithm 1 from [1]
        """
        max_distance = 5.0
        K = (-M.clamp_max(max_distance) / mu).exp()
        if r is None:
            r = 1.0 / M.size()[-2]
            u = M.new_full(M.size()[:-1], r).unsqueeze(-1)
        else:
            r = r.unsqueeze(-1)
            u = r.clone()
        if c is None:
            c = 1.0 / M.size()[-1]
        else:
            c = c.unsqueeze(-1)
        if iterations is None:
            i = 0
            max_iterations = 100
            u_prev = torch.ones_like(u)
            while (u - u_prev).norm(dim=-1).max() > tolerance:
                if i > max_iterations:
                    break
                i += 1
                u_prev = u
                u = r / K.matmul(c / K.transpose(-2, -1).matmul(u))
        else:
            for i in range(iterations):
                u = r / K.matmul(c / K.transpose(-2, -1).matmul(u))
        v = c / K.transpose(-2, -1).matmul(u)
        P = (u * K) * v.transpose(-2, -1)
        return P

    def gradientFn(P, mu, v):
        """ Compute vector-Jacobian product DJ(M) = DJ(P) DP(M) [b x m*n]

        DP(M) = (H^-1 * A^T * (A * H^-1 * A^T)^-1 * A * H^-1 - H^-1) * B
        H = D_YY^2 f(x, y) = diag(mu / vec(P))
        B = D_XY^2 f(x, y) = I

        Using:
            Lemma 4.4 from
            Stephen Gould, Richard Hartley, and Dylan Campbell, 2019
            "Deep Declarative Networks: A New Hope", arXiv:1909.04866

        Arguments:
            P: (b, m, n) Torch tensor
                batch of transport matrices

            mu: float,
                regularisation factor

            v: (b, m*n) Torch tensor
                batch of gradients of J with respect to P

        Return Values:
            gradient: (b, m*n) Torch tensor,
                batch of gradients of J with respect to M

        """
        with torch.no_grad():
            b, m, n = P.size()
            B = P / mu
            hinv = B.flatten(start_dim=-2)
            d1inv = B.sum(-1)[:, 1:].reciprocal() # Remove first element
            d2 = B.sum(-2)
            B = B[:, 1:, :] # Remove top row
            S = -B.transpose(-2, -1).matmul(d1inv.unsqueeze(-1) * B)
            S[:, range(n), range(n)] += d2
            Su = torch.linalg.cholesky(S)
            Sinv = torch.zeros_like(S)
            for i in range (b):
                Sinv[i, ...] = torch.cholesky_inverse(Su[i, ...]) # Currently cannot handle batches
            R = -B.matmul(Sinv) * d1inv.unsqueeze(-1)
            Q = -R.matmul(B.transpose(-2, -1)  * d1inv.unsqueeze(-2))
            Q[:, range(m - 1), range(m - 1)] += d1inv
            # Build vector-Jacobian product from left to right:
            vHinv = v * hinv # bxmn * bxmn -> bxmn
            # Break vHinv into m blocks of n elements:
            u1 = vHinv.reshape((-1, m, n)).sum(-1)[:, 1:].unsqueeze(-2) # remove first element
            u2 = vHinv.reshape((-1, m, n)).sum(-2).unsqueeze(-2)
            u3 = u1.matmul(Q) + u2.matmul(R.transpose(-2, -1))
            u4 = u1.matmul(R) + u2.matmul(Sinv)
            u5 = u3.expand(-1, n, -1).transpose(-2, -1)+u4.expand(-1, m-1, -1)
            uHinv = torch.cat((u4, u5), dim=-2).flatten(start_dim=-2) * hinv
            gradient = uHinv - vHinv
        return gradient

    @staticmethod
    def forward(ctx, M, r=None, c=None, mu=0.1, tolerance=1e-9, iterations=None):
        """ Optimise the entropy-regularised Sinkhorn distance

        Solves:
            argmin_u   sum_{i=1}^n (x_i * u_i + mu * u_i * (log(u_i) - 1))
            subject to Au = 1, u_i >= 0 

        Using:
            Algorithm 1 from [1]
        
        Arguments:
            M: (b, m, n) Torch tensor,
                batch of cost matrices,
                assumption: non-negative

            mu: float,
                regularisation factor,
                assumption: positive,
                default: 0.1

            tolerance: float,
                stopping criteria for Sinkhorn algorithm,
                assumption: positive,
                default: 1e-9

            iterations: int,
                number of Sinkhorn iterations,
                assumption: positive,
                default: None

        Return Values:
            P: (b, m, n) Torch tensor,
                batch of transport (joint probability) matrices
        """
        M = M.detach()
        if r is not None:
            r = r.detach()
        if c is not None:
            c = c.detach()
        P = RegularisedTransportFn.sinkhorn(M, r, c, mu, tolerance, iterations)
        ctx.mu = mu
        ctx.save_for_backward(P, r, c)
        return P.clone()

    @staticmethod
    def backward(ctx, grad_output):
        P, r, c = ctx.saved_tensors
        mu = ctx.mu
        input_size = P.size()
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Only compute gradient for non-zero rows and columns of P
            if r is None or c is None or ((r > 0.0).all() and (c > 0.0).all()):
                grad_output = grad_output.flatten(start_dim=-2) # bxmn
                grad_input = RegularisedTransportFn.gradientFn(P, mu, grad_output) # bxmn
                grad_input = grad_input.reshape(input_size) # bxmxn
            else:
                b, m, n = input_size
                r_num_nonzero = (r > 0).sum(dim=-1)
                c_num_nonzero = (c > 0).sum(dim=-1)
                grad_input = torch.empty_like(P)
                for i in range(b):
                    p = r_num_nonzero[i]
                    q = c_num_nonzero[i]
                    grad_output_i = grad_output[i:(i+1), :p, :q].flatten(start_dim=-2) # bxpq
                    grad_input_i = RegularisedTransportFn.gradientFn(P[i:(i+1), :p, :q], mu, grad_output_i)
                    grad_input_i = grad_input_i.reshape((1, p, q))
                    grad_input_i = torch.nn.functional.pad(grad_input_i, (0, n - q, 0, m - p), "constant", 0.0)
                    grad_input[i:(i+1), ...] = grad_input_i
        return grad_input, None, None, None, None, None

class RegularisedTransport(torch.nn.Module):
    def __init__(self, mu=0.1, tolerance=1e-9, iterations=None):
        super(RegularisedTransport, self).__init__()
        self.mu = mu
        self.tolerance = tolerance
        self.iterations = iterations
            
    def forward(self, M, r=None, c=None):
        return RegularisedTransportFn.apply(M, r, c, self.mu, self.tolerance, self.iterations)

def pairwiseL2Dist(x1, x2):
    """ Computes the pairwise L2 distance between batches of feature vector sets

    res[..., i, j] = ||x1[..., i, :] - x2[..., j, :]||
    since 
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b

    Adapted to batch case from:
        jacobrgardner
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """
    x1_norm2 = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm2 = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm2.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm2).clamp_min_(1e-30).sqrt_()
    return res

def ransac_p3p(P, p2d, p3d, num_points_2d, num_points_3d):
    '''
        k:  1000
        P_topk_i:     torch.Size([1, 1000])
        p2d_indices:  torch.Size([1, 1000])
        p3d_indices:  torch.Size([1, 1000])
        p2d:  torch.Size([1, 3399, 3])
        p3d:  torch.Size([1, 6890, 3])
        P:  torch.Size([1, 3399, 6890])
        num_points_2d:  3399
        num_points_3d:  6890
    '''
    # 1. Choose top k correspondences:
    k = min(1000, round(1.5 * p2d.size(-2))) # Choose at most 1000 points
    _, P_topk_i = torch.topk(P.flatten(start_dim=-2), k=k, dim=-1, largest=True, sorted=True)
    p2d_indices = P_topk_i / P.size(-1) # bxk (integer division)
    p3d_indices = P_topk_i % P.size(-1) # bxk
    K = np.float32(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    dist_coeff = np.float32(np.array([0.0, 0.0, 0.0, 0.0]))
    theta0 = P.new_zeros((P.size(0), 6))

    # print("p2d: ", p2d.size())
    # print("p3d: ", p3d.size())
    # print("after P: ", P.size())

    # 2. Loop over batch and run RANSAC:
    # --- bug fix: there is only one batch
    for i in range(P.size(0)):
        num_points_ransac = min(k, round(1.5 * num_points_2d), round(1.5 * num_points_3d))
        num_points_ransac = min(k, max(num_points_ransac, 10)) # At least 10 points
        p2d_np = p2d[i, p2d_indices[i, :num_points_ransac].long(), :].cpu().numpy()
        p3d_np = p3d[i, p3d_indices[i, :num_points_ransac].long(), :].cpu().numpy()
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            p3d_np, p2d_np, K, dist_coeff,
            iterationsCount=1000,
            reprojectionError=0.01,
            flags=cv2.SOLVEPNP_P3P)
        # print(inliers.shape[0], '/',  num_points_2d[i].item())
        if rvec is not None and tvec is not None and retval:
            rvec = torch.as_tensor(rvec, dtype=P.dtype, device=P.device).squeeze(-1)
            tvec = torch.as_tensor(tvec, dtype=P.dtype, device=P.device).squeeze(-1)
            if torch.isfinite(rvec).all() and torch.isfinite(tvec).all():
                theta0[i, :3] = rvec
                theta0[i, 3:] = tvec
    return theta0