# 优化了 LoS_angle_est_byhand 函数的计算效率
import numpy as np
import torch
import math
from scipy.special import iv as besseli
from scipy.linalg import kron

def LoS_angle_iniest(y, sysPar):
    y_copy = y
    Nx = sysPar['Nx']
    Ny = sysPar['Ny']
    UE_high = sysPar['UE_posZ']
    BS_pos = sysPar['BS_pos']
    lam = 1.0
    d = lam / 2

    # 定义方向余弦网格
    angle_grid_x = np.linspace(-1, 1, 2 * Nx)
    angle_grid_y = np.linspace(-1, 1, 2 * Ny)

    # 网格化并筛选单位球内方向
    grid_mu_x, grid_mu_y = np.meshgrid(angle_grid_x, angle_grid_y, indexing='ij')
    valid_mask = (grid_mu_x**2 + grid_mu_y**2) < 1
    grid_mu_x = grid_mu_x[valid_mask]
    grid_mu_y = grid_mu_y[valid_mask]

    if len(grid_mu_x) == 0:
        raise ValueError("No valid (mu_x, mu_y) in unit circle.")

    # 批量构建 phase_x 和 phase_y
    idx_x = np.arange(Nx)
    idx_y = np.arange(Ny)

    # 批量构造 phase_x 和 phase_y
    phase_x = np.exp(1j * 2 * np.pi / lam * d * idx_x[None, :] * grid_mu_x[:, None])  # [N, Nx]
    phase_y = np.exp(1j * 2 * np.pi / lam * d * idx_y[None, :] * grid_mu_y[:, None])  # [N, Ny]

    # 批量构造 steering 向量（Kronecker 积）
    steering = np.einsum('ni,nj->nij', phase_y, phase_x)  # 外积：[N, Ny, Nx]
    steering = steering.reshape(len(grid_mu_x), -1)  # [N, Ny*Nx]

    # 计算匹配度指标：能量形式（abs^2）
    y_flat = y_copy.flatten()  # [Ny*Nx]
    metrics = np.abs(np.dot(steering, y_flat.conj())) ** 2  # [N]

    # 找最佳方向
    best_idx = np.argmax(metrics)
    best_mu_x = grid_mu_x[best_idx]
    best_mu_y = grid_mu_y[best_idx]

    # 推 UE 位置（固定高度）
    mu_z = np.sqrt(max(0, 1 - best_mu_x**2 - best_mu_y**2))

    dir_vec = (
        best_mu_x * np.array([1, 0, 0]) +
        best_mu_y * np.array([0, 0, 1]) +
        mu_z * np.array([0, 1, 0])
    )
    dir_vec /= np.linalg.norm(dir_vec)

    t = (UE_high - BS_pos[2]) / dir_vec[2]
    UE_pos = BS_pos + t * dir_vec

    return UE_pos

def LoS_angle_iniest_tensor(y, sysPar):
    device = y.device
    BS_pos = torch.tensor(sysPar['BS_pos'], device=device, dtype=torch.float32)
    Nx = sysPar['Nx']
    Ny = sysPar['Ny']
    UE_high = sysPar['UE_posZ']
    BS_ex = torch.tensor(sysPar['BS_ex'], device=device, dtype=torch.float32)
    BS_ey = torch.tensor(sysPar['BS_ey'], device=device, dtype=torch.float32)
    BS_ez = torch.cross(BS_ey, BS_ex, dim=-1)
    lam = 1.0  # 归一化波长
    d = lam / 2

    angle_grid_x = torch.linspace(-1, 1, 2 * Nx, device=device, dtype=torch.float32)
    angle_grid_y = torch.linspace(-1, 1, 2 * Ny, device=device, dtype=torch.float32)

    # 网格点筛选：保留单位圆内
    grid_mu_x, grid_mu_y = torch.meshgrid(angle_grid_x, angle_grid_y, indexing='ij')
    valid_mask = (grid_mu_x ** 2 + grid_mu_y ** 2) < 1
    grid_mu_x = grid_mu_x[valid_mask]
    grid_mu_y = grid_mu_y[valid_mask]

    if len(grid_mu_x) == 0:
        raise ValueError("No valid (mu_x, mu_y) in unit circle.")

    # 批量构建导向向量
    phase_x = torch.exp(1j * 2 * math.pi / lam * d * torch.arange(Nx, device=device).unsqueeze(0) * grid_mu_x.unsqueeze(-1))  # [N, Nx]
    phase_y = torch.exp(1j * 2 * math.pi / lam * d * torch.arange(Ny, device=device).unsqueeze(0) * grid_mu_y.unsqueeze(-1))  # [N, Ny]
    
    # 外积：Kronecker product 批量实现
    phase_x = phase_x.unsqueeze(1)  # [N, 1, Nx]
    phase_y = phase_y.unsqueeze(2)  # [N, Ny, 1]
    steering = torch.matmul(phase_y, phase_x)  # [N, Ny, Nx]
    steering = steering.view(-1, Ny * Nx)  # [N, Ny*Nx]

    # reshape y
    y_flat = y.view(-1)  # [Ny*Nx]

    # 批量计算匹配度指标
    y_conj = torch.conj(y_flat)
    metrics = torch.abs(torch.matmul(steering, y_conj)) ** 2

    # 找最优方向
    best_idx = torch.argmax(metrics)
    best_mu_x = grid_mu_x[best_idx].clone()
    best_mu_y = grid_mu_y[best_idx].clone()

    # 推 UE 位置（固定高度）
    
    mu_z = torch.sqrt(1 - best_mu_x ** 2 - best_mu_y ** 2)

    # 构造方向向量（注意坐标轴定义）
    dir_vec = (
        best_mu_x * BS_ex +
        best_mu_y * BS_ey +
        mu_z * BS_ez
    )

    dir_vec = dir_vec / torch.norm(dir_vec)

    t = (UE_high - BS_pos[2]) / dir_vec[2]
    UE_pos = BS_pos + t * dir_vec
    return UE_pos

def LoS_angle_iniest_1D(y, sysPar):
    device = y.device
    BS_pos = torch.tensor(sysPar['BS_pos'], device=device, dtype=torch.float32)
    Nx = sysPar['Nx']
    Ny = sysPar['Ny']
    UE_high = sysPar['UE_posZ']
    BS_ex = torch.tensor(sysPar['BS_ex'], device=device, dtype=torch.float32)
    BS_ey = torch.tensor(sysPar['BS_ey'], device=device, dtype=torch.float32)
    BS_ez = torch.cross(BS_ey, BS_ex, dim=-1)
    lam = 1.0  # 归一化波长
    d = lam / 2

    angle_grid_x = torch.linspace(-1, 1, 2 * Nx, device=device, dtype=torch.float32)
    angle_grid_y = torch.linspace(-1, 1, 2 * Ny, device=device, dtype=torch.float32)


    # 批量构建导向向量
    phase_x = torch.exp(1j * 2 * math.pi / lam * d * torch.arange(Nx, device=device).unsqueeze(0) * angle_grid_x.unsqueeze(-1))  # [N, Nx]
    phase_y = torch.exp(1j * 2 * math.pi / lam * d * torch.arange(Ny, device=device).unsqueeze(0) * angle_grid_y.unsqueeze(-1))  # [N, Ny]
    
    metricX = torch.sum(torch.abs(torch.matmul(y, torch.conj(phase_x).T)) ** 2,dim=0)
    metricY = torch.sum(torch.abs(torch.matmul(torch.conj(phase_y),y)) ** 2,dim=1)

    # 找最优方向
    best_idxX = torch.argmax(metricX)
    best_idxY = torch.argmax(metricY)
    best_mu_x = angle_grid_x[best_idxX].clone()
    best_mu_y = angle_grid_y[best_idxY].clone()

    # 推 UE 位置（固定高度）
    
    mu_z = torch.sqrt(1 - best_mu_x ** 2 - best_mu_y ** 2)

    # 构造方向向量（注意坐标轴定义）
    dir_vec = (
        best_mu_x * BS_ex +
        best_mu_y * BS_ey +
        mu_z * BS_ez
    )

    dir_vec = dir_vec / torch.norm(dir_vec)

    t = (UE_high - BS_pos[2]) / dir_vec[2]
    UE_pos = BS_pos + t * dir_vec
    return UE_pos

def LoS_angle_est_byhand(y, C_sigma2, UE_pos_ini,sysPar,
                  max_iters=100, tol=1e-4,scalar_cov_flag=1):
    device = y.device
    
    lam = 1.0
    d = lam / 2
    Nx = sysPar['Nx']
    Ny = sysPar['Ny']
    BS_pos = torch.tensor(sysPar['BS_pos'], device=device, dtype=torch.float32)
    BS_ex = torch.tensor(sysPar['BS_ex'], device=device, dtype=torch.float32)
    BS_ey = torch.tensor(sysPar['BS_ey'], device=device, dtype=torch.float32)
    BS_ez = torch.cross(BS_ey, BS_ex, dim=-1)
    UE_high = sysPar['UE_posZ']
    y_vec = y.reshape(-1,1)
    y_vec_conjT = torch.conj(y_vec).T
    
    # Pre-compute constants
    pi_factor = 2 * math.pi / lam * d
    pi_factor_partial = 1j * math.pi
    x_indices = torch.arange(Nx, device=device, dtype=torch.float32)
    y_indices = torch.arange(Ny, device=device, dtype=torch.float32)
    NxNy = Nx * Ny

    if scalar_cov_flag == 1:
        inv_C_sigma2 = 1.0/C_sigma2
        # Pre-compute matA and matB for scalar case
        matA_prefactor = inv_C_sigma2**2
        matB_scalar = inv_C_sigma2
        y_outer = y_vec @ y_vec_conjT
    else:
        inv_C_sigma2 = torch.inverse(C_sigma2)
        # Pre-compute for matrix case
        inv_C_y = inv_C_sigma2 @ y_vec
        matA_base = inv_C_y @ y_vec_conjT @ inv_C_sigma2
        matB = inv_C_sigma2

    # 从初始位置估计方向余弦 mu_x, mu_y
    dir_vec = UE_pos_ini - BS_pos
    dir_vec = dir_vec / torch.norm(dir_vec)
    mu_x0 = torch.dot(dir_vec, BS_ex)
    mu_y0 = torch.dot(dir_vec, BS_ey)
    mu = torch.tensor([mu_x0.item(), mu_y0.item()], device=device, dtype=torch.float32)

    def compute_phases(mu_x, mu_y):
        """Compute phase vectors and their partials efficiently"""
        phase_x = torch.exp(1j * pi_factor * x_indices * mu_x)
        phase_y = torch.exp(1j * pi_factor * y_indices * mu_y)
        phase_x_partial = pi_factor_partial * x_indices * phase_x
        phase_y_partial = pi_factor_partial * y_indices * phase_y
        return phase_x, phase_y, phase_x_partial, phase_y_partial
    
    def grad_cal_optimized(mu):
        mu_x = mu[0]
        mu_y = mu[1]
        phase_x, phase_y, phase_x_partial, phase_y_partial = compute_phases(mu_x, mu_y)
        
        # Use outer product instead of kron for efficiency
        steer_matrix = torch.outer(phase_y, phase_x)
        steer_vec = steer_matrix.reshape(-1, 1)
        steer_vec_conjT = torch.conj(steer_vec).T

        if scalar_cov_flag == 1:
            matA = matA_prefactor * y_outer
            scalar_a = (steer_vec_conjT @ matA @ steer_vec).squeeze()
            scalar_b = NxNy * matB_scalar
            matC = 2*steer_vec_conjT@matA/scalar_b - 2*scalar_a/(scalar_b**2)*steer_vec_conjT*matB_scalar
        else:
            scalar_a = (steer_vec_conjT @ matA_base @ steer_vec).squeeze()
            scalar_b = (steer_vec_conjT @ matB @ steer_vec).squeeze()
            matC = 2*steer_vec_conjT@matA_base/scalar_b - 2*scalar_a/(scalar_b**2)*steer_vec_conjT@matB
        
        # Compute Jacobian using outer products
        steer_partialX = torch.outer(phase_y, phase_x_partial).reshape(-1, 1)
        steer_partialY = torch.outer(phase_y_partial, phase_x).reshape(-1, 1)
        jacobian = torch.cat((steer_partialX, steer_partialY), dim=1)
        
        return torch.real(matC @ jacobian)
    
    # gradient ascent with optimized parameters
    lr = 1e-2
    decay_rate = 0.5
    prev_mu = mu.clone()
    prev_obj = 0.0
    
    for iter_idx in range(max_iters):
        grad = grad_cal_optimized(mu).flatten()
        grad_norm = torch.norm(grad)
        
        if grad_norm > 1e-10:
            grad_normal = grad / grad_norm
        else:
            break
            
        with torch.no_grad():
            mu += lr * grad_normal

            # 投影到单位球内
            norm_mu = torch.norm(mu)
            if norm_mu > 1:
                mu *= 0.99 / norm_mu

            # 收敛判断 - 只在必要时计算目标函数
            diff = torch.norm(mu - prev_mu)
            if diff < tol:
                break
            prev_mu = mu.clone()

            # Reduce learning rate periodically
            if (iter_idx + 1) % 10 == 0:
                lr *= decay_rate

    # 得到 refined mu_x, mu_y
    mu_x = mu[0]
    mu_y = mu[1]
    mu_z = torch.sqrt(torch.clamp(1 - mu_x ** 2 - mu_y ** 2, min=0))
    dir_vec = mu_x * BS_ex + mu_y * BS_ey + mu_z * BS_ez
    dir_vec = dir_vec / torch.norm(dir_vec)
    t = (UE_high - BS_pos[2]) / dir_vec[2]
    UE_pos = BS_pos + t * dir_vec
    UE_var = 1e-5

    # Compute final steering vector
    phase_x = torch.exp(1j * pi_factor * x_indices * mu_x)
    phase_y = torch.exp(1j * pi_factor * y_indices * mu_y)
    steer_vec = torch.outer(phase_y, phase_x).reshape(-1, 1)
    steer_vec_conjT = torch.conj(steer_vec).T
    
    if scalar_cov_flag == 1:
        rho_est = steer_vec_conjT@y_vec/NxNy
    else:
        rho_est = torch.inverse(steer_vec_conjT@inv_C_sigma2@steer_vec)@steer_vec_conjT@inv_C_sigma2@y_vec
                
    return UE_pos, torch.tensor(UE_var, device=device, dtype=torch.complex64), rho_est

def LoS_angle_est_byhand_newton(y, C_sigma2, UE_pos_ini, sysPar,
                  max_iters=40, tol=1e-4, scalar_cov_flag=1, newton_threshold=0.01):
    """
    使用混合优化策略的角度估计函数：前期梯度下降 + 后期 Newton 加速
    
    Args:
        y: 观测信号
        C_sigma2: 协方差矩阵或标量
        UE_pos_ini: 初始位置估计
        sysPar: 系统参数
        max_iters: 最大迭代次数
        tol: 收敛容忍度
        scalar_cov_flag: 1表示标量协方差，0表示矩阵协方差
        newton_threshold: 切换到Newton法的梯度阈值
    
    Returns:
        UE_pos: 估计的UE位置
        UE_var: 位置方差
        rho_est: 估计的信道系数
    """
    device = y.device
    
    lam = 1.0
    d = lam / 2
    Nx = sysPar['Nx']
    Ny = sysPar['Ny']
    BS_pos = torch.tensor(sysPar['BS_pos'], device=device, dtype=torch.float32)
    BS_ex = torch.tensor(sysPar['BS_ex'], device=device, dtype=torch.float32)
    BS_ey = torch.tensor(sysPar['BS_ey'], device=device, dtype=torch.float32)
    BS_ez = torch.cross(BS_ey, BS_ex, dim=-1)
    UE_high = sysPar['UE_posZ']
    y_vec = y.reshape(-1,1)
    y_vec_conjT = torch.conj(y_vec).T
    
    # Pre-compute constants
    pi_factor = 2 * math.pi / lam * d
    pi_factor_partial = 1j * math.pi
    x_indices = torch.arange(Nx, device=device, dtype=torch.float32)
    y_indices = torch.arange(Ny, device=device, dtype=torch.float32)
    NxNy = Nx * Ny

    if scalar_cov_flag == 1:
        inv_C_sigma2 = 1.0/C_sigma2
        matA_prefactor = inv_C_sigma2**2
        matB_scalar = inv_C_sigma2
        y_outer = y_vec @ y_vec_conjT
    else:
        inv_C_sigma2 = torch.inverse(C_sigma2)
        inv_C_y = inv_C_sigma2 @ y_vec
        matA_base = inv_C_y @ y_vec_conjT @ inv_C_sigma2
        matB = inv_C_sigma2

    # 从初始位置估计方向余弦 mu_x, mu_y
    dir_vec = UE_pos_ini - BS_pos
    dir_vec = dir_vec / torch.norm(dir_vec)
    mu_x0 = torch.dot(dir_vec, BS_ex)
    mu_y0 = torch.dot(dir_vec, BS_ey)
    mu = torch.tensor([mu_x0.item(), mu_y0.item()], device=device, dtype=torch.float32)

    def compute_phases(mu_x, mu_y):
        """Compute phase vectors and their partials efficiently"""
        phase_x = torch.exp(1j * pi_factor * x_indices * mu_x)
        phase_y = torch.exp(1j * pi_factor * y_indices * mu_y)
        phase_x_partial = pi_factor_partial * x_indices * phase_x
        phase_y_partial = pi_factor_partial * y_indices * phase_y
        return phase_x, phase_y, phase_x_partial, phase_y_partial
    
    def grad_cal_optimized(mu):
        mu_x = mu[0]
        mu_y = mu[1]
        phase_x, phase_y, phase_x_partial, phase_y_partial = compute_phases(mu_x, mu_y)
        
        steer_matrix = torch.outer(phase_y, phase_x)
        steer_vec = steer_matrix.reshape(-1, 1)
        steer_vec_conjT = torch.conj(steer_vec).T

        if scalar_cov_flag == 1:
            matA = matA_prefactor * y_outer
            scalar_a = (steer_vec_conjT @ matA @ steer_vec).squeeze()
            scalar_b = NxNy * matB_scalar
            matC = 2*steer_vec_conjT@matA/scalar_b - 2*scalar_a/(scalar_b**2)*steer_vec_conjT*matB_scalar
        else:
            scalar_a = (steer_vec_conjT @ matA_base @ steer_vec).squeeze()
            scalar_b = (steer_vec_conjT @ matB @ steer_vec).squeeze()
            matC = 2*steer_vec_conjT@matA_base/scalar_b - 2*scalar_a/(scalar_b**2)*steer_vec_conjT@matB
        
        steer_partialX = torch.outer(phase_y, phase_x_partial).reshape(-1, 1)
        steer_partialY = torch.outer(phase_y_partial, phase_x).reshape(-1, 1)
        jacobian = torch.cat((steer_partialX, steer_partialY), dim=1)
        
        return torch.real(matC @ jacobian)
    
    def hessian_cal_optimized(mu):
        """计算 Hessian 矩阵用于 Newton 法"""
        mu_x = mu[0]
        mu_y = mu[1]
        phase_x = torch.exp(1j * pi_factor * x_indices * mu_x)
        phase_y = torch.exp(1j * pi_factor * y_indices * mu_y)
        phase_x_partial = pi_factor_partial * x_indices * phase_x
        phase_y_partial = pi_factor_partial * y_indices * phase_y
        phase_x_partial2 = -torch.pi**2 * x_indices**2 * phase_x
        phase_y_partial2 = -torch.pi**2 * y_indices**2 * phase_y
        
        steer_vec = torch.outer(phase_y, phase_x).reshape(-1, 1)
        steer_vec_conjT = torch.conj(steer_vec).T
        
        # 一阶偏导
        steer_partialX = torch.outer(phase_y, phase_x_partial).reshape(-1, 1)
        steer_partialY = torch.outer(phase_y_partial, phase_x).reshape(-1, 1)
        
        # 二阶偏导
        steer_partialXX = torch.outer(phase_y, phase_x_partial2).reshape(-1, 1)
        steer_partialYY = torch.outer(phase_y_partial2, phase_x).reshape(-1, 1)
        steer_partialXY = torch.outer(phase_y_partial, phase_x_partial).reshape(-1, 1)
        
        if scalar_cov_flag == 1:
            matA = matA_prefactor * y_outer
            scalar_a = (steer_vec_conjT @ matA @ steer_vec).squeeze()
            scalar_b = NxNy * matB_scalar
            
            H = torch.zeros(2, 2, device=device, dtype=torch.float32)
            
            # H_xx
            term1 = 2*torch.real(steer_partialX.conj().T @ matA @ steer_partialX / scalar_b)
            term2 = 2*torch.real(steer_vec_conjT @ matA @ steer_partialXX / scalar_b)
            term3 = -4*torch.real(steer_partialX.conj().T @ matA @ steer_vec) * torch.real(steer_vec_conjT @ matA @ steer_partialX) / (scalar_b**2)
            H[0, 0] = term1 + term2 + term3
            
            # H_yy
            term1 = 2*torch.real(steer_partialY.conj().T @ matA @ steer_partialY / scalar_b)
            term2 = 2*torch.real(steer_vec_conjT @ matA @ steer_partialYY / scalar_b)
            term3 = -4*torch.real(steer_partialY.conj().T @ matA @ steer_vec) * torch.real(steer_vec_conjT @ matA @ steer_partialY) / (scalar_b**2)
            H[1, 1] = term1 + term2 + term3
            
            # H_xy = H_yx
            term1 = 2*torch.real(steer_partialX.conj().T @ matA @ steer_partialY / scalar_b)
            term2 = 2*torch.real(steer_vec_conjT @ matA @ steer_partialXY / scalar_b)
            term3 = -4*torch.real(steer_partialX.conj().T @ matA @ steer_vec) * torch.real(steer_vec_conjT @ matA @ steer_partialY) / (scalar_b**2)
            H[0, 1] = H[1, 0] = term1 + term2 + term3
        else:
            scalar_a = (steer_vec_conjT @ matA_base @ steer_vec).squeeze()
            scalar_b = (steer_vec_conjT @ matB @ steer_vec).squeeze()
            
            H = torch.zeros(2, 2, device=device, dtype=torch.float32)
            
            # H_xx
            term1 = 2*torch.real(steer_partialX.conj().T @ matA_base @ steer_partialX / scalar_b)
            term2 = 2*torch.real(steer_vec_conjT @ matA_base @ steer_partialXX / scalar_b)
            term3 = -4*torch.real(steer_partialX.conj().T @ matA_base @ steer_vec)*torch.real(steer_vec_conjT @ matB @ steer_partialX)/(scalar_b**2)
            H[0, 0] = term1 + term2 + term3
            
            # H_yy
            term1 = 2*torch.real(steer_partialY.conj().T @ matA_base @ steer_partialY / scalar_b)
            term2 = 2*torch.real(steer_vec_conjT @ matA_base @ steer_partialYY / scalar_b)
            term3 = -4*torch.real(steer_partialY.conj().T @ matA_base @ steer_vec)*torch.real(steer_vec_conjT @ matB @ steer_partialY)/(scalar_b**2)
            H[1, 1] = term1 + term2 + term3
            
            # H_xy = H_yx
            term1 = 2*torch.real(steer_partialX.conj().T @ matA_base @ steer_partialY / scalar_b)
            term2 = 2*torch.real(steer_vec_conjT @ matA_base @ steer_partialXY / scalar_b)
            term3 = -4*torch.real(steer_partialX.conj().T @ matA_base @ steer_vec)*torch.real(steer_vec_conjT @ matB @ steer_partialY)/(scalar_b**2)
            H[0, 1] = H[1, 0] = term1 + term2 + term3
        
        return H
    
    # 混合优化策略：梯度下降 + Newton 法
    lr = 1e-2
    decay_rate = 0.5
    prev_mu = mu.clone()
    use_newton = False
    
    for iter_idx in range(max_iters):
        grad = grad_cal_optimized(mu).flatten()
        grad_norm = torch.norm(grad)
        
        # 提前终止
        if grad_norm < 1e-8:
            break
        
        # 切换到 Newton 法的条件：梯度较小且还没切换
        if not use_newton and grad_norm < newton_threshold and iter_idx >= 5:
            use_newton = True
        
        with torch.no_grad():
            if use_newton:
                # Newton 法：使用 Hessian 信息
                try:
                    H = hessian_cal_optimized(mu)
                    # 添加正则化确保 Hessian 正定（最大化问题取负）
                    H_reg = -H + 1e-6 * torch.eye(2, device=device)
                    delta = torch.linalg.solve(H_reg, grad)
                    
                    # 限制步长避免过冲
                    step_norm = torch.norm(delta)
                    if step_norm > 0.1:
                        delta = delta * 0.1 / step_norm
                    
                    mu_new = mu + delta
                except:
                    # Hessian 求解失败，回退到梯度法
                    grad_normal = grad / (grad_norm + 1e-10)
                    mu_new = mu + 0.01 * grad_normal
            else:
                # 梯度下降（归一化）
                grad_normal = grad / (grad_norm + 1e-10)
                mu_new = mu + lr * grad_normal
            
            # 投影到单位球内
            norm_mu = torch.norm(mu_new)
            if norm_mu > 1:
                mu_new *= 0.99 / norm_mu
            
            # 收敛判断
            diff = torch.norm(mu_new - mu)
            if diff < tol:
                mu = mu_new
                break
            
            mu = mu_new
            prev_mu = mu.clone()
            
            # 学习率衰减（仅梯度下降阶段）
            if not use_newton and (iter_idx + 1) % 10 == 0:
                lr *= decay_rate

    # 得到 refined mu_x, mu_y
    mu_x = mu[0]
    mu_y = mu[1]
    mu_z = torch.sqrt(torch.clamp(1 - mu_x ** 2 - mu_y ** 2, min=0))
    dir_vec = mu_x * BS_ex + mu_y * BS_ey + mu_z * BS_ez
    dir_vec = dir_vec / torch.norm(dir_vec)
    t = (UE_high - BS_pos[2]) / dir_vec[2]
    UE_pos = BS_pos + t * dir_vec
    UE_var = 1e-5

    # Compute final steering vector
    phase_x = torch.exp(1j * pi_factor * x_indices * mu_x)
    phase_y = torch.exp(1j * pi_factor * y_indices * mu_y)
    steer_vec = torch.outer(phase_y, phase_x).reshape(-1, 1)
    steer_vec_conjT = torch.conj(steer_vec).T
    
    if scalar_cov_flag == 1:
        rho_est = steer_vec_conjT@y_vec/NxNy
    else:
        rho_est = torch.inverse(steer_vec_conjT@inv_C_sigma2@steer_vec)@steer_vec_conjT@inv_C_sigma2@y_vec
                
    return UE_pos, torch.tensor(UE_var, device=device, dtype=torch.complex64), rho_est

def CRB_calculate(estPar,sysPar):
    gene_y = gene_y_func(sysPar)
    FIM_partial_mat = grad_numerical_vec(gene_y,estPar,1e-5)
    dig_FIM_inv = np.diag(np.linalg.inv(2*np.real(FIM_partial_mat.conj().T@FIM_partial_mat/sysPar['noise_pow'])))
    return np.sum(dig_FIM_inv[-2:])


def gene_y_func(sysPar):
    def gene_y(estPar):
        rho = estPar[0] * np.exp(1j*estPar[1])
        UE_pos = np.concatenate((estPar[2:4],np.array([sysPar['UE_posZ']])))
        BS_pos = sysPar['BS_pos']
        BS_ex = sysPar['BS_ex']
        BS_ey = sysPar['BS_ey']
        Nx = sysPar['Nx']
        Ny = sysPar['Ny']

        dir_vec = UE_pos - BS_pos
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        LoS_AoA_x = np.dot(dir_vec, BS_ex) # cosine of AoA BS in horizontal
        LoS_AoA_y = np.dot(dir_vec, BS_ey) # cosine of AoA BS in vertical
        steervec_x = np.exp(1j * np.pi * np.arange(Nx) * LoS_AoA_x)
        steervec_y = np.exp(1j * np.pi * np.arange(Ny) * LoS_AoA_y)
        H_LoS = rho * np.outer(steervec_y, steervec_x)
        y_Los = H_LoS
        return y_Los.reshape(-1,)
    return gene_y


def grad_numerical_vec(f, x0, dx, mode='fix'):
    x0 = np.asarray(x0)
    y = np.asarray(f(x0))
    
    output_len = y.size
    n = x0.size
    g = np.zeros((output_len, n), dtype=np.complex128)
    
    for i in range(n):
        x1 = x0.copy()
        if mode == 'fix':
            d = dx
        elif mode == 'prop':
            d = dx * abs(x1[i])
        if d == 0:
            d = 1e-9
        
        x1[i] += d
        yi = np.asarray(f(x1))
        g[:, i] = (yi - y) / d  # 偏导数近似
    return g


def H_LoS_from_pos(UE_pos,rho_post, sysPar, device='cpu'):
    Nx = sysPar['Nx']
    Ny = sysPar['Ny']
    BS_pos = torch.tensor(sysPar['BS_pos'], device=device, dtype=torch.float32)
    BS_ex = torch.tensor(sysPar['BS_ex'], device=device, dtype=torch.float32)
    BS_ey = torch.tensor(sysPar['BS_ey'], device=device, dtype=torch.float32)
    
    # 转为 numpy
    # 方向向量
    dir_vec = UE_pos - BS_pos
    dir_vec = dir_vec / torch.norm(dir_vec)

    # 方向余弦
    mu_x = torch.dot(dir_vec, BS_ex)
    mu_y = torch.dot(dir_vec, BS_ey)

    # 构造相位矩阵
    n_x = torch.arange(Nx, device=device)
    n_y = torch.arange(Ny, device=device)

    phase_x = torch.exp(1j * np.pi * n_x * mu_x)  # d = λ/2
    phase_y = torch.exp(1j * np.pi * n_y * mu_y)

    H_LoS_est = rho_post.item() * torch.outer(phase_y, phase_x).to(torch.complex64)
    return H_LoS_est, mu_x, mu_y

def m2valse_2domp_ini_clear(Y, mx, my, X, Mrx, T, nu):
    """
    MVALSE algorithm for 2D line spectral estimation
    Inputs:
        Y: (M, G) measurement matrix
        mx: indices in x-dim (length Mrx)
        my: indices in y-dim (length Mry)
        X: true signal (N,) for MSE calculation
        Mrx: number of measurements in x
        T: max iterations
        nu: initial noise variance
    Returns:
        dict with estimated frequencies, amplitudes, etc.
    """
    def grad(y, thx, thy, mx, my, nu):
        ax = np.exp(1j * mx * thx)
        ay = np.exp(1j * my * thy)
        a = np.kron(ax, ay)  # (M,)
        ax_p = 1j * mx * np.exp(1j * mx * thx)
        ay_p = 1j * my * np.exp(1j * my * thy)
        a_p1 = np.kron(ax_p, ay)
        a_p2 = np.kron(ax, ay_p)
        a_p = np.column_stack([a_p1, a_p2])  # (M, 2)

        M = len(a)
        grad_val = 2 * np.real(a.conj().T @ y @ y.conj().T @ a_p) / nu / M
        return grad_val
    
    def hessian(y, thx, thy, mx, my, nu):
        ax = np.exp(1j * mx * thx)
        ay = np.exp(1j * my * thy)
        a = np.kron(ax, ay)
        ax_p = 1j * mx * np.exp(1j * mx * thx)
        ax_pp = -mx**2 * np.exp(1j * mx * thx)
        ay_p = 1j * my * np.exp(1j * my * thy)
        ay_pp = -my**2 * np.exp(1j * my * thy)

        term1 = (np.kron(ax_p, ay)).conj().T @ y @ y.conj().T @ (np.kron(ax_p, ay))
        term2 = a.conj().T @ y @ y.conj().T @ np.kron(ax_pp, ay)
        Pxx = 2 * np.real(term1 + term2) / nu / len(a)

        term1 = (np.kron(ax, ay_p)).conj().T @ y @ y.conj().T @ (np.kron(ax, ay_p))
        term2 = a.conj().T @ y @ y.conj().T @ np.kron(ax, ay_pp)
        Pyy = 2 * np.real(term1 + term2) / nu / len(a)

        term1 = (np.kron(ax_p, ay)).conj().T @ y @ y.conj().T @ (np.kron(ax, ay_p))
        term2 = a.conj().T @ y @ y.conj().T @ np.kron(ax_p, ay_p)
        Pxy = 2 * np.real(term1 + term2) / nu / len(a)

        return np.array([[Pxx, Pxy], [Pxy, Pyy]])
    
    def obj_func(y, thx, thy, mx, my, nu):
        ax = np.exp(1j * mx * thx)
        ay = np.exp(1j * my * thy)
        a = np.kron(ax, ay)
        M = len(a)
        return np.real(a.conj().T @ y @ y.conj().T @ a) / nu / M

    M, G = Y.shape
    Mry = M // Mrx
    Nx = mx[-1] + 1
    Ny = my[-1] + 1
    N = Nx * Ny
    L = min(Nx, Ny)  # assumed number of components

    # Reshape Y to (Mry, Mrx, G)
    Y_d = Y.reshape(Mry, Mrx, G)
    Yx = np.zeros((Mrx, Mry * G), dtype=complex)
    for i in range(G):
        Yx[:, i*Mry:(i+1)*Mry] = Y_d[:, :, i].T

    Y2 = Y.conj().T @ Y
    A = np.zeros((N, L), dtype=complex)
    A_x = np.zeros((Nx, L), dtype=complex)
    A_y = np.zeros((Ny, L), dtype=complex)
    W = np.zeros((L, G), dtype=complex)
    mse = np.zeros(T)
    Kt = np.zeros(T, dtype=int)
    t = 0

    res = Y.copy()
    Xangle_grid_len = 4 * Nx
    Yangle_grid_len = 4 * Ny
    Xth_grid = np.linspace(-np.pi, np.pi - 2*np.pi/Xangle_grid_len, Xangle_grid_len)
    Yth_grid = np.linspace(-np.pi, np.pi - 2*np.pi/Yangle_grid_len, Yangle_grid_len)

    Dic_xAngle = np.exp(1j * np.outer(mx, Xth_grid))         # (Mrx, Xangle)
    Dic_yAngle = np.exp(1j * np.outer(my, Yth_grid))         # (Mry, Yangle)
    Dic = kron(Dic_xAngle, Dic_yAngle)                       # (M, Xangle*Yangle)

    used_idx = []

    for l in range(L):
        if l == 0:
            K = L // 2
            rho = K / L

        power = np.abs(Dic.conj().T @ res.reshape(M, -1)).sum(axis=1)  # (X*Y,)
        power[used_idx] = -1
        idx_a = np.argmax(power)
        used_idx.append(idx_a)

        idx_xAngle, idx_yAngle = np.unravel_index(idx_a, (Xangle_grid_len, Yangle_grid_len))
        th = np.array([[Xth_grid[idx_xAngle]], [Yth_grid[idx_yAngle]]])

        # Gradient descent refinement
        out_val = [0]
        for gdm_iter in range(1, 100):
            obj_prev = out_val[-1]
            obj_curr = obj_func(res, th[0,0], th[1,0], mx, my, nu)
            grad_val = grad(res, th[0,0], th[1,0], mx, my, nu)
            norm_grad = np.linalg.norm(grad_val)
            if norm_grad == 0:
                break
            th_grad_normal = grad_val / norm_grad
            th_new = th + 0.01 * th_grad_normal.reshape(2,1)

            if obj_curr < obj_prev:
                break
            th = th_new
            out_val.append(obj_curr)

        # Newton step
        hess = hessian(res, th[0,0], th[1,0], mx, my, nu)
        try:
            delta_th = np.linalg.solve(hess, grad_val)
        except:
            delta_th = np.zeros(2)
        th -= delta_th

        # Compute concentration parameters kappa
        kappax = Ainv(np.exp(0.5 / hess[0,0]))
        kappay = Ainv(np.exp(0.5 / hess[1,1]))

        A_x[:, l] = np.exp(1j * mx * th[0,0]) * (besseli(mx, kappax, 1) / besseli(0, kappax, 1))
        A_y[:, l] = np.exp(1j * my * th[1,0]) * (besseli(my, kappay, 1) / besseli(0, kappay, 1))

        A[:, l] = np.kron(A_x[:, l], A_y[:, l])
        #? 切片错误
        W[:l+1, :] = np.linalg.solve(A[:, :l+1].conj().T @ A[:, :l+1], A[:, :l+1].conj().T @ Y)

        res = Y - A[:, :l+1] @ W[:l+1, :]
        if l == K - 1:
            xro = A[:, :l+1] @ W[:l+1, :]
            mse[t] = np.linalg.norm(X - xro.ravel())**2 / np.linalg.norm(X)**2
            Kt[t] = K

    # Initialize J, H
    tau = 1e9
    J = A.conj().T @ A
    np.fill_diagonal(J, M)
    H = A.conj().T @ Y

    cont = True
    while cont:
        t += 1
        if t >= T:
            break

        K, s, W, C = maxZ_t1(J, H, M, nu, rho, tau)
 
        if K > 0:
            s_idx = np.where(s)[0]
            W_s = W[s_idx, :]
            J_ss = J[np.ix_(s_idx, s_idx)]
            C_ss = C[np.ix_(s_idx, s_idx)]
            H_s = H[s_idx, :]

            nu = (np.trace(Y2 - 2*np.real(H_s.conj().T @ W_s)) +
                  np.trace(W_s.conj().T @ J_ss @ W_s) / G +
                  np.trace(J_ss @ C_ss)) / M

            tau = (np.trace(W_s @ W_s.conj().T) + G * np.trace(C_ss)) / (K * G)
            rho = K / L if K < L else (L - 1) / L
        else:
            rho = 1 / L

        inz = np.where(s)[0]
        th = np.zeros((K, 2))
        ka = np.zeros((2, K))

        for i in range(K):
            if K == 1:
                R = Y
                eta = (2 / nu) * (R @ W[inz[i], :].conj())
                eta_2d = np.reshape(eta, (Mrx, Mry), order='F')
                eta_x = (eta_2d @ A_y[:, inz[i]]).conj().T
                eta_y = (eta_2d.T @ A_x[:, inz[i]]).conj().T
            else:
                idx_other = np.ones(L, dtype=bool)
                idx_other[inz[i]] = False
                A_i = A[:, idx_other]
                R = Y - A_i @ W[idx_other, :]
                C_i = C[idx_other, inz[i]]
                eta = (2 / nu) * (R @ W[inz[i], :].conj() - A_i @ C_i)
                eta_2d = np.reshape(eta, (Mrx, Mry), order='F')
                eta_y = (eta_2d @ A_x[:, inz[i]]).conj().T
                eta_x = (eta_2d.T @ A_y[:, inz[i]]).conj().T

            A_x_col, theta_x, kappa_x = Heuristic2(eta_x, mx)
            A_y_col, theta_y, kappa_y = Heuristic2(eta_y, my)

            A_x[:, inz[i]] = A_x_col
            A_y[:, inz[i]] = A_y_col
            th[i, 0] = theta_x
            th[i, 1] = theta_y
            ka[0, i] = kappa_x
            ka[1, i] = kappa_y

        A = np.array([np.kron(A_x[:, k], A_y[:, k]) for k in range(L)]).T

        J[:, s] = A.conj().T @ A[:, s]
        J[s, :] = J[:, s].conj().T
        s_idx = np.where(s)[0]
        J[np.ix_(s_idx, s_idx)] = J[np.ix_(s_idx, s_idx)] - np.diag(np.diag(J[np.ix_(s_idx, s_idx)])) + M * np.eye(K)
        H[s, :] = A[:, s].conj().T @ Y

        xr = A[:, s] @ W[s, :]
        mse[t] = np.linalg.norm(xr - X)**2 / np.linalg.norm(X)**2
        Kt[t] = K

        if t > 0 and (np.linalg.norm(xr - xro) < 1e-6 * np.linalg.norm(xro) or t >= T - 1):
            cont = False
            mse[t+1:] = mse[t]
            Kt[t+1:] = Kt[t]
        xro = xr.copy()

    return {
        'freqs': th,
        'kaps': ka,
        'amps': W[s, :],
        'x_estimate': xr,
        'noise_var': nu,
        'tau': tau,
        'iterations': t,
        'mse': mse,
        'K': Kt
    }

def Heuristic2(eta, m):
    N = len(m)
    ka = np.abs(eta)
    A = besseli(1, ka, 1) / (besseli(0, ka, 1) + 1e-10)
    kmix = Ainv(A**(1.0 / (m + 1e-10)**2))

    k = N - 1
    eta_q = kmix[k] * np.exp(1j * (np.angle(eta[k]) + 2 * np.pi * np.arange(1, m[k]+1)) / m[k])
    for k in range(N-2, -1, -1):
        if m[k] != 0:
            phi = np.angle(eta[k])
            wrapped = phi + 2 * np.pi * np.round((m[k] * np.angle(eta_q) - phi) / (2 * np.pi))
            eta_q += kmix[k] * np.exp(1j * wrapped / m[k])

    in_max = np.argmax(np.abs(eta_q))
    mu = np.angle(eta_q[in_max])

    n = np.arange(0, m[-1]+1)
    d1 = -np.imag(eta.conj().T @ (m * np.exp(1j * m * mu)))
    d2 = -np.real(eta.conj().T @ (m**2 * np.exp(1j * m * mu)))

    if d2 < 0:
        theta = mu - d1 / d2
        kappa = Ainv(np.exp(0.5 / d2))
    else:
        theta = mu
        kappa = np.abs(eta_q[in_max])

    a = np.exp(1j * n * theta) * (besseli(n, kappa, 1) / (besseli(0, kappa, 1) + 1e-10))
    return a, theta, kappa

def maxZ_t1(J, h, M, nu, rho, tau):
    L = h.shape[0]
    cnst = np.log(rho / (1 - rho) / tau)
    K = 0
    s = np.zeros(L, dtype=bool)
    w = np.zeros((L, h.shape[1]), dtype=complex)
    C = np.zeros((L, L), dtype=complex)
    Delta = np.zeros(L)

    cont = True
    while cont and K < L:
        if K < M - 1:
            not_s = ~s
            if not np.any(not_s):
                break
            J_s_ns = J[s][:, not_s] if np.any(s) else np.zeros((0, np.sum(not_s)))
            C_ss = C[s][:, s] if np.any(s) else np.zeros((0,0))
            temp_sum = 0
            if np.any(s):
                temp_sum = np.sum(J_s_ns.conj() * (C_ss @ J_s_ns), axis=0).real
            v = nu / (M + nu/tau - temp_sum)
            h_term = h[not_s] - (J[s, :][:, not_s].conj().T @ w[s]) if np.any(s) else h[not_s]
            u = v[:, None] * h_term / nu
            Delta[not_s] = np.log(v) + np.sum(u * u.conj(), axis=1).real / v + cnst
        else:
            Delta[~s] = -1

        if np.any(s):
            diag_C = np.diag(C[s, s])
            w_s = w[s]
            Delta[s] = -np.log(diag_C) - (w_s * w_s.conj()).sum(axis=1).real / diag_C - cnst

        k = np.argmax(Delta)
        if Delta[k] <= 0:
            break

        if not s[k]:
            # Activate
            s[k] = True
            w[k] = u[k - (~s[:k]).sum()]  # careful indexing
            if K > 0:
                ctemp = C[s[:-1], s[:-1]] @ J[s[:-1], k] / nu
                w[s[:-1]] -= ctemp * w[k]
                C[s[:-1], s[:-1]] += np.outer(ctemp, ctemp.conj()) * v[k]
                C[s[:-1], k] = -v[k] * ctemp
                C[k, s[:-1]] = C[s[:-1], k].conj()
            C[k, k] = v[k]
            K += 1
        else:
            # Deactivate
            s[k] = False
            if K > 1:
                ck = C[s, k]
                w[s] -= (ck * w[k]) / C[k, k]
                C[s, s] -= np.outer(ck, C[k, s]) / C[k, k]
            K -= 1
        C = (C + C.conj().T) / 2
    return K, s, w, C

def Ainv(R):
    """
    近似求解 R = I1(k)/I0(k) 的 k（Mardia & Jupp, 2000, pp.85–86）
    支持标量/向量 R∈[0,1)
    """
    R = np.asarray(R, dtype=float)
    k = np.array(R, dtype=float)  # 初始化为相同形状

    in1 = R < 0.53
    in3 = R >= 0.85
    in2 = ~(in1 | in3)

    if np.any(in1):
        R1 = R[in1]
        t = R1 * R1
        k[in1] = R1 * (2 + t + (5.0/6.0) * t * t)

    if np.any(in2):
        R2 = R[in2]
        k[in2] = -0.4 + 1.39 * R2 + 0.43 / (1 - R2)

    if np.any(in3):
        R3 = R[in3]
        k[in3] = 1.0 / (R3 * (R3 - 1.0) * (R3 - 3.0))

    return k

