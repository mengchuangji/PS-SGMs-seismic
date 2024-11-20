import torch
import numpy as np
import tqdm
import torch.nn.functional as F

def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

def mat_by_vec(M, v):
    vshape = v.shape[2]
    if len(v.shape) > 3: vshape = vshape * v.shape[3]
    return torch.matmul(M, v.view(v.shape[0] * v.shape[1], vshape,
                    1)).view(v.shape[0], v.shape[1], M.shape[0])

def vec_to_image(v, img_dim):
    return v.view(v.shape[0], v.shape[1], img_dim, img_dim)

def vec_to_image_(v, img_shape):
    return v.view(v.shape[0], v.shape[1], img_shape[0], img_shape[1])

def invert_diag(M):
    M_inv = torch.zeros_like(M)
    M_inv[M != 0] = 1 / M[M != 0]
    return M_inv


@torch.no_grad()
def general_anneal_Langevin_dynamics(H, y_0, x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, c_begin = 0, sigma_0 = 1):
    U, singulars, V = torch.svd(H, some=False)# H:([11264, 16384])  y_0:[8, 1, 11264] x_mod:(8, 1, 128, 128)
    V_t = V.transpose(0, 1) # ([16384, 16384])

    # ZERO = 1e-3
    ZERO = 1e-3
    singulars[singulars < ZERO] = 0 #([11264])

    Sigma = torch.zeros_like(H)
    for i in range(singulars.shape[0]): Sigma[i, i] = singulars[i]
    S_1, S_n = singulars[0], singulars[-1]

    S_S_t = torch.zeros_like(U) #[11264, 11264]
    for i in range(singulars.shape[0]): S_S_t[i, i] = singulars[i] ** 2

    num_missing = V.shape[0] - torch.count_nonzero(singulars) #v: [16384, 16384]

    s0_2_I = ((sigma_0 ** 2) * torch.eye(U.shape[0])).to(x_mod.device) #[11264, 11264]

    V_t_x = mat_by_vec(V_t, x_mod) #[16384, 16384]* [8, 1, 128, 128]=[8, 1,16384]
    U_t_y = mat_by_vec(U.transpose(0,1), y_0)  #[11264, 11264]*[8, 1, 11264]= [8, 1,11264]

    img_dim = x_mod.shape[2]#128

    images = []
    x_mode_list = []

    with torch.no_grad():
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='general annealed Langevin sampling'):

            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * (c + c_begin)#[8]
            labels = labels.long()
            step_size = step_lr * ((1 / sigmas[-1]) ** 2)

            falses = torch.zeros(V_t_x.shape[2] - singulars.shape[0], dtype=torch.bool, device=x_mod.device)#16384-11264
            cond_before_lite = singulars * sigma > sigma_0
            cond_after_lite = singulars * sigma < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            step_vector = torch.zeros_like(V_t_x)
            step_vector[:, :, :] = step_size * (sigma**2)
            step_vector[:, :, cond_before] = step_size * ((sigma**2) - (sigma_0 / singulars[cond_before_lite])**2)
            step_vector[:, :, cond_after] = step_size * (sigma**2) * (1 - (singulars[cond_after_lite] * sigma / sigma_0)**2)

            for s in range(n_steps_each):
                grad = torch.zeros_like(V_t_x)
                score = mat_by_vec(V_t, scorenet(x_mod, labels))#[8, 1, 16384]=[16384, 16384]*[8, 1, 128, 128]

                diag_mat = S_S_t * (sigma ** 2) - s0_2_I
                diag_mat[cond_after_lite, cond_after_lite] = diag_mat[cond_after_lite, cond_after_lite] * (-1)

                first_vec = U_t_y - mat_by_vec(Sigma, V_t_x)
                cond_grad = mat_by_vec(invert_diag(diag_mat), first_vec)
                cond_grad = mat_by_vec(Sigma.transpose(0,1), cond_grad)
                grad = torch.zeros_like(cond_grad)
                grad[:, :, cond_before] = cond_grad[:, :, cond_before]
                grad[:, :, cond_after] = cond_grad[:, :, cond_after] + score[:, :, cond_after]
                grad[:, :, -num_missing:] = score[:, :, -num_missing:]

                noise = torch.randn_like(V_t_x)
                V_t_x = V_t_x + step_vector * grad + noise * torch.sqrt(step_vector * 2)
                x_mod = vec_to_image(mat_by_vec(V, V_t_x), img_dim)

                if not final_only:
                    images.append(x_mod.to('cpu'))
            # if c%20==0:
            if c in range(0,501,50):
                x_mode_list.append(x_mod)
            # if c in range(0,451,45):
            #     x_mode_list.append(x_mod)

            # if c in [0,250,300,350,400,425,450,475,499]:
            #     x_mode_list.append(x_mod)

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))
            x_mode_list.append(x_mod.to('cpu'))
        if final_only:
            return [x_mod.to('cpu')], x_mode_list
        else:
            return images


@torch.no_grad()
def general_anneal_Langevin_dynamics_den(y_0, x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, c_begin = 0, sigma_0 = 1):
    # 找到最接近给定sigma_noise的元素的索引
    index = np.abs(sigmas - sigma_0).argmin()
    print('index of sigma:',index,'sigma_index=', sigmas[index])
    # print("最接近的sigma_noise值的索引:", index)

    img_dim = x_mod.shape[2] #128 x_mod:(1,1,128,128)
    img_shape = (x_mod.shape[2], x_mod.shape[3])
    images = []
    x_mode_list=[]
    x_mod=y_0.view(x_mod.shape[0],x_mod.shape[1],x_mod.shape[2],x_mod.shape[3])
    with torch.no_grad(): #[-127:-1] 226 233
        for c, sigma in tqdm.tqdm(enumerate(sigmas[-(len(sigmas)-index):]), total=len(sigmas[-(len(sigmas)-index):]), desc='general annealed Langevin sampling'):
            # c=c+500-194
            c=c+index
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * (c + c_begin)
            labels = labels.long()
            step_size = step_lr * ((1 / sigmas[-1]) ** 2)
            step_vector= step_size * (sigma**2)
            for s in range(n_steps_each):
                score = scorenet(x_mod, labels).view(y_0.shape[0], y_0.shape[1], y_0.shape[2])
                x_mod=x_mod.view(y_0.shape[0], y_0.shape[1], y_0.shape[2])#y_0 x_mod[1, 1, 16384]
                grad= score+ (y_0-x_mod)/((sigma_0)**2-sigma**2)
                noise = torch.randn_like(x_mod)
                step_vector = torch.as_tensor(step_vector, dtype=torch.float64)
                x_mod= x_mod + step_vector * grad + noise * torch.sqrt(step_vector * 2)
                # x_mod = vec_to_image(x_mod, img_dim)
                x_mod = vec_to_image_(x_mod, img_shape)

                if not final_only:
                    images.append(x_mod.to('cpu'))
            # if c%20==0:
            if c== 499: print(499)
            if c in range(index,len(sigmas),(len(sigmas)-index)//4):
                x_mode_list.append(x_mod)
                print('noise index in plot:', c, 'sigmas[index]:', sigmas[c])
        x_mode_list[0] = y_0.view(x_mod.shape[0], x_mod.shape[1], x_mod.shape[2], x_mod.shape[3])

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))
            x_mode_list.append(x_mod)
            # x_mode_list[-1]=x_mod

        if final_only:
            return [x_mod.to('cpu')], x_mode_list
        else:
            return images

@torch.no_grad()
def general_anneal_Langevin_dynamics_inp(M, y_0, x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                                         final_only=False, verbose=False, denoise=True, c_begin=0, sigma_0=1):
    # 找到最接近给定sigma_noise的元素的索引
    index = np.abs(sigmas - sigma_0).argmin()
    print('index of sigma:', index, 'sigma_index=', sigmas[index])

    img_dim = x_mod.shape[2]  # 128 If the data is square
    img_shape = (x_mod.shape[2], x_mod.shape[3])
    images = []
    # x_mod= torch.rand_like(y_0)
    x_mode_list = []
    x_mod = torch.rand_like(x_mod) #(x_mod.shape[0], x_mod.shape[1], x_mod.shape[2], x_mod.shape[3])
    with torch.no_grad():  # [-127:-1] 226 233
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas),
                                  desc='general annealed Langevin sampling'):
            # c = c + 500 - 233
            if sigma > sigma_0:
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * (c + c_begin)
                labels = labels.long()
                step_size = step_lr * ((1 / sigmas[-1]) ** 2)
                step_vector = step_size * (sigma ** 2)
                for s in range(n_steps_each):
                    score = scorenet(x_mod, labels).view(y_0.shape[0], y_0.shape[1], y_0.shape[2])
                    x_mod = x_mod.view(y_0.shape[0], y_0.shape[1], y_0.shape[2])  # y_0 x_mod[1, 1, 16384]
                    M=M.view(y_0.shape[1], y_0.shape[2])
                    grad=torch.zeros_like(score)
                    grad[:,M == 0] = score[:,M == 0]
                    grad[:,M == 1] = (y_0 - x_mod)[:,M == 1] / ( sigma ** 2-(sigma_0) ** 2)

                    noise = torch.randn_like(x_mod)
                    step_vector = torch.as_tensor(step_vector, dtype=torch.float64)
                    x_mod = x_mod + step_vector * grad + noise * torch.sqrt(step_vector * 2)
                    x_mod = vec_to_image_(x_mod, img_shape)

                    if not final_only:
                        images.append(x_mod.to('cpu'))
                # if c%20==0:
                if c in range(0, 501, 50):
                    x_mode_list.append(x_mod)
                    print('noise index in plot:', c, 'sigmas[index]:', sigmas[c])
                # if c in [0, 249, 299, 349, 399, 424, 449, 474, 499]:
                #     x_mode_list.append(x_mod)
            elif sigma < sigma_0:
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * (c + c_begin)
                labels = labels.long()
                step_size = step_lr * ((1 / sigmas[-1]) ** 2)
                step_vector = step_size * (sigma ** 2)
                for s in range(n_steps_each):
                    score = scorenet(x_mod, labels).view(y_0.shape[0], y_0.shape[1], y_0.shape[2])
                    x_mod = x_mod.view(y_0.shape[0], y_0.shape[1], y_0.shape[2])  # y_0 x_mod[1, 1, 16384]
                    M=M.view(y_0.shape[1], y_0.shape[2])
                    grad = torch.zeros_like(x_mod)
                    # grad[:,M == 1] = score[:,M == 1]
                    # grad[:,M == 0] = (y_0 - x_mod)[:,M == 0] / ((sigma_0) ** 2-sigma ** 2)

                    grad[:, M == 1] = score[:, M == 1] + (y_0 - x_mod)[:, M == 1] / ((sigma_0) ** 2 - sigma ** 2)
                    grad[:, M == 0] = score[:, M == 0]

                    noise = torch.randn_like(x_mod)
                    step_vector = torch.as_tensor(step_vector, dtype=torch.float64)
                    x_mod = x_mod + step_vector * grad + noise * torch.sqrt(step_vector * 2)
                    x_mod = vec_to_image_(x_mod, img_shape)

                    if not final_only:
                        images.append(x_mod.to('cpu'))
                # if c%20==0:
                if c in range(0, 501,50):
                    x_mode_list.append(x_mod)
                    print('noise index in plot:', c, 'sigmas[index]:', sigmas[c])
                # if c in [0, 249, 299, 349, 399, 424, 449, 474, 499]:
                #     x_mode_list.append(x_mod)

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))
            x_mode_list.append(x_mod.to('cpu'))
            # x_mode_list[-1]=x_mod.to('cpu')

        if final_only:
            return [x_mod.to('cpu')], x_mode_list
        else:
            return images




#from ncsnv2 mcj
@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True):
    images = []
    x_mode_list=[]
    with torch.no_grad():
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas),
                                  desc='annealed Langevin sampling'):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

            # if c%20==0:
            if c in range(0, 500, 50):
            # if c in [0, 250, 300, 350, 400, 425, 450, 475, 499]:
                x_mode_list.append(x_mod)

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))
            x_mode_list.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')], x_mode_list
        else:
            return images

@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size,
                                        n_steps_each=100, step_lr=0.000008):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """

    images = []

    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().view(-1, 3, image_size, image_size)
    x_mod = x_mod.view(-1, 3, image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for s in range(n_steps_each):
                images.append(x_mod.to('cpu'))
                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :cols] = corrupted_half_image
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                                                                         grad.abs().max()))

        return images

@torch.no_grad()
def anneal_Langevin_dynamics_interpolation(x_mod, scorenet, sigmas, n_interpolations, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False,denoise=True):
    images = []

    n_rows = x_mod.shape[0]

    x_mod = x_mod[:, None, ...].repeat(1, n_interpolations, 1, 1, 1)
    x_mod = x_mod.reshape(-1, *x_mod.shape[2:])

    # for c, sigma in enumerate(sigmas):
    for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas),
                                  desc='annealed Langevin sampling for interpolation'):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)

            noise_p = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            noise_q = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            angles = torch.linspace(0, np.pi / 2., n_interpolations, device=x_mod.device)

            noise = noise_p[:, None, ...] * torch.cos(angles)[None, :, None, None, None] + \
                        noise_q[:, None, ...] * torch.sin(angles)[None, :, None, None, None]

            noise = noise.reshape(-1, *noise.shape[2:])
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()

            x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm

            # if denoise:
            #     last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            #     last_noise = last_noise.long()
            #     x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            #     images.append(x_mod.to('cpu'))

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print(
                    "level: {}, step_size: {}, image_norm: {}, grad_norm: {}, snr: {}".format(
                        c, step_size, image_norm.item(), grad_norm.item(), snr.item()))


    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images