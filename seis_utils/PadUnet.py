class PadUnet:
    '''
    im: N x C x H x W torch tensor
    dep_U: depth of UNet
    '''
    def __init__(self, im, dep_U, mode='reflect'):
        self.im_old = im
        self.dep_U = dep_U
        self.mode = mode
        self.H_old = im.shape[2]
        self.W_old = im.shape[3]

    def pad(self):
        # lenU = 2 ** self.dep_U
        # padH = 0 if ((self.H_old % lenU) == 0) else ((self.H_old//lenU+1)* lenU-self.H_old)
        # padW = 0 if ((self.W_old % lenU) == 0) else ((self.W_old//lenU+1)* lenU-self.W_old)
        # padding = (0, padW, 0, padH)
        # import torch.nn.functional as F
        # out = F.pad(self.im_old, pad=padding, mode=self.mode)
        # return out

        lenU = 2 ** (self.dep_U-1)
        padH = 0 if ((self.H_old % lenU) == 0) else (lenU - (self.H_old % lenU))
        padW = 0 if ((self.W_old % lenU) == 0) else (lenU - (self.W_old % lenU))
        padding = (0, padW, 0, padH)
        import torch.nn.functional as F
        out = F.pad(self.im_old, pad=padding, mode=self.mode)
        return out

    def pad_inverse(self, im_new):
        return im_new[:, :, :self.H_old, :self.W_old]
