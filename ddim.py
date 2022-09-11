"""
Use the deterministic generative process proposed by Song et al.[1]
[1] Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising diffusion implicit models." arXiv preprint arXiv:2010.02502 (2020).
"""
import torch
import ddpm_torch
import numpy as np


# def get_selection_schedule(schedule, size, timesteps):
#     """
#     :param schedule: selection schedule
#     :param size: length of subsequence
#     :param timesteps: total timesteps of pretrained ddpm model
#     :return: a mapping from subsequence index to original one
#     """
#     assert schedule in {"linear", "quadratic"}
#     power = 1 if schedule == "linear" else 2
#     c = timesteps / size ** power
#
#     def subsequence(t: np.ndarray):
#         return np.floor(c * np.power(t + 1, power) - 1).astype(np.int64)
#     return subsequence


def get_selection_schedule(schedule, size, timesteps):
    """
    :param schedule: selection schedule
    :param size: length of subsequence
    :param timesteps: total timesteps of pretrained ddpm model
    :return: subsequence
    """
    assert schedule in {"linear", "quadratic"}

    if schedule == "linear":
        subsequence = np.arange(0, timesteps, timesteps // size)
    else:
        subsequence = np.power(np.linspace(0, np.sqrt(timesteps * 0.8), size), 2).astype(np.int32)

    return subsequence


class DDIM(ddpm_torch.GaussianDiffusion):
    def __init__(self, betas, model_mean_type, model_var_type, loss_type, eta, subsequence):
        super().__init__(betas, model_mean_type, model_var_type, loss_type)
        self.eta = eta  # coefficient between [0, 1] that decides the behavior of generative process
        self.subsequence = subsequence  # subsequence of the accelerated generation

        eta2 = eta ** 2
        assert not (eta2 != 1. and model_var_type != "fixed-small"),\
            'Cannot use DDIM (eta < 1) with var type other than "fixed-small"'

        self.alphas_bar = self.alphas_bar[subsequence]
        self.alphas_bar_prev = np.concatenate([np.ones(1, dtype=np.float64), self.alphas_bar[:-1]])
        self.alphas = self.alphas_bar / self.alphas_bar_prev
        self.betas = 1 - self.alphas
        self.sqrt_alphas_bar_prev = np.sqrt(self.alphas_bar_prev)
        self.posterior_var = self.betas * (1 - self.alphas_bar_prev) / (1 - self.alphas_bar) * eta2
        self.posterior_logvar_clipped = np.log(np.concatenate([
            np.array([self.posterior_var[1], ], dtype=np.float64), self.posterior_var[1:]]))

        # coefficients to recover x_0 from x_t and \epsilon_t
        self.sqrt_recip_alphas_bar = np.sqrt(1. / self.alphas_bar)
        self.sqrt_recip_m1_alphas_bar = np.sqrt(1. / self.alphas_bar - 1.)

        # coefficients to calculate E[x_{t-1}|x_0, x_t]
        self.posterior_mean_coef2 = np.sqrt(
            1 - self.alphas_bar - eta2 * self.betas
        ) * np.sqrt(1 - self.alphas_bar_prev) / (1 - self.alphas_bar)
        self.posterior_mean_coef1 = self.sqrt_alphas_bar_prev * (1 - np.sqrt(self.alphas) * self.posterior_mean_coef2)

        self.subsequence = torch.as_tensor(subsequence)

    def p_sample(self, denoise_fn, shape, device=torch.device("cpu"), noise=None):
        S = len(self.subsequence)
        B, *_ = shape
        subsequence = self.subsequence.to(device)
        _denoise_fn = lambda x, t: denoise_fn(x, subsequence[t])
        t = torch.empty((B, ), dtype=torch.int64, device=device)
        if noise is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = noise.to(device)
        for ti in range(S - 1, -1, -1):
            t.fill_(ti)
            x_t = self.p_sample_step(_denoise_fn, x_t, t)
        return x_t


if __name__ == "__main__":
    print(get_selection_schedule("linear", 10, 1000)(np.arange(10)))