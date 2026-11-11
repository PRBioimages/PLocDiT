# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


import math

import numpy as np
import torch as th
import enum

from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    Original ported from this codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type
    ):

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        t = t.view(-1) 
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * img.shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        t = t.view(-1) 
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        # Expand model_kwargs to doubled batch for CFG. Only do this for a real CFG
        # call; otherwise an ordinary even batch (e.g. 2 samples) would silently have
        # its second half treated as the unconditional branch.
        _cfg_active = model_kwargs is not None and (
            float(model_kwargs.get("cfg_scale", 1.0) or 1.0) > 1.0
            or getattr(getattr(model, "module", model), "cfg_scale", None) is not None
        )
        if _cfg_active and img.shape[0] % 2 == 0 and img.shape[0] >= 2:
            doubled_kwargs = {}
            for k, v in model_kwargs.items():
                if isinstance(v, th.Tensor) and v.shape[0] == img.shape[0] // 2:
                    doubled_kwargs[k] = th.cat([v, th.zeros_like(v)], dim=0)
                elif k == 'cfg_scale':
                    doubled_kwargs[k] = v
                else:
                    doubled_kwargs[k] = v
            model_kwargs = doubled_kwargs

        if progress:
            # Lazy import so we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * img.shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, t, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)


#################################################################################
#                         Flow Matching Diffusion                               #
#################################################################################

class FlowMatching:
    """
    Flow Matching diffusion for velocity prediction (v = ε − x₀).

    Forward process:  x_t = (1 − t) * x₀ + t * ε     (linear interpolation)
    Model predicts:   v_θ(x_t, t)
    Target:           v = ε − x₀

    Training:
        t ~ LogitNormal(0, 1)  →  sigmoid(randn())
        loss = MSE(v_θ(x_t, t*1000), v)

    Sampling:
        Euler or Heun ODE from t=1 down to t=sigma_min.

    Reference: OmniControl / Flux style flow matching.
    """

    def __init__(
        self,
        num_steps: int = 28,
        sigma_min: float = 0.0,
        use_heun: bool = False,
        time_shift_mu: float = 0.0,
    ):
        self.num_timesteps = int(num_steps)
        self.flow_num_steps = int(num_steps)
        self.flow_sigma_min = float(sigma_min)
        self.flow_use_heun = bool(use_heun)
        self.flow_time_shift_mu = float(time_shift_mu)
        # Dummy alphas_cumprod for Min-SNR / stratified_t compatibility
        self.alphas_cumprod = np.ones(num_steps, dtype=np.float64)

    @staticmethod
    def _cfg_zero_star_scale(v_c: th.Tensor, v_u: th.Tensor) -> th.Tensor:
        """CFG-Zero* optimized scale st* = ⟨v_c, v_u⟩ / ‖v_u‖² (per sample)."""
        b = v_c.shape[0]
        pos = v_c.reshape(b, -1)
        neg = v_u.reshape(b, -1)
        dot = (pos * neg).sum(dim=1, keepdim=True)
        sq = (neg * neg).sum(dim=1, keepdim=True) + 1e-8
        return (dot / sq).view(b, *([1] * (v_c.ndim - 1)))

    @classmethod
    def _cfg_zero_star_mix(cls, v_c: th.Tensor, v_u: th.Tensor, cfg_scale: float) -> th.Tensor:
        """CFG-Zero* optimized scale (arXiv:2503.18886)."""
        st = cls._cfg_zero_star_scale(v_c, v_u)
        return v_u * st + cfg_scale * (v_c - v_u * st)

    @staticmethod
    def _latent_freq_decompose_bilinear(x: th.Tensor, tsize=(16, 16)):
        h, w = x.shape[-2], x.shape[-1]
        lo = th.nn.functional.interpolate(
            th.nn.functional.interpolate(
                x, size=tsize, mode="bilinear", align_corners=False
            ),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        return lo, x - lo

    @staticmethod
    def _latent_freq_decompose_fft(x: th.Tensor, ratio: float = 0.3):
        h, w = x.shape[-2], x.shape[-1]
        ratio = float(max(0.05, min(0.6, ratio)))
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy = th.arange(h, device=x.device, dtype=th.float32) - cy
        xx = th.arange(w, device=x.device, dtype=th.float32) - cx
        gy, gx = th.meshgrid(yy, xx, indexing="ij")
        radius = ratio * min(h, w) / 2.0
        mask = ((gy ** 2 + gx ** 2) <= radius ** 2).to(x.dtype)
        mask = mask.view(1, 1, h, w)
        xf = th.fft.fftshift(th.fft.fft2(x.float(), dim=(-2, -1)), dim=(-2, -1))
        lo = th.fft.ifft2(
            th.fft.ifftshift(xf * mask, dim=(-2, -1)), dim=(-2, -1)
        ).real.to(dtype=x.dtype)
        hi = th.fft.ifft2(
            th.fft.ifftshift(xf * (1.0 - mask), dim=(-2, -1)), dim=(-2, -1)
        ).real.to(dtype=x.dtype)
        return lo, hi

    @classmethod
    def _latent_freq_decompose(cls, x, mode="fft", ratio=0.3, tsize=(16, 16)):
        if str(mode).lower() == "bilinear":
            return cls._latent_freq_decompose_bilinear(x, tsize=tsize)
        return cls._latent_freq_decompose_fft(x, ratio=ratio)

    @staticmethod
    def _facfg_smooth_time(t: float) -> float:
        t = float(max(0.0, min(1.0, t)))
        return 0.5 * (1.0 - math.cos(math.pi * t))

    @classmethod
    def _facfg_resolve_ratio(
        cls,
        freq_ratio: float,
        freq_ratio_min: float | None,
        t_cur: float | None,
        freq_schedule: str,
    ) -> float:
        sched = str(freq_schedule or "legacy").lower()
        r_max = float(freq_ratio)
        if sched not in ("ratio_anneal", "fresca_ratio_anneal") or t_cur is None:
            return r_max
        r_min = float(freq_ratio_min) if freq_ratio_min is not None else max(0.05, r_max - 0.15)
        t = float(max(0.0, min(1.0, t_cur)))
        # t=1 is the noisy end with a small ratio (narrow low-frequency band); t=0 is the clean end with a large ratio
        return r_min + (r_max - r_min) * (1.0 - t)

    @classmethod
    def _facfg_resolve_weights(
        cls,
        w_low: float,
        w_high: float,
        w_low_min: float,
        w_high_min: float,
        t_cur: float | None,
        freq_schedule: str,
        time_schedule: bool = True,
    ) -> tuple[float, float]:
        w_l = float(w_low)
        w_h = float(w_high)
        w_l_min = float(w_low_min)
        w_h_min = float(w_high_min)
        sched = str(freq_schedule or "legacy").lower()
        if sched in ("ratio_anneal", "fresca_ratio_anneal"):
            weight_sched = "dual_cosine"
        elif sched == "fresca":
            weight_sched = "dual_cosine"
        else:
            weight_sched = sched
        if weight_sched == "static":
            pass
        elif not time_schedule and weight_sched == "legacy":
            pass
        elif t_cur is not None and weight_sched != "static":
            t = float(max(0.0, min(1.0, t_cur)))
            if weight_sched in ("dual_linear", "ratio_anneal"):
                w_l = w_l_min + (w_l - w_l_min) * t
                w_h = w_h_min + (w_h - w_h_min) * (1.0 - t)
            elif weight_sched == "dual_cosine":
                ts = cls._facfg_smooth_time(t)
                w_l = w_l_min + (w_l - w_l_min) * ts
                w_h = w_h_min + (w_h - w_h_min) * (1.0 - ts)
            else:
                w_h = 1.0 + (w_h - 1.0) * t
        return w_l, w_h

    @classmethod
    def _cfg_fresca_mix(
        cls,
        v_c: th.Tensor,
        v_u: th.Tensor,
        w_low: float,
        w_high: float,
        *,
        freq_mode: str = "fft",
        freq_ratio: float = 0.3,
        freq_ratio_min: float | None = None,
        freq_tsize=(16, 16),
        t_cur: float | None = None,
        time_schedule: bool = False,
        freq_schedule: str = "fresca",
        w_low_min: float = 1.0,
        w_high_min: float = 1.0,
    ) -> th.Tensor:
        """Fresca style: split the CFG delta into frequency bands, then scale each band."""
        ratio = cls._facfg_resolve_ratio(freq_ratio, freq_ratio_min, t_cur, freq_schedule)
        delta = v_c - v_u
        delta_lo, delta_hi = cls._latent_freq_decompose(
            delta, mode=freq_mode, ratio=ratio, tsize=freq_tsize
        )
        s_l, s_h = cls._facfg_resolve_weights(
            w_low, w_high, w_low_min, w_high_min, t_cur, freq_schedule, time_schedule
        )
        return v_u + s_l * delta_lo + s_h * delta_hi

    @classmethod
    def _cfg_freq_weighted_mix(
        cls,
        v_c: th.Tensor,
        v_u: th.Tensor,
        w_low: float,
        w_high: float,
        *,
        freq_mode: str = "fft",
        freq_ratio: float = 0.3,
        freq_ratio_min: float | None = None,
        freq_tsize=(16, 16),
        t_cur: float | None = None,
        time_schedule: bool = False,
        freq_schedule: str = "legacy",
        w_low_min: float = 1.0,
        w_high_min: float = 1.0,
    ) -> th.Tensor:
        """FA-CFG*: apply CFG separately to the low and high frequency bands.

        ``freq_schedule``:
          - ``static``: keep w_low / w_high fixed throughout;
          - ``legacy``: only w_high decays from w_high to 1 with t (strongest at the noisy end t=1);
          - ``dual_linear``: linear schedule for both w_low and w_high;
          - ``dual_cosine``: like dual_linear but with a cosine smoothstep;
          - ``ratio_anneal``: dual_cosine weights plus ratio(t) annealing from min to max;
          - ``fresca`` / ``fresca_ratio_anneal``: frequency split of the delta (see ``_cfg_fresca_mix``).
        """
        sched = str(freq_schedule or "legacy").lower()
        if sched in ("fresca", "fresca_ratio_anneal"):
            return cls._cfg_fresca_mix(
                v_c, v_u, w_low, w_high,
                freq_mode=freq_mode,
                freq_ratio=freq_ratio,
                freq_ratio_min=freq_ratio_min,
                freq_tsize=freq_tsize,
                t_cur=t_cur,
                time_schedule=time_schedule,
                freq_schedule=freq_schedule,
                w_low_min=w_low_min,
                w_high_min=w_high_min,
            )
        ratio = cls._facfg_resolve_ratio(freq_ratio, freq_ratio_min, t_cur, sched)
        v_c_lo, v_c_hi = cls._latent_freq_decompose(
            v_c, mode=freq_mode, ratio=ratio, tsize=freq_tsize
        )
        v_u_lo, v_u_hi = cls._latent_freq_decompose(
            v_u, mode=freq_mode, ratio=ratio, tsize=freq_tsize
        )
        if sched == "static":
            weight_sched = "static"
        elif not time_schedule and sched == "legacy":
            weight_sched = "static"
        else:
            weight_sched = sched
        w_l, w_h = cls._facfg_resolve_weights(
            w_low, w_high, w_low_min, w_high_min, t_cur, weight_sched, time_schedule
        )
        v_lo = v_u_lo + w_l * (v_c_lo - v_u_lo)
        v_hi = v_u_hi + w_h * (v_c_hi - v_u_hi)
        return v_lo + v_hi

    @classmethod
    def _cfg_zero_star_facfg_mix(
        cls,
        v_c: th.Tensor,
        v_u: th.Tensor,
        w_low: float,
        w_high: float,
        *,
        freq_mode: str = "fft",
        freq_ratio: float = 0.3,
        freq_ratio_min: float | None = None,
        freq_tsize=(16, 16),
        t_cur: float | None = None,
        time_schedule: bool = False,
        freq_schedule: str = "legacy",
        w_low_min: float = 1.0,
        w_high_min: float = 1.0,
        zero_star_scale: str = "band",
    ) -> th.Tensor:
        """FA-CFG + CFG-Zero*: correct uncond per band with st*, then apply w_low/w_high CFG.

        ``zero_star_scale``:
          - ``band`` (default): compute st*_lo / st*_hi per band;
          - ``global``: one shared st* for the whole velocity field, then band-wise FA-CFG.
        """
        sched = str(freq_schedule or "legacy").lower()
        ratio = cls._facfg_resolve_ratio(freq_ratio, freq_ratio_min, t_cur, sched)
        v_c_lo, v_c_hi = cls._latent_freq_decompose(
            v_c, mode=freq_mode, ratio=ratio, tsize=freq_tsize
        )
        v_u_lo, v_u_hi = cls._latent_freq_decompose(
            v_u, mode=freq_mode, ratio=ratio, tsize=freq_tsize
        )
        if sched == "static":
            weight_sched = "static"
        elif not time_schedule and sched == "legacy":
            weight_sched = "static"
        else:
            weight_sched = sched
        w_l, w_h = cls._facfg_resolve_weights(
            w_low, w_high, w_low_min, w_high_min, t_cur, weight_sched, time_schedule
        )
        scale_mode = str(zero_star_scale or "band").lower()
        if scale_mode == "global":
            st = cls._cfg_zero_star_scale(v_c, v_u)
            st_lo = st_hi = st
        else:
            st_lo = cls._cfg_zero_star_scale(v_c_lo, v_u_lo)
            st_hi = cls._cfg_zero_star_scale(v_c_hi, v_u_hi)
        if sched in ("fresca", "fresca_ratio_anneal"):
            v_lo = v_u_lo * st_lo + w_l * (v_c_lo - v_u_lo * st_lo)
            v_hi = v_u_hi * st_hi + w_h * (v_c_hi - v_u_hi * st_hi)
            return v_lo + v_hi
        v_lo = v_u_lo * st_lo + w_l * (v_c_lo - v_u_lo * st_lo)
        v_hi = v_u_hi * st_hi + w_h * (v_c_hi - v_u_hi * st_hi)
        return v_lo + v_hi

    def sample_t(self, batch_size: int, device) -> th.Tensor:
        """Logit-normal time sampling: t = sigmoid(randn())."""
        t = th.sigmoid(th.randn((batch_size,), device=device))
        return t

    def q_sample(self, x_start: th.Tensor, t: th.Tensor,
                 noise: th.Tensor = None) -> th.Tensor:
        """
        Linear interpolation forward process: x_t = (1−t)·x₀ + t·ε.
        t: (B,) continuous in [0, 1].
        """
        if noise is None:
            noise = th.randn_like(x_start)
        t_ = t.view(-1, *([1] * (x_start.ndim - 1)))
        return (1.0 - t_) * x_start + t_ * noise

    # ------------------------------------------------------------------
    #  ODE Sampling (Euler / Heun)
    # ------------------------------------------------------------------

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        start_t=None,
    ):
        """ODE sampling loop for flow matching (Euler by default, Heun optional).

        CFG support: when batch is 2n with n-sized conditioning tensors,
        the sampler doubles kwargs, runs the model, and mixes velocity
        predictions internally: v = v_u + cfg * (v_c - v_u).

        start_t: if set in (0,1], begin ODE at the schedule node nearest to
        this t (for SDEdit / GeNIe). ``noise`` must then be x_t at that time,
        not pure Gaussian.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        B = shape[0]
        N = self.flow_num_steps
        sigma_min = self.flow_sigma_min

        # Time discretisation: from 1.0 down to sigma_min (N+1 points, N steps)
        ts = th.linspace(1.0, sigma_min, N + 1, device=device)

        # Flux-style time shift
        if self.flow_time_shift_mu > 0:
            mu = self.flow_time_shift_mu
            ts = mu * ts / (1.0 + (mu - 1.0) * ts)

        start_i = 0
        if start_t is not None:
            st = float(start_t)
            # pick node on the decreasing schedule closest to st
            start_i = int(th.argmin(th.abs(ts[:-1] - st)).item())
            start_i = max(0, min(start_i, N - 1))

        # ---- CFG setup ----
        _model_kwargs = dict(model_kwargs) if model_kwargs else {}
        cfg_scale = float(_model_kwargs.pop('cfg_scale', 1.0))
        cfg_zero_star = bool(_model_kwargs.pop('cfg_zero_star', False))
        cfg_zero_init_steps = _model_kwargs.pop('cfg_zero_init_steps', None)
        if cfg_zero_init_steps is None:
            cfg_zero_init_steps = max(1, int(round(0.04 * N)))
        else:
            cfg_zero_init_steps = int(cfg_zero_init_steps)
        cfg_freq_weighted = bool(_model_kwargs.pop('cfg_freq_weighted', False))
        cfg_w_low = float(_model_kwargs.pop('cfg_w_low', cfg_scale))
        cfg_w_high = float(_model_kwargs.pop('cfg_w_high', 1.0))
        cfg_freq_mode = str(_model_kwargs.pop('cfg_freq_mode', 'fft'))
        cfg_freq_ratio = float(_model_kwargs.pop('cfg_freq_ratio', 0.3))
        cfg_freq_ratio_min = _model_kwargs.pop('cfg_freq_ratio_min', None)
        if cfg_freq_ratio_min is not None:
            cfg_freq_ratio_min = float(cfg_freq_ratio_min)
        cfg_freq_tsize = _model_kwargs.pop('cfg_freq_tsize', (16, 16))
        if isinstance(cfg_freq_tsize, (list, tuple)):
            cfg_freq_tsize = tuple(int(x) for x in cfg_freq_tsize)
        else:
            cfg_freq_tsize = (16, 16)
        cfg_freq_time_schedule = bool(_model_kwargs.pop('cfg_freq_time_schedule', True))
        cfg_freq_schedule = str(_model_kwargs.pop('cfg_freq_schedule', 'legacy'))
        if not cfg_freq_time_schedule and cfg_freq_schedule == 'legacy':
            cfg_freq_schedule = 'static'
        cfg_w_low_min = float(_model_kwargs.pop('cfg_w_low_min', 1.0))
        cfg_w_high_min = float(_model_kwargs.pop('cfg_w_high_min', 1.0))
        cfg_zero_star_scale = str(_model_kwargs.pop('cfg_zero_star_scale', 'band'))
        use_cfg = cfg_scale > 1.0 and B % 2 == 0 and B >= 2
        half = B // 2
        if use_cfg:
            for k, v in _model_kwargs.items():
                if isinstance(v, th.Tensor) and v.shape[0] == half:
                    # Doubled: first half = condition, second half = zeros (uncond)
                    _model_kwargs[k] = th.cat([v, th.zeros_like(v)], dim=0)

        for i in range(start_i, N):
            t_cur = ts[i]
            t_next = ts[i + 1]
            t_in = t_cur.expand(B).to(device) * 1000.0
            zero_init = bool(cfg_zero_star and use_cfg and i < cfg_zero_init_steps)
            _step_kw = dict(
                cfg_zero_star=cfg_zero_star,
                zero_init=zero_init,
                cfg_freq_weighted=cfg_freq_weighted,
                cfg_w_low=cfg_w_low,
                cfg_w_high=cfg_w_high,
                cfg_freq_mode=cfg_freq_mode,
                cfg_freq_ratio=cfg_freq_ratio,
                cfg_freq_ratio_min=cfg_freq_ratio_min,
                cfg_freq_tsize=cfg_freq_tsize,
                cfg_freq_time_schedule=cfg_freq_time_schedule,
                cfg_freq_schedule=cfg_freq_schedule,
                cfg_w_low_min=cfg_w_low_min,
                cfg_w_high_min=cfg_w_high_min,
                cfg_zero_star_scale=cfg_zero_star_scale,
                t_cur=float(t_cur.item()) if hasattr(t_cur, "item") else float(t_cur),
            )

            if self.flow_use_heun:
                # Heun 2nd order
                v1 = self._model_step(
                    model, img, t_in, _model_kwargs,
                    use_cfg, cfg_scale, half,
                    **_step_kw,
                )
                x_euler = img + (t_next - t_cur) * v1
                t_in2 = t_next.expand(B).to(device) * 1000.0
                t_next_f = float(t_next.item()) if hasattr(t_next, "item") else float(t_next)
                v2 = self._model_step(
                    model, x_euler, t_in2, _model_kwargs,
                    use_cfg, cfg_scale, half,
                    **{**_step_kw, "t_cur": t_next_f},
                )
                img = img + (t_next - t_cur) * 0.5 * (v1 + v2)
            else:
                # Euler 1st order
                v = self._model_step(
                    model, img, t_in, _model_kwargs,
                    use_cfg, cfg_scale, half,
                    **_step_kw,
                )
                img = img + (t_next - t_cur) * v

        # Return only protein channels (first half if CFG)
        img = img[:, :4] if img.shape[1] > 4 else img
        return img[:half] if use_cfg else img

    def _model_step(
        self,
        model,
        img,
        t_in,
        kwargs,
        use_cfg,
        cfg_scale,
        half,
        cfg_zero_star: bool = False,
        zero_init: bool = False,
        cfg_freq_weighted: bool = False,
        cfg_w_low: float = 1.25,
        cfg_w_high: float = 1.05,
        cfg_freq_mode: str = "fft",
        cfg_freq_ratio: float = 0.3,
        cfg_freq_ratio_min: float | None = None,
        cfg_freq_tsize=(16, 16),
        cfg_freq_time_schedule: bool = True,
        cfg_freq_schedule: str = "legacy",
        cfg_w_low_min: float = 1.0,
        cfg_w_high_min: float = 1.0,
        cfg_zero_star_scale: str = "band",
        t_cur: float | None = None,
    ):
        """Single ODE step. If CFG, runs 2B model and mixes outputs."""
        out = model(img, t_in, **kwargs)
        if use_cfg:
            v_c = out[:half]
            v_u = out[half:]
            if zero_init:
                v_mix = th.zeros_like(v_c)
            elif cfg_freq_weighted and cfg_zero_star:
                v_mix = self._cfg_zero_star_facfg_mix(
                    v_c, v_u, cfg_w_low, cfg_w_high,
                    freq_mode=cfg_freq_mode,
                    freq_ratio=cfg_freq_ratio,
                    freq_ratio_min=cfg_freq_ratio_min,
                    freq_tsize=cfg_freq_tsize,
                    t_cur=t_cur,
                    time_schedule=cfg_freq_time_schedule,
                    freq_schedule=cfg_freq_schedule,
                    w_low_min=cfg_w_low_min,
                    w_high_min=cfg_w_high_min,
                    zero_star_scale=cfg_zero_star_scale,
                )
            elif cfg_freq_weighted:
                v_mix = self._cfg_freq_weighted_mix(
                    v_c, v_u, cfg_w_low, cfg_w_high,
                    freq_mode=cfg_freq_mode,
                    freq_ratio=cfg_freq_ratio,
                    freq_ratio_min=cfg_freq_ratio_min,
                    freq_tsize=cfg_freq_tsize,
                    t_cur=t_cur,
                    time_schedule=cfg_freq_time_schedule,
                    freq_schedule=cfg_freq_schedule,
                    w_low_min=cfg_w_low_min,
                    w_high_min=cfg_w_high_min,
                )
            elif cfg_zero_star:
                v_mix = self._cfg_zero_star_mix(v_c, v_u, cfg_scale)
            else:
                v_mix = v_u + cfg_scale * (v_c - v_u)
            return th.cat([v_mix, v_mix], dim=0)
        return out

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True,
                      denoised_fn=None, cond_fn=None, model_kwargs=None,
                      device=None, progress=False):
        """Alias for ddim_sample_loop (same ODE solver)."""
        return self.ddim_sample_loop(
            model, shape, noise=noise, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs,
            device=device, progress=progress,
        )
