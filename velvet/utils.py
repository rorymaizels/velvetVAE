"""functional utils"""
import random
import warnings
from typing import Optional
import torch
import numpy as np

from scvi.distributions._negative_binomial import ZeroInflatedNegativeBinomial, log_zinb_positive


def set_seed(seed=0):
    """
    across numpy and torch, set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NoWarningZINB(ZeroInflatedNegativeBinomial):
    """
    data for Velvet is normalised, which throws a warning with the standard
    ZeroInflatedNegativeBinomial function, so here is a version that won't complain.
    """
    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        zi_logits: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):
        """
        scvi ZINB throws warnings with non-integer data,
        this is a wraparound that doesn't throw those warnings.
        """
        super().__init__(
            total_count=total_count,
            probs=probs,
            logits=logits,
            mu=mu,
            theta=theta,
            zi_logits=zi_logits,
            scale=scale,
            validate_args=validate_args,
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log probability."""
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                )
        return log_zinb_positive(value, self.mu, self.theta, self.zi_logits, eps=1e-08)
