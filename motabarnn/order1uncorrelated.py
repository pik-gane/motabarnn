import torch
from torch import nn

class Order1Uncorrelated(nn.Module):

  # parameters:
  sigma_psi: torch.Tensor
  """shape: (in_features, hidden_size or 1, out_features) prior stddev of f' (possibly shared across anchor inputs)"""
  chi: torch.Tensor
  """shape: (in_features, hidden_size, 1) anchor inputs (shared across out_features)"""
  sigma_eps: torch.Tensor
  """shape: (hidden_size, out_features) measurement error stddev (possibly shared across anchor inputs)"""
  eta: torch.Tensor
  """shape: (hidden_size, out_features) anchor outputs"""
  mu_phi: torch.Tensor
  """shape: (out_features,) prior means of f(input)"""
  sigma_phi: torch.Tensor
  """shape: (out_features,) prior stddevs of f(input)"""


  def __init__(self, in_features, out_features=1, 
               hidden_size=128, share_sigma_psi=False, share_eps=False):
    super().__init__()

    # parameters:
    self.in_features = in_features
    self.sigma_psi = nn.Parameter(torch.randn(1, in_features, 1 if share_sigma_psi else hidden_size, out_features))
    self.chi = nn.Parameter(torch.randn(1, in_features, hidden_size, 1))
    self.sigma_eps = nn.Parameter(torch.randn(1, 1 if share_eps else hidden_size, out_features))
    self.eta = nn.Parameter(torch.randn(1, hidden_size, out_features))
    self.mu_phi = nn.Parameter(torch.randn(1, out_features))
    self.sigma_phi = nn.Parameter(torch.randn(1, out_features))


  def forward(self, input):
    # see paper for details
    s = ((self.sigma_psi * (self.chi - input.reshape(-1, self.in_features, 1, 1)))**2).sum(axis=1)  # shape: (-1, hidden_size, out_features)
    w = 1 / (s + self.sigma_eps**2)  # shape: (-1, hidden_size, out_features)
    denominator = 1 + self.sigma_phi**2 * w.sum(axis=1)  # shape: (-1, out_features)
    posterior_mu_phi = (self.mu_phi + self.sigma_phi**2 * (w * self.eta).sum(axis=1)) / denominator  # shape: (-1, out_features)
    posterior_sigma2_phi = self.sigma_phi**2 / denominator  # shape: (-1, out_features)
#     print("shapes:", input.shape, s.shape, w.shape, denominator.shape, posterior_mu_phi.shape, posterior_sigma2_phi.shape)
    return posterior_mu_phi, posterior_sigma2_phi
