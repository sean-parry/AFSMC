# should put my expected imporvement function here
import gpflow
import torch
import numpy as np

class EI():
    def __init__(self, gp : gpflow.models.GPR, 
                 limits : list[tuple]):
        self.gp = gp
        self.limits = limits
        return

    def _expected_improvement(self, x, y_best):

        x_np = x.detach().numpy()
        mean, var = self.gp.predict_f(x_np)
        mean = torch.tensor(mean, dtype=torch.float64)
        std = torch.tensor(np.sqrt(var), dtype=torch.float64)

        std = torch.maximum(std, torch.tensor(1e-9, dtype=torch.float64))

        z = (mean - y_best) / std

        # Compute EI using the formula
        ei = (mean - y_best) * torch.sigmoid(z) + std * torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        return ei

    def sample(self, x):
        x = torch.tensor(x, dtype=torch.float64)

        if self.limits is not None:
            lower_bound = torch.tensor(self.limits[0], dtype=torch.float64)
            upper_bound = torch.tensor(self.limits[1], dtype=torch.float64)
            is_within_limits = torch.all((x >= lower_bound) & (x <= upper_bound), dim=1)
        else:
            is_within_limits = torch.ones(x.shape[0], dtype=torch.bool)

        y_best = torch.min(torch.tensor(self.gp.data[1], dtype=torch.float64))

        ei = torch.zeros(x.shape[0], dtype=torch.float64)
        ei[is_within_limits] = self._expected_improvement(x[is_within_limits], y_best)

        return ei

