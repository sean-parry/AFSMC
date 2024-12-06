import torch
import utils  

def main():
    branin_func = utils.test_functions_torch.Branin()

    x = torch.tensor([0.0,0.0], requires_grad=True)
    optimizer = torch.optim.LBFGS([x], lr=1.0, max_iter = 100, history_size=10)

    def closure(opt = optimizer, func = branin_func.eval, x = x):
        opt.zero_grad()
        loss = func(x)
        loss.backward()
        return loss

    for _ in range(10):
        optimizer.step(closure)

    print("Optimized x:", x.data)
    print("Minimum value of func:", branin_func.eval(x).item())


main()


