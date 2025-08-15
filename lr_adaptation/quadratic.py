import torch

# import numpy as np


def deal_with_scaling(scaling):
    """Figure out upper and lower bound when scaling is a tuple or a scalar"""
    scale = scaling
    if isinstance(scale, tuple):
        if len(scale) > 2:
            raise ValueError(
                f"Length of tuple too long. Expected 1 or 2 got {len(scale)}"
            )
        elif len(scale) == 1:
            scale = scale[0]  # pass to float below
        else:
            sort_scale = sorted(scale)
            if 0 < sort_scale[0] < 1 and sort_scale[1] > 1:
                downscale, upscale = sort_scale
            else:
                raise ValueError("Invalid scaling values")

    if isinstance(scale, float):
        if 0 < scale and scale < 1:
            downscale = scale
            upscale = 1 / scale
        elif 1 < scale:
            downscale = 1 / scale
            upscale = scale
        else:
            raise ValueError("Scaling must be positive")

    return downscale, upscale


#############
# Correction terms for various optimizers
############

#
# def sgd_correction(self, group):
#     beta = group['momentum']
#     if beta == 0:
#         # correction_term = torch.tensor(1.0)
#         correction_term =1.0
#     else:
#         t = self.state['step']
#         # c =(1.0-beta**t)/(1.0-beta)
#         # correction_term = c
#         correction_term = 1.0
#
#
#     return torch.tensor(correction_term)
#
#
# def standard_correction(self, group):
#     return torch.tensor(1.0)
#
#
# def adam_correction(self, group):
#     beta1, beta2 = group['betas']
#     t = self.state['step']
#
#     # correction_term = (1 - beta1**t)
#
#     # correction_term = (1 - beta1)
#     correction_term=1.0
#
#     return torch.tensor(correction_term)
#     # return torch.tensor(1.0)


#################
# Pointers to gradient estimates
#################

# def standard_grad(self, p):
#     return p.grad
#
#
# def sgd_grad(self, p):
#     if 'momentum_buffer' in self.optimizer.state[p]:
#         # return self.optimizer.state[p]['momentum_buffer']
#         return p.grad
#     else:
#         return p.grad
#
#
# def adam_grad(self, p):
#     # return p
#     # return self.optimizer.state[p]['exp_avg']
#     return p.grad


##################
# get handle to correction term and gradient estimate
##################
# def get_optimizer_properties(opt):
#     opt_name = opt.__class__.__name__
#     # print(opt_name)
#     # print(opt.param_groups[0])
#     if opt_name == 'SGD':
#         return sgd_correction, sgd_grad
#
#     elif opt_name in set(('RMSprop', 'Adagrad')):
#         return standard_correction, standard_grad
#
#     elif opt_name == 'Adam':
#         return adam_correction, adam_grad
#
#     else:
#         raise NotImplemented(
#             "The requested optimizer is not added to the index")


class QuadraticAdaptation:
    """Wrapper that iteratively update the learning rate of standard optimizers
    in pytorch according to a quadratic approximation.
    """

    def __init__(
        self,
        optimizer,
        scale=(0.75, 1.25),
        scale_limits=(0.75, 1.25),
        eps=1e-12,
        verbose=False,
    ):
        """
        Wrapper requires:
        scale: scaling to apply when lr too small/large
        """

        self.optimizer = optimizer
        if len(optimizer.param_groups) > 1:
            raise NotImplementedError("Several parameter groups not yet implemented")

        self.state = dict(eps=eps)

        self.defaults = {"scale": scale, **self.optimizer.defaults}

        # Update name to highlight adaptation
        self.__name__ = "adapted_" + optimizer.__class__.__name__

        down_scale, up_scale = deal_with_scaling(scale)

        down_threshold, up_threshold = deal_with_scaling(scale)

        self.state["down_scale"] = down_scale
        self.state["up_scale"] = up_scale
        self.state["down_threshold"] = down_threshold
        self.state["up_threshold"] = up_threshold
        self.state["step"] = 0

        # Keep track of scaling
        self.state["accumulated_scaling"] = torch.tensor(1.0, requires_grad=False)

        self.parameter_copy = {}
        self.save_parameters(zero_data=False)
        self.inner_prod = torch.zeros(1, requires_grad=False)

        # Tensors for storing short term accumulated information
        # self.state['l_acc'] = torch.tensor(0.0, requires_grad=False)
        # self.state['l_acc_squared'] = torch.tensor(0.0, requires_grad=False)
        # self.state['delta_acc'] = torch.tensor(0.0, requires_grad=False)
        # self.state['M'] = M
        # self.state['R'] = torch.tensor(
        #     0.0, requires_grad=False)

        # Identify optimizer and get handles to gradient estimate
        # correction_term is identity for all implemented optimizers except Adam
        # self._correction_term_handle, self._gradient_estimate_handle = get_optimizer_properties(
        #     optimizer)
        # self._correction_term_handle, self._gradient_estimate_handle = get_optimizer_properties(
        #     optimizer)

        self.VERBOSE = verbose

    # def correction_term(self, p):
    #     return self._correction_term_handle(self, p)
    #
    # def gradient_estimate(self, p):
    #     return self._gradient_estimate_handle(self, p)

    def save_parameters(self, zero_data=True):
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    if p not in self.parameter_copy:
                        self.parameter_copy[p] = torch.zeros_like(p.data)
                    self.parameter_copy[p].copy_(p.data)
                    if zero_data:
                        p.data.zero_()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def scale_learning_rate(self, scaling=1.0):
        for group in self.optimizer.param_groups:
            group["lr"] = group["lr"] * scaling

    # def _get_delta_l(self):
    #     """
    #     Convenience function to get delta_l if step<M and otherwise
    #     """
    #     state = self.state
    #     M = state['M']
    #     if state['step'] < M:
    #         delta_l = state['delta_acc'] / state['step']
    #     else:
    #         delta_l = state['delta_acc'] / M
    #     return delta_l

    # def _initialize_deques(self, loss, inner_prod):
    #     """
    #     Initialize memory buffer to track last M losses and differences
    #     """
    #     state = self.state
    #     M = state['M']
    #     state['lastM_losses'] = deque([loss.item()] * M, M)
    #     state['lastM_deltas'] = deque([0.0] * M, M)
    #     state['lastM_deltas'].append(inner_prod.item())
    #     state['delta_acc'] = torch.tensor(
    #         inner_prod.item(), requires_grad=False)

    #     state['l_acc'] = torch.tensor(M * loss.item(), requires_grad=False)

    # def _update_deques(self, loss):
    #     """
    #     Update memory buffer
    #     """
    #     state = self.state
    #     delta = abs(loss.item() - state['lastM_losses'][-1])
    #     state['l_acc'] += loss.item() - state['lastM_losses'].popleft()
    #     state['delta_acc'] += delta - state['lastM_deltas'].popleft()
    #     state['lastM_losses'].append(loss.item())
    #     # state['lastM_losses_squared'].append(loss**2)
    #     state['lastM_deltas'].append(delta)

    ##########
    # Override step function
    ###########

    def step(self, closure):
        # if loss is None:
        #     raise RuntimeError('Loss is required for step')
        self.inner_prod.zero_()
        loss = closure()
        f1 = loss.item()
        loss.backward()

        state = self.state

        ############################
        # Save current parameter values and zero them afterwards
        # to use step functionality of optimizer for v=eta W g
        ############################

        self.save_parameters(zero_data=True)
        # copy original data for parameters and zero current

        # step in optimizer direction go get lr*Wg
        self.optimizer.step()
        # parameters.data now contain -v=-Wg

        if "step" not in state:
            state["step"] = 0

        state["step"] += 1

        ############################
        # Calculate the inner product with gradient
        ############################

        # inner_prod = torch.zeros(1, requires_grad=False)

        for group in self.optimizer.param_groups:
            # correction_term = self.correction_term(group)
            for p in group["params"]:
                if p.grad is None:
                    continue

                # grad_estimate = self.gradient_estimate(p)

                # p.data now contains the step i.e. v=-p.data
                # inner_prod -= torch.sum(grad_estimate *
                self.inner_prod -= torch.sum(p.grad * p.data)  # .div(correction_term)

        eps = state["eps"]

        ##################
        # Take updated optimizer step
        ##################
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    # v = -p.data.clone()
                    # p.data.copy_(originals[p])
                    # p.data.add_(-step_modifier, v)
                    # p.data.add_(-step_modifier, self.parameter_copy[p])
                    p.data.add_(self.parameter_copy[p])

        #######################
        # Update learning rate of optimizer
        #######################

        with torch.no_grad():
            f2 = closure().item()

        delta_f = f1 - f2  # *(1-0.9)
        state["delta_f"] = delta_f

        frac = delta_f / (self.inner_prod.item() / 2 + eps)

        if frac > self.state["up_threshold"]:
            self.scale_learning_rate(scaling=state["up_scale"])
            state["accumulated_scaling"] *= state["up_scale"]

        elif frac < self.state["down_threshold"]:
            self.scale_learning_rate(scaling=state["down_scale"])
            state["accumulated_scaling"] *= state["down_scale"]

        # state['optimizer_lr'] = self.optimizer.param_groups[0]['lr']
        if self.VERBOSE:
            if self.inner_prod.item() < 0:
                print(
                    "\033[91m"
                    + f"f1: {f1:.2e}, f2: {f2:.2e}, f1-f2: {f1 - f2:.2e}, phi: {self.inner_prod.item() / 2:.2e}, df/phi:{frac:.2e}, lr:{state['accumulated_scaling'].item():.2e} "
                    + "\033[0m"
                )
            else:
                print(
                    f"f1: {f1:.2e}, f2: {f2:.2e}, f1-f2: {f1 - f2:.2e}, phi: {self.inner_prod.item() / 2:.2e}, df/phi:{frac:.2e}, lr:{state['accumulated_scaling'].item():.2e} "
                )

        return loss

    def get_hyperparameters(self):
        # hp = self.optimizer.state['defaults']

        # for group in self.param_groups
        # hp = self.defaults
        hp = self.optimizer.defaults

        return hp

    def __repr__(self):
        pres = (
            self.__class__.__name__
            + f" ({self.state['downscale']},{self.state['upscale']}) for:\n"
        )
        pres += self.optimizer.__repr__()

        return pres
