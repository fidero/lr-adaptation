import torch
# import torch.optim as optim



class VA_wrapper():
    ''' Wrapper that iteratively update the learning rate of standard optimizers
    in pytorch according to a quadratic approximation.
    '''

    def __init__(self, optimizer, scale=(0.5, 1.2), eps=1e-12):
        """
        Wrapper requires:
        - A Pytorch optimizer  
        - scale for increasing and decreasing the learning rate
        """
        if not 0.0 < scale[0] <= 1.0:
            raise ValueError("Invalid scale parameter at index 0: {}".format(scale[0]))

        if not 1.0 <= scale[1]:
            raise ValueError("Invalid scale parameter at index 1: {}".format(scale[1]))
        
        if not 0.0 < eps:
            raise ValueError("Invalid eps parameter: {}".format(eps))

        self.optimizer = optimizer


        self.state = dict(eps=eps)
        self.defaults = {"scale":scale,**self.optimizer.defaults}

        # Update name to highlight adaptation
        self.__name__ = 'a-' + optimizer.__class__.__name__

        self.state['downscale'] = scale[0]
        self.state['upscale'] = scale[1]

        # Keep track of scaling
        self.state['accumulated_scaling'] = torch.tensor(
            1.0, requires_grad=False)

        self.state['R'] = torch.tensor(
            0.0, requires_grad=False)


    def zero_grad(self):
        self.optimizer.zero_grad()

    def scale_learning_rate(self, scaling=1.0):
        for group in self.optimizer.param_groups:
            group['lr'] = group['lr'] * scaling

    def step(self, closure, fmin=None, fvar=0.0):

        loss=closure()
        f1 = loss.item()

        loss.backward()
        R = fvar

        state = self.state

        ############################
        # Save current parameter values and zero them afterwards
        # to use step functionality of optimizer for v=eta W g
        ############################

        # copy original data for parameters
        originals = {}

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if p not in originals:
                    originals[p] = torch.zeros_like(p.data)
                originals[p].copy_(p.data)
                p.data.zero_()


        # step in optimizer direction go get lr*Wg
        self.optimizer.step()
        # parameters.data now contain -v=-Wg

        if 'step' not in state:
            state['step'] = 0

        state['step'] += 1

        ############################
        # Calculate the inner product with gradient
        ############################

        phi = torch.zeros(1, requires_grad=False)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # p.data now contains the step i.e. v=-p.data
                phi -= torch.sum(p.grad*p.data)


        state['phi']=phi


        if fmin is None:
            df = phi/2.0
        else:
            df = f1-fmin


        eps = state['eps']

        #########################
        #     Calculate step update
        #########################
        step_modifier = 2.0 * \
            (df / (phi + R + eps)).item()


        # effective scaling of learning rate
        state['step_modifier'] = step_modifier

        ##################
        # Take updated optimizer step
        ##################
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                v = -p.data.clone()
                p.data.copy_(originals[p])
                p.data.add_(-step_modifier, v)


        #######################
        # Update learning rate of optimizer
        #######################
        with torch.no_grad():
            f2 = closure().item()

        delta_f = (f1-f2)
        state['delta_f'] = delta_f

        frac = delta_f / (phi.item() / 2 + eps)

        if frac > 4/3:
            self.scale_learning_rate(scaling=state['upscale'])
            state['accumulated_scaling'] *= state['upscale']

        elif frac < 3/4:
            self.scale_learning_rate(scaling=state['downscale'])
            state['accumulated_scaling'] *= state['downscale']

        return loss


    def get_hyperparameters(self):

        # hp = self.optimizer.state['defaults']

        # for group in self.param_groups
        # hp = self.defaults
        hp=self.optimizer.defaults


        return hp


    def __repr__(self):
        pres= self.__class__.__name__ + f" ({self.state['downscale']},{self.state['upscale']}) for:\n"
        pres+=self.optimizer.__repr__()


        return pres



if __name__ == '__main__':
    import torch.nn as nn
    torch.manual_seed(42)
    print('Executing standalone')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    N = 64
    M = 16
    M1 = 64
    O = 5

    net = nn.Sequential(nn.Linear(M, M1), nn.ReLU(), nn.Linear(M1, O))
    net.to(device)

    # nn.CrossEntropyLoss(reduction='none')
    criterion = nn.MSELoss(reduction='none')

    # optimizer = optim.SGD(net.parameters(), lr=0.01)#, momentum=0.9)
    optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9)
    # optimizer = optim.Adagrad(net.parameters(), lr=0.01)
    # optimizer = optim.RMSprop(net.parameters(), lr=0.1)
    # optimizer = optim.Adam(net.parameters(), betas=(0.2,0.999),lr=0.001)

    correction_term, grad_estimate_handle = get_optimizer_properties(
        optimizer)

    va_optimizer = VA_wrapper(optimizer,verbose=True)
    
    # print(va_optimizer)
    # print(va_optimizer.defaults)

    for i in range(5):        
        X = 0.3*torch.randn(N, M).to(device)
        Y = (5+torch.randn(N, O)).to(device)

        def closure():
            va_optimizer.zero_grad()
            # optimizer.zero_grad()
            outputs = net(X)
            batch_loss = criterion(outputs, Y)
            # R = torch.var(batch_loss)
            loss = torch.mean(batch_loss)
            # loss.backward()
            return loss



        va_optimizer.step(closure)
        # va_optimizer.step(closure,fmin=torch.tensor([0.0],requires_grad=False))
