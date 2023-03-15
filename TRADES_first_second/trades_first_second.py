import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import copy


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                alpha=0.2,
                distance='l_inf'):
    # define KL-loss
    model2 = copy.deepcopy(model)
    
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    model2.eval()
    batch_size = len(x_natural)
    
    with torch.no_grad():
        for name, param in model2.named_parameters():
            if 'conv' in name:
                param.add_(0.0001 * torch.randn(param.shape).cuda().detach())

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model2(x_adv), dim=1),
                                       F.softmax(model2(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    loss_natural = F.cross_entropy(model(x_natural), y)
    loss_natural.backward(retain_graph=True)
    for name, param in model.named_parameters():
        if name == 'fc.weight':
            tl1_natural = abs(torch.sum(param.grad,1))
            tl2_natural = torch.sum(param.grad,1)**2
            
    optimizer.zero_grad()
    loss_adv = F.cross_entropy(model(x_adv), y)
    loss_adv.backward()
    for name, param in model.named_parameters():
        if name == 'fc.weight':
            tl1_adv = abs(torch.sum(param.grad,1))
            tl2_adv = torch.sum(param.grad,1)**2
    
    def reg_gap(tl_gap):
        tl_gap = tl_gap-torch.min(tl_gap)
        tl_gap = tl_gap/torch.sum(tl_gap)
        
        if len(tl_gap)==10:
            cc = 0.8
        elif len(tl_gap)==100:
            cc = 0.98
        
        tt = torch.zeros(len(tl_gap),len(tl_gap)).cuda()
        for i in range(len(tt)):
            tt[i] = tl_gap 
        for i in range(len(tt)):
            tt[i,i]=0.
        tt = cc*tt/tt.sum(1).reshape(-1,1)
        for i in range(len(tt)):
            tt[i,i]=1-cc
        return tt
        
    tl1_gap = reg_gap(tl1_natural-tl1_adv)
    tl1_gap_ = reg_gap(tl1_adv-tl1_natural)
    tl2_gap = reg_gap(tl2_natural-tl2_adv)
    tl2_gap_ = reg_gap(tl2_adv-tl2_natural)
         
    y1 = tl1_gap[y]
    y2 = tl1_gap_[y]
    yy1 = tl2_gap_[y]
    yy2 = tl2_gap_[y]
    
    optimizer.zero_grad()
  
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    
    loss_tl1 = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),y1)
    loss_tl1_ = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_natural), dim=1),y2)
    loss_tl2 = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),yy1)
    loss_tl2_ = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_natural), dim=1),yy2)
    
    first_loss = 0.5*(loss_tl1+loss_tl1_)
    second_loss = 0.5*(loss_tl2+loss_tl2_)
    
    loss = loss_natural*(1-alpha) + alpha*first_loss + 0.5*alpha*second_loss + beta * loss_robust 
    return loss

def trades_adv(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv