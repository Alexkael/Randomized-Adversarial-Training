from __future__ import print_function
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy

import models
from utils import Bar, Logger, AverageMeter, accuracy
from utils_awp import TradesAWP


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--arch', type=str, default='WideResNet')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='retrain from which epoch')
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
parser.add_argument('--data-path', type=str, default='../data',
                    help='where is the dataset CIFAR-10')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'],
                    help='The threat model')
parser.add_argument('--epsilon', default=8, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2, type=float,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./model-cifar10-WRN3410/first_second',
                    help='directory of model for saving checkpoint')
parser.add_argument('--resume-model', default='', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim', default='', type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--alpha', default=0.2,
                    help='regularization')
parser.add_argument('--awp-gamma', default=0.005, type=float,
                    help='whether or not to add parametric noise')
parser.add_argument('--awp-warmup', default=10, type=int,
                    help='We could apply AWP after some epochs for accelerating.')

args = parser.parse_args()
epsilon = args.epsilon / 255
step_size = args.step_size / 255
if args.awp_gamma <= 0.0:
    args.awp_warmup = np.infty
if args.data == 'CIFAR100':
    NUM_CLASSES = 100
else:
    NUM_CLASSES = 10

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
#args.arch = 'ResNet18'
    
# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = getattr(datasets, args.data)(
    root=args.data_path, train=True, download=True, transform=transform_train)
testset = getattr(datasets, args.data)(
    root=args.data_path, train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def perturb_input(model,
                  x_natural,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf'):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1),
                                   reduction='sum')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.kl_div(F.log_softmax(model(adv), dim=1),
                                       F.softmax(model(x_natural), dim=1),
                                       reduction='sum')
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            # if (grad_norms == 0).any():
            #     delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

def train(model, train_loader, optimizer, epoch, awp_adversary):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    criterion_kl = nn.KLDivLoss(size_average=False)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        x_natural, target = data.to(device), target.to(device)
        
        batch_size = len(x_natural)
        # craft adversarial examples
        x_adv = perturb_input(model=model,
                              x_natural=x_natural,
                              step_size=step_size,
                              epsilon=epsilon,
                              perturb_steps=args.num_steps,
                              distance=args.norm)

        model.train()
        # calculate adversarial weight perturbation
        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=x_adv,
                                         inputs_clean=x_natural,
                                         targets=target,
                                         beta=args.beta)
            awp_adversary.perturb(awp)

        optimizer.zero_grad()
        output = model(x_natural)
        output_adv = model(x_adv)
        ###############################approximately train on Taylor terms (ref Ineq. (8) andAppendix D)##############################
        optimizer.zero_grad()
        # calculate robust loss
        loss_natural = F.cross_entropy(output, target)
        loss_natural.backward(retain_graph=True)
        for name, param in model.named_parameters():
            if name == 'module.fc.weight':
                tl1_natural = abs(torch.sum(param.grad,1))
                tl2_natural = torch.sum(param.grad,1)**2   #use Hadamard product to replace Kronecker product

        optimizer.zero_grad()
        loss_adv = F.cross_entropy(output_adv, target)
        loss_adv.backward(retain_graph=True)
        for name, param in model.named_parameters():
            if name == 'module.fc.weight':
                tl1_adv = abs(torch.sum(param.grad,1))
                tl2_adv = torch.sum(param.grad,1)**2   #use Hadamard product to replace Kronecker product

        def reg_gap(tl_gap):
            if args.data == 'CIFAR10':
                cc = 0.8
            elif args.data == 'CIFAR100':
                cc = 0.98
                
            tl_gap = tl_gap-torch.min(tl_gap)
            tl_gap = tl_gap/torch.sum(tl_gap)

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

        y1 = tl1_gap[target]
        y2 = tl1_gap_[target]
        yy1 = tl2_gap_[target]
        yy2 = tl2_gap_[target]

        optimizer.zero_grad()
        
        loss_robust = F.kl_div(F.log_softmax(output_adv, dim=1),
                               F.softmax(output, dim=1),
                               reduction='batchmean')
        
        loss_tl1 = (1.0 / batch_size) * criterion_kl(F.log_softmax(output_adv, dim=1),y1)
        loss_tl1_ = (1.0 / batch_size) * criterion_kl(F.log_softmax(output, dim=1),y2)
        loss_tl2 = (1.0 / batch_size) * criterion_kl(F.log_softmax(output_adv, dim=1),yy1)
        loss_tl2_ = (1.0 / batch_size) * criterion_kl(F.log_softmax(output, dim=1),yy2)
        
        first_loss = 0.5*(loss_tl1+loss_tl1_)
        second_loss = 0.5*(loss_tl2+loss_tl2_)
        ################################################################################################################
        loss = loss_natural*(1-args.alpha) + args.beta*loss_robust + args.alpha*first_loss + args.alpha*0.5*second_loss

        prec1, prec5 = accuracy(output_adv, target, topk=(1, 5))
        losses.update(loss.item(), x_natural.size(0))
        top1.update(prec1.item(), x_natural.size(0))

        # update the parameters at last
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=epsilon,
                  num_steps=args.num_steps,
                  step_size=step_size):
    model.eval()
    out = model(X)

    X_pgd = Variable(X.data, requires_grad=True)
    #if args.random:
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss2 = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss2.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_adv_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                X, y = Variable(data, requires_grad=True), Variable(target)
                data_adv = _pgd_whitebox(copy.deepcopy(model), X, y)
            output = model(data_adv)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.1
    if epoch >= 150:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    # init model, ResNet18() can be also used here for training
    model = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # We use a proxy model to calculate AWP, which does not affect the statistics of BN.
    proxy = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
    awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma)

    criterion = nn.CrossEntropyLoss()

    logger = Logger(os.path.join(model_dir, 'log.txt'), title=args.arch)
    logger.set_names(['Learning Rate',
                      'Adv Train Loss', 'Nat Train Loss', 'Nat Val Loss',
                      'Adv Train Acc.', 'Nat Train Acc.', 'Nat Val Acc.'])

    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device))
    if args.resume_optim:
        optimizer.load_state_dict(torch.load(args.resume_optim, map_location=device))

    tstt = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        start_time = time.time()
        # adjust learning rate for SGD
        lr = adjust_learning_rate(optimizer, epoch)

        # adversarial training
        adv_loss, adv_acc = train(model, train_loader, optimizer, epoch, awp_adversary)

        # evaluation on natural examples
        tstloss, tstacc = eval_test(model, device, test_loader)
        advtstloss, advtstacc = eval_adv_test(model, device, test_loader)
        print('Epoch '+str(epoch)+': '+str(int(time.time()-start_time))+'s', end=', ')
        #print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
        print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(tstloss, 100. * tstacc), end=', ')
        print('adv_test_loss: {:.4f}, adv_test_acc: {:.2f}%'.format(advtstloss, 100. * advtstacc))
        
        tstt.append(advtstacc)
        # save checkpoint
        if (epoch>99 and epoch%5==0) or (epoch>99 and epoch<110) or (epoch>149 and epoch<160) or (epoch>99 and advtstacc==max(tstt)):
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'epoch{}.pt'.format(epoch)))


if __name__ == '__main__':
    main()
