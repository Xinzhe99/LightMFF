import torch
# Optimizers
def Get_optimizers(args, net):
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),lr=args.lr, momentum=0.9)
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.b1, args.b2))  # lr=2e-4
    elif args.optimizer == 'ADAMW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(args.b1, args.b2))  # lr=1e-3
    else:
        optimizer = torch.optim.SGD(net.parameters(),lr=args.lr, momentum=0.9)
    return optimizer

# Loss functions
def Get_loss_func(args):
    criterion_GAN = torch.nn.BCELoss()
    criterion_pixelwise = torch.nn.MSELoss()
    if torch.cuda.is_available():
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()
    return criterion_GAN, criterion_pixelwise

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
