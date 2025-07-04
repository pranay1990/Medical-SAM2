from .btcv import BTCV
from .amos import AMOS
from .aisd import AISD
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os



def get_dataloader(args):
    # transform_train = transforms.Compose([
    #     transforms.Resize((args.image_size,args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_train_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.Resize((args.image_size, args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])
    
    if args.dataset == 'btcv':
        '''btcv data'''
        btcv_train_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        btcv_test_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(btcv_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'amos':
        '''amos data'''
        amos_train_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        amos_test_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(amos_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(amos_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'aisd':
        '''aisd data'''
        image_root = os.path.join(args.data_path, "images")
        case_ids = sorted(os.listdir(image_root))
        split_idx = int(len(case_ids) * 0.8)
        train_ids = case_ids[:split_idx]
        test_ids = case_ids[split_idx:]

        train_dataset = AISD(
            args,
            args.data_path,
            case_ids=train_ids,
            transform=None,
            transform_msk=None,
            mode="Training",
            prompt=None if args.use_spg else args.prompt,
        )
        test_dataset = AISD(
            args,
            args.data_path,
            case_ids=test_ids,
            transform=None,
            transform_msk=None,
            mode="Test",
            prompt=None if args.use_spg else args.prompt,
        )

        nice_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
        nice_test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader
