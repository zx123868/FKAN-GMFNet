import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset,Datasets
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90, Resize
import time
from archs import UNext


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='dongmailiuMfinal_UNext_woDS',  # CVC_ClinicDB,kvasir-sessile
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=100)

    model.load_state_dict(torch.load('models/%s/model.pth' % config['name']),strict=False)
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Datasets(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()

    count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()

            # Compute output
            pre_gt, outputs = model(input)
            # If pre_gt and outputs are tuples, ensure they are unpacked correctly
            if isinstance(pre_gt, tuple):
                pre_gt = pre_gt[0]
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            iou_outputs, dice_outputs = iou_score(outputs, target)
            # Assuming pre_gt and outputs are tuples
            pre_gt = pre_gt[0] if isinstance(pre_gt, tuple) else pre_gt
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs

            # Here you may choose how to aggregate these scores; for simplicity, let's use the output scores
            iou_avg_meter.update(iou_outputs, input.size(0))
            dice_avg_meter.update(dice_outputs, input.size(0))

            # Generate binary mask from model output
            outputs = outputs.cpu().numpy()
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            for i in range(len(outputs)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
                                (outputs[i,c] * 255).astype('uint8'))  # .jpg改为.png

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
