import argparse
from data.dataset_denoising import DatasetDenoising
import logging
import numpy as np
import os
import os.path
from typing import Dict, List
import cv2
from prettytable import PrettyTable
from torch.utils.data import DataLoader
import time
from data.select_dataset import select_dataset
from models.model_test import Model
from utils import utils_image as util
from utils import utils_logger
from utils import utils_option as option
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main(config_path: str = 'options/test_denoising.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',
                        type=str,
                        default=config_path,
                        help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.makedirs(
        [path for key, path in opt['path'].items() if 'pretrained' not in key])

    option.save(opt)

    # logger
    logger_name = 'test'
    utils_logger.logger_info(
        logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # data
    opt_data_test = opt["data"]["test"]
    test_sets: List[DatasetDenoising] = select_dataset(opt_data_test, "test")
    test_loaders: List[DataLoader[DatasetDenoising]] = []
    for test_set in test_sets:
        test_loaders.append(
            DataLoader(test_set,
                       batch_size=1,
                       shuffle=False,
                       num_workers=1,
                       drop_last=False,
                       pin_memory=True))

    # model
    model = Model(opt)
    model.init()

    # test
    avg_psnrs: Dict[str, List[float]] = {}
    avg_ssims: Dict[str, List[float]] = {}
    tags = []
    for test_loader in test_loaders:
        test_set: DatasetDenoising = test_loader.dataset
        avg_psnr = 0.
        avg_ssim = 0.
        avg_test_loss = 0.
        for test_data in test_loader:

            # patches
            images, h, w = util.slide_window(test_data['y'])
            y_gts, _, _ = util.slide_window(test_data['y_gt'])
            image_number = len(images['image'])
            dxs = []
            for i in range(image_number):
                model.feed_data(images['image'][i], y_gts['image'][i], test_data['sigma'], test_data['path'])
                dx = model.test(images['image'][i], test_data['sigma'])
                print(dx.size())
                dxs.append(dx.squeeze(0).squeeze(0))
                psnr, ssim, test_loss = model.cal_metrics_test()
                avg_psnr += psnr
                avg_ssim += ssim
                avg_test_loss += test_loss

            image_denoise = util.Slid_cating(dxs, images['x'], images['y'], h, w)
            image = (image_denoise * 255).to('cpu').numpy()
            
            image = np.transpose(image, (1, 2, 0))
            

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # plt.imshow(image, cmap='gray')
            # plt.show()
            outpth = 'outpath' + str(time.time()) + str('.png')
            print(outpth)
            cv2.imwrite(outpth, image)
            model.feed_data_y_gt(test_data['y_gt'])
            model.save_visuals(test_set.tag)
        avg_psnr = round(avg_psnr / (len(test_loader) * image_number), 2)
        avg_ssim = round(avg_ssim * 100 / (len(test_loader) * image_number), 2)
        avg_test_loss = (avg_test_loss.item() / (len(test_loader) * image_number), 2)
        print('test_loss', avg_test_loss)

        name = test_set.name

        if name in avg_psnrs:
            avg_psnrs[name].append(avg_psnr)
            avg_ssims[name].append(avg_ssim)
        else:
            avg_psnrs[name] = [avg_psnr]
            avg_ssims[name] = [avg_ssim]

        tags.append(test_set.tag)

    header = ['Dataset'] + list(set(tags))

    t = PrettyTable(header)
    for key, value in avg_psnrs.items():
        t.add_row([key] + value)
    logger.info(f"Test PSNR:\n{t}")

    t = PrettyTable(header)
    for key, value in avg_ssims.items():
        t.add_row([key] + value)
    logger.info(f"Test SSIM:\n{t}")


main()
