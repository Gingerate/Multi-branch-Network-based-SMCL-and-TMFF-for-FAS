from dataset import AlignedDataset
from torch.utils.data import DataLoader
from torch import nn
from model import FaceModel
from options import opt
import torchvision.utils as vutils
import os
import torch
from statistics import PADMeter
import logging
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler
from test import eval_model

file_name = os.path.join(opt.checkpoints_dir, opt.name, "log")
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename=file_name, filemode="w")
run_dir = os.path.join(opt.checkpoints_dir, opt.name, "runs")
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
writer = SummaryWriter(log_dir=run_dir)

if __name__ == '__main__':
    best_res = 101
    train_batch_size = opt.batch_size
    test_batch_size = opt.batch_size

    train_file_list = opt.train_file_list
    dev_file_list = opt.dev_file_list
    test_file_list = opt.test_file_list
    model = FaceModel(opt, isTrain=True, input_nc=3)
    test_data_loader = DataLoader(AlignedDataset(test_file_list, isTrain=False), batch_size=test_batch_size,
                                  shuffle=True, num_workers=8,drop_last=True)
    # dev_data_loader = DataLoader(AlignedDataset(dev_file_list, isTrain=False), batch_size=test_batch_size,
    #                              shuffle=True, num_workers=8)

    train_dataset = AlignedDataset(train_file_list)
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                   shuffle=True, num_workers=8,drop_last=True)

    writer.iter = 0
    for e in range(opt.epoch):
        model.train()
        pad_meter_train = PADMeter()
        for i, data in enumerate(train_data_loader):
            model.set_input(data)
            model.optimize_parameters()

        # model.scheduler_cls.step()

        if e % 1 == 0:
            model.eval()
            with torch.no_grad():

                pad_meter = eval_model(test_data_loader, model)
                pad_meter.get_eer_and_thr()
                # pad_meter.get_eer_and_thr_dev_threshold()
                pad_meter.get_accuracy(pad_meter.threshold)

                logging.info("epoch %d" % e)
                logging.info('FP {pad_meter.FP:.4f} TN {pad_meter.TN:.4f} FN {pad_meter.FN:.4f} TP {pad_meter.TP:.4f}'.format(
                        pad_meter=pad_meter))
                logging.info('HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                    pad_meter=pad_meter))
                is_best = pad_meter.hter <= best_res
                best_res = min(pad_meter.hter, best_res)
                if is_best:
                    best_name = "best"
                    model.save_networks(best_name)

        filename = "lastest"
        model.save_networks(filename)