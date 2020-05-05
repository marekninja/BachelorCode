import argparse
import logging
import os
import pathlib
import random
import sys

import numpy as np
import scipy.optimize as opt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.voc_dataset import VOCDataset
from models.multibox_loss import MultiboxLoss
from ssd.config import mobilenetv1_ssd_config
from ssd.config import squeezenet_ssd_config
from ssd.config import vgg_ssd_config
from ssd.data_preprocessing import TestTransform
from ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from ssd.ssd import MatchPrior
from ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from utils import box_utils, measurements
from utils.misc import str2bool, Timer

home = str(Path.home())
sys.path.append(home+'/FIIT/BP/BP/Zdroje_kod/quantization_jupyters/SSD_2')

from quantization.quantizer import ModelQuantizer
from quantization.posttraining.module_wrapper import ActivationModuleWrapperPost, ParameterModuleWrapperPost
from utils.mllog import MLlogger
from itertools import count


parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)
parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
parser.add_argument('-cb', '--cal-batch-size', default=32, type=int, help='Batch size for calibration')
parser.add_argument('-cs', '--cal-set-size', default=32, type=int, help='Batch size for calibration')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--maxiter', '-maxi', type=int, help='Maximum number of iterations to minimize algo', default=1)
parser.add_argument('--maxfev', '-maxf', type=int, help='Maximum number of function evaluations of minimize algo',
                    default=None)

parser.add_argument('--quantize', '-q', action='store_true', help='Enable quantization', default=False)
parser.add_argument('--experiment', '-exp', help='Name of the experiment', default='default')
parser.add_argument('--bit_weights', '-bw', type=int, help='Number of bits for weights', default=None)
parser.add_argument('--bit_act', '-ba', type=int, help='Number of bits for activations', default=None)
parser.add_argument('--pre_relu', dest='pre_relu', action='store_true', help='use pre-ReLU quantization')
parser.add_argument('--qtype', default='max_static', help='Type of quantization method')
parser.add_argument('-lp', type=float, help='p parameter of Lp norm', default=3.)
parser.add_argument('--bcorr_w', '-bcw', action='store_true', help='Bias correction for weights', default=False)

args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
# DEVICE = torch.device("cpu")


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index] = {}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


def validate(args, predictor, coef = 1):
    results = []
    length = (int)(len(dataset) * coef)
    for i in tqdm(range(length)):
        # print("process image", i)
        timer.start("Load Image")
        image = dataset.get_image(i)
        # print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        # print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            args.iou_threshold,
            args.use_2007_metric
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")
        ml_logger.log_metric(class_name, ap)

    mAp = sum(aps) / len(aps)
    print(f"\nAverage Precision Across All Classes:{mAp}")
    return mAp


def create_calibration_set(args, loader, device, size):
    # train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare Validation datasets.")
    val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                             target_transform=target_transform, is_test=False)
    logging.info(val_dataset)

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)

    cal_set = []
    for i, data in enumerate(val_loader):
        if i >= size:
            break
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        cal_set.append((images, boxes, labels))

    return cal_set


def evaluate_calibration(cal_set, net, criterion, device):
    # print('device is: ',device)
    with torch.no_grad():
        res = torch.tensor([0.]).to(device)
        for i in range(len(cal_set)):
            images, boxes, labels = cal_set[i]
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            # compute output
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            # print("Loss: (reg, cls) - ({}, {})".format(regression_loss.item(), classification_loss.item()))
            loss = regression_loss + classification_loss
            res += loss

        return res / len(cal_set)


def create_model(args):
    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')
    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    return net, predictor


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval_pnorm_on_calibration(args, layers, replacement_factory, cal_set, device, p):
    args.qtype = 'lp_norm'
    args.lp = p
    fix_seed(args.seed)
    net, predictor = create_model(args)
    mq = ModelQuantizer(net, args, layers, replacement_factory)
    loss = evaluate_calibration(cal_set, net, criterion, device)
    point = mq.get_clipping()

    del net
    del mq

    return point, loss


if __name__ == '__main__':
    with MLlogger(os.path.join(home, 'mxt-sim/mllog_runs'), args.experiment, args,
                  name_args=['ssd', args.net, args.dataset,
                             "W{}A{}".format(args.bit_weights, args.bit_act)]) as ml_logger:

        fix_seed(args.seed)
        eval_path = pathlib.Path(args.eval_dir)
        eval_path.mkdir(exist_ok=True)
        timer = Timer()
        class_names = [name.strip() for name in open(args.label_file).readlines()]

        if args.net == 'vgg16-ssd':
            create_net = create_vgg_ssd
            config = vgg_ssd_config
        elif args.net == 'mb1-ssd':
            create_net = create_mobilenetv1_ssd
            config = mobilenetv1_ssd_config
        elif args.net == 'mb1-ssd-lite':
            create_net = create_mobilenetv1_ssd_lite
            config = mobilenetv1_ssd_config
        elif args.net == 'sq-ssd-lite':
            create_net = create_squeezenet_ssd_lite
            config = squeezenet_ssd_config
        elif args.net == 'mb2-ssd-lite':
            create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
            config = mobilenetv1_ssd_config
        else:
            logging.fatal("The net type is wrong.")
            parser.print_help(sys.stderr)
            sys.exit(1)

        dataset = VOCDataset(args.dataset, is_test=False)
        true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
        net, predictor = create_model(args)

        cal_loader = DataLoader(dataset, args.cal_batch_size, num_workers=args.num_workers, shuffle=True)
        cal_set = create_calibration_set(args, cal_loader, DEVICE, int(args.cal_set_size / args.cal_batch_size))
        criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                                 center_variance=0.1, size_variance=0.2, device=DEVICE)

        if args.quantize:
            all_convs = [n for n, m in net.named_modules() if isinstance(m, nn.Conv2d)][1: -1]
            all_linear = [n for n, m in net.named_modules() if isinstance(m, nn.Linear)]
            all_relu = [n for n, m in net.named_modules() if isinstance(m, nn.ReLU)][1: -1]
            all_relu6 = [n for n, m in net.named_modules() if isinstance(m, nn.ReLU6)][1: -1]
            layers = all_relu + all_relu6 + all_linear + all_convs

            replacement_factory = {nn.ReLU: ActivationModuleWrapperPost,
                                   nn.ReLU6: ActivationModuleWrapperPost,
                                   nn.Linear: ParameterModuleWrapperPost,
                                   nn.Conv2d: ParameterModuleWrapperPost,
                                   nn.Embedding: ActivationModuleWrapperPost}
            mq = ModelQuantizer(net, args, layers, replacement_factory)

        maxabs_loss = evaluate_calibration(cal_set, net, criterion, DEVICE)
        print("max loss: {:.4f}".format(maxabs_loss.item()))
        # max_point = mq.get_clipping()
        # ml_logger.log_metric('Loss max', maxabs_loss.item(), step='auto')

        del net
        del mq

        ps = np.linspace(2, 4, 10)
        losses = []
        for p in tqdm(ps):
            point, loss = eval_pnorm_on_calibration(args, layers, replacement_factory, cal_set, DEVICE, p)
            losses.append(loss.item())
            print("(p, loss) - ({}, {})".format(p, loss.item()))

        # Interpolate optimal p
        z = np.polyfit(ps, losses, 2)
        y = np.poly1d(z)
        p_intr = y.deriv().roots[0]
        print("p intr: {:.2f}".format(p_intr))

        args.qtype = 'lp_norm'
        args.lp = p_intr
        fix_seed(args.seed)
        net, predictor = create_model(args)
        mq = ModelQuantizer(net, args, layers, replacement_factory)
        loss = evaluate_calibration(cal_set, net, criterion, DEVICE)
        point = mq.get_clipping()
        print("(p intr, loss) - ({}, {})".format(point, loss.item()))
        ml_logger.log_metric("p_intr", p_intr)
        ml_logger.log_metric("loss p_intr", loss.item())

        mAp = validate(args, predictor, 0.1)
        # ml_logger.log_metric('mAp p_intr', mAp)

        # run optimizer
        min_options = {}
        if args.maxiter is not None:
            min_options['maxiter'] = args.maxiter
        if args.maxfev is not None:
            min_options['maxfev'] = args.maxfev
        min_method = 'Powell'
        _iter = count(0)
        init = point


        def local_search_callback(x):
            it = next(_iter)
            mq.set_clipping(x, DEVICE)
            loss = evaluate_calibration(cal_set, net, DEVICE)
            print("\n[{}]: Local search callback".format(it))
            print("loss: {:.4f}\n".format(loss.item()))
            print(x)
            ml_logger.log_metric('Loss {}'.format(min_method), loss.item(), step='auto')

            # evaluate
            acc = validate()
            ml_logger.log_metric('Acc {}'.format(min_method), acc, step='auto')


        _eval_count = count(0)
        _min_loss = 1e6


        def evaluate_calibration_callback(scales, net, mq):
            global _eval_count, _min_loss
            eval_count = next(_eval_count)

            mq.set_clipping(np.abs(scales), DEVICE)
            loss = evaluate_calibration(cal_set, net, criterion, DEVICE).item()

            if loss < _min_loss:
                _min_loss = loss

            print_freq = 20
            if eval_count % 20 == 0:
                print("func eval iteration: {}, minimum loss of last {} iterations: {:.4f}".format(
                    eval_count, print_freq, _min_loss))

            return loss


        res = opt.minimize(lambda scales: evaluate_calibration_callback(scales, net, mq), init.cpu().numpy(),
                           method="Powell", options=min_options)
        res.x = np.abs(res.x)
        print(res)

        scales = np.abs(res.x)
        mq.set_clipping(scales, DEVICE)
        loss = evaluate_calibration(cal_set, net, criterion, DEVICE)
        ml_logger.log_metric('Loss {}'.format(min_method), loss.item(), step='auto')

        # evaluate
        mAp = validate(args, predictor)
        ml_logger.log_metric('mAp {}'.format(min_method), mAp, step='auto')
