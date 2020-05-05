import argparse
import logging
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from datasets.voc_dataset import VOCDataset
from ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from utils import str2bool, Timer, box_utils, measurements
from utils.mllog import MLlogger

home = str(Path.home())
sys.path.append(home+'/FIIT/BP/BP/Zdroje_kod/quantization_jupyters/SSD_2')


from quantization.quantizer import ModelQuantizer
from quantization.posttraining.module_wrapper import ActivationModuleWrapperPost, ParameterModuleWrapperPost

parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)

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

parser.add_argument('--quantize', '-q', action='store_true', help='Enable quantization', default=False)
parser.add_argument('--experiment', '-exp', help='Name of the experiment', default='default')
parser.add_argument('--bit_weights', '-bw', type=int, help='Number of bits for weights', default=None)
parser.add_argument('--bit_act', '-ba', type=int, help='Number of bits for activations', default=None)
parser.add_argument('--pre_relu', dest='pre_relu', action='store_true', help='use pre-ReLU quantization')
parser.add_argument('--qtype', default='max_static', help='Type of quantization method')
parser.add_argument('-lp', type=float, help='p parameter of Lp norm', default=3.)
parser.add_argument('--bcorr_w', '-bcw', action='store_true', help='Bias correction for weights', default=False)

parser.add_argument('--gpu_ids', default=[0], type=int, nargs='+',
                    help='GPU ids to use (e.g 0 1 2 3)')

args = parser.parse_args()
DEVICE = torch.device(args.gpu_ids[0])


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


if __name__ == '__main__':
    with MLlogger(os.path.join(home, 'mxt-sim/mllog_runs'), args.experiment, args,
                  name_args=['ssd', args.net, args.dataset,
                             "W{}A{}".format(args.bit_weights, args.bit_act)]) as ml_logger:
        eval_path = pathlib.Path(args.eval_dir)
        eval_path.mkdir(exist_ok=True)
        timer = Timer()
        class_names = [name.strip() for name in open(args.label_file).readlines()]

        # random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        dataset = VOCDataset(args.dataset, is_test=False)

        true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
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

        if args.quantize:
            all_convs = [n for n, m in predictor.net.named_modules() if isinstance(m, nn.Conv2d)][1: -1]
            all_linear = [n for n, m in predictor.net.named_modules() if isinstance(m, nn.Linear)]
            all_relu = [n for n, m in predictor.net.named_modules() if isinstance(m, nn.ReLU)][1: -1]
            all_relu6 = [n for n, m in predictor.net.named_modules() if isinstance(m, nn.ReLU6)][1: -1]
            layers = all_relu + all_relu6 + all_linear + all_convs

            replacement_factory = {nn.ReLU: ActivationModuleWrapperPost,
                                   nn.ReLU6: ActivationModuleWrapperPost,
                                   nn.Linear: ParameterModuleWrapperPost,
                                   nn.Conv2d: ParameterModuleWrapperPost,
                                   nn.Embedding: ActivationModuleWrapperPost}
            mq = ModelQuantizer(predictor.net, args, layers, replacement_factory)


        results = []
        length = (int)(len(dataset) * 0.1)
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
        ml_logger.log_metric('mAp', mAp)
