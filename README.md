# BachelorCode
## Quantization of Image classifiers and Object detectors

Resources:

https://github.com/NervanaSystems/distiller
https://nervanasystems.github.io/distiller/
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
https://github.com/amdegroot/ssd.pytorch
https://github.com/qfgaohao/pytorch-ssd


## ResNet
*   https://nervanasystems.github.io/distiller/
*   PyTorch impl.

### [ResNet_2.ipynb](./resnet/ResNet_2.ipynb)
Quantization of ResNet18
*   Tried 8bit, 4bit

### [ResNet50.ipynb](./resnet/ResNet50.ipynb)
Quantization of ResNet50
*   Tried 8bit, 4bit
*   Not everything calculated yet...
*   (maybe deeper network won't lose acc due to quant.)

## [SSD Nvidia impl.](./DeepLearningExamples/PyTorch/Detection/SSD/SSD_NVIDIA.ipynb)
*	latest
*   fixed SSD_3 issues
*   base AP 13 %
*   4 bit quant AP 4 %
*   from: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD

### [SSD_3](./SSD/SSD_3.ipynb)
*	used model with PyTorch Checkopint https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD#model-architecture
*		modified for quantization purposes
*	problems with softmax (?), src/evaluate.py and src/utils.py 
```
            # mask = score > 0.05 original
            mask = score > 0.02

            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0: continue

            score_sorted, score_idx_sorted = score.sort(dim=0)
```
*	problems with masking probabilities in tensors (maybe needs full prec. layer)
*	if ``mask = score > 0.02``, then has **some** predictions, if ``mask = score > 0.05``, then all probabalities are False


SSD.ipynb
*	doesnt work

SSD_2.ipynb
*	based on https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/ tutorial


	
