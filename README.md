# BachelorCode
Quantization of Image classifiers and Object detectors
Used Distiller Framework
https://github.com/NervanaSystems/distiller
https://nervanasystems.github.io/distiller/

## ResNet

### ResNet_2.ipynb 
Quantization of ResNet18
Tried 8bit, 4bit

### ResNet50.ipynb
Quantization of ResNet18
Tried 8bit, 4bit
Not everything calculated yet...
(maybe deeper network won't lose acc due to quant.)

## SSD

### SSD_3.ipynb
*	latest
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


	
