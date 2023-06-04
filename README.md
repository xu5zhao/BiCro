
PyTorch implementation for [BiCro: Noisy Correspondence Rectification for Multi-modality Data via Bi-directional Cross-modal Similarity Consistency](https://arxiv.org/pdf/2303.12419.pdf) (CVPR 2023).

If you have any questions, feel free to contact 20b903054@stu.hit.edu.cn

## Requirements

- Python 3.7
- PyTorch ~1.7.1
- numpy
- scikit-learn
- Punkt Sentence Tokenizer:
  
```
import nltk
nltk.download()
> d punkt
```

## Datasets

### MS-COCO and Flickr30K
We follow [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features and vocabularies.

### CC152K
We use a subset of [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions) (CC), named CC152K. CC152K contains training 150,000 samples from the CC training split, 1,000 validation samples and 1,000 testing samples from the CC validation split. We follow the pre-processing step in [SCAN](https://github.com/kuanghuei/SCAN) to obtain the image features and vocabularies. 

[Download Dataset](https://ncr-paper.cdn.bcebos.com/data/NCR-data.tar)


## Training and Evaluation

### Training new models
Modify some necessary parameters and run it.

For Flickr30K:
```
sh train_f30k.sh
```

For MSCOCO:
```
sh train_coco.sh
```

For CC152K:
```
sh train_cc152k.sh
```

### Pre-trained models and evaluation
The pre-trained models are available here:

F30K 20% noise model [Download](https://1drv.ms/f/s!At35ksCBMmxRjDu0KWs2U_2p0ZQU?e=nhsvk2)

F30K 40% noise model [Download](https://1drv.ms/f/s!At35ksCBMmxRi3HSoV-3qOx2KRZC?e=AA4hKi)

F30K 60% noise model [Download](https://1drv.ms/f/s!At35ksCBMmxRjDyUgPjrIbbwdR5a?e=FnLMIH)


## Citation
If BiCro is useful for your research, please cite the following paper:
```
@inproceedings{BiCro2023,
    author = {Shuo Yang, xu Zhao Pan, Kai Wang, Yang You, Hongxun Yao, Tongliang Liu, Min Xu},
    title = {BiCro: Noisy Correspondence Rectification for Multi-modality Data via Bi-directional Cross-modal Similarity Consistency},
    year = {2023},
    booktitle = {CVPR},
}
```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Acknowledgements
The code is based on [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR), [SGRAF](https://github.com/Paranioar/SGRAF), and [SCAN](https://github.com/kuanghuei/SCAN) licensed under Apache 2.0.
