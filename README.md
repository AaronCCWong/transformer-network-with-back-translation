# Transformer Network Trained With Back-Translation

Assumes python3 is being used.

### To train:

```bash
python train.py [PARAMS]
```

### To evaluate:

```bash
python evaluate.py --model [PATH TO MODEL TO EVALUATE]
```

### To translate:

Some configurations to `translate.py` are required to translate custom sentences.
`translate.py` is currently set up to spit out DE->EN translations of the IWSLT test split as provided by TorchText.

### To preprocess:

`preprocess.py` sets up the translations from `translate.py` to be importable by TorchText.

### Model Weights:

Download the weights from here

https://drive.google.com/file/d/1AGG5sdyEPr9uCSv-1KXnFkIk-qYnlClW/view?usp=sharing

and extract them into `models/` directory.
