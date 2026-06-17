## Step-by-step transfer learning

1. **Load dataset:**
   - Load dataset with name `Splitted dataset`. Refer to Miro for dataset, with the name dataset RSHS.

2. **Train model**
   - The chexnet model is in `model.pth.tar`
   - Continue training as usual 

## Transfer learning pipeline

The transfer learning implementation is kept in [`transfer_learning/README.md`](transfer_learning/README.md). It adds a separate CheXNet/DenseNet121 multi-label training, evaluation, threshold tuning, and Grad-CAM workflow without changing the existing `inference.py` path.
