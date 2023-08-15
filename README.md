# Simple module to use imagenette in pytorch

This repository contains a python file that makes it easy to download the imagenette dataset and allows you to create the respective pytorch datasets for training and validation.

### Example for imagenette2-160

```
from pytorch_imagenette import download_imagenette, Imagenette

download_imagenette("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", local_path="./")

# Path where the CSV file containing information about the labels and images is.
annotations_file = "./imagenette2-160/noisy_imagenette.csv"

# Path where the images dataset is hosted
img_dir = "./imagenette2-160"
```

```
trainset = Imagenette(annotations_file, img_dir, train=True)
testset = Imagenette(annotations_file, img_dir, train=False)
```