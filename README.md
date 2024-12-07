# P2SAM: Probabilistically Prompted SAMs for Ambiguous Medical Images

This repository contains the implementation of **P2SAM**, a novel framework for ambiguous medical image segmentation. P2SAM leverages the prior knowledge of the Segment Anything Model (SAM) to enhance segmentation precision and diversity with minimal annotated data. The framework addresses the challenges of ambiguity and limited annotations in medical imaging.

---

## Features

- **Probabilistic Prompt Generation**: Generates prompt distributions to guide SAM in ambiguous segmentation tasks.
- **Diversity-Aware Assembling**: Aggregates diverse segmentation masks with learnable weights.
- **Efficient Training**: Achieves high performance with significantly reduced training data requirements.
- **State-of-the-Art Performance**: Outperforms baseline methods on metrics such as GED, HM-IoU, and Dmax.

---

## Repository Structure

```
.
├── environment.yml           # Conda environment file for dependencies
├── evaluate.py               # Evaluation script for segmentation results
├── load_LIDC_data.py         # Data loader for the LIDC-IDRI dataset
├── p2sam.py                  # Implementation of the P2SAM framework
├── sam_lora_image_encoder.py # Enhanced SAM image encoder using LoRA
├── train_first_stage.py      # Training script for the first stage (fine-tuning SAM)
├── train_second_stage.py     # Training script for the second stage (probabilistic prompts)
├── utils.py                  # Utility functions for data preprocessing and model operations
```

---

## Installation

### Prerequisites

- Python >= 3.7
- Conda (recommended for environment setup)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yu2hi13/p2sam.git
   cd p2sam
   ```

2. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate p2sam
   ```

3. Download the `sam_vit_b` weights and place them in the current directory:
   - [Download sam_vit_b weights](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

---

## Usage

### Data Preparation

- Download the required datasets:
  - [LIDC-IDRI](https://drive.google.com/drive/folders/1xKfKCQo8qa6SAr3u7qWNtQjIphIrvmd5)
- Place the downloaded datasets in the `/data` directory.
- Ensure the datasets are preprocessed to the format required by `load_LIDC_data.py`.

### Training

1. Train the first stage (fine-tuning SAM):
   ```bash
   python train_first_stage.py
   ```

2. Train the second stage (probabilistic prompt generation):
   ```bash
   python train_second_stage.py
   ```

### Evaluation

Evaluate the trained model on the test dataset:
```bash
python evaluate.py 
```

---

## Results

P2SAM demonstrates superior performance in ambiguous medical image segmentation compared to state-of-the-art methods, achieving:

- Higher accuracy in segmentation
- Enhanced diversity in predictions
- Robust performance with limited annotated data

Detailed experimental results can be found in the [paper](https://doi.org/10.1145/3664647.3680628).

---

## Citation

If you use this repository, please cite:

```
@inproceedings{huang2024p2sam,
  title={P2SAM: Probabilistically Prompted SAMs Are Efficient Segmentator for Ambiguous Medical Images},
  author={Huang, Yuzhi and Li, Chenxin and others},
  booktitle={ACM MM '24},
  year={2024},
  doi={10.1145/3664647.3680628}
}
```

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.# P2SAM
