# Generative Models Project

This repository contains the data and code for the Generative Models project, part of the second semester of the MVA program.

## Project Overview

The goal of this project is to explore and implement various generative models. These models are designed to generate new data samples that resemble a given dataset.

## Directory Structure

- `data/`: Contains the datasets used for training and evaluating the generative models.
- `src/`: Contains the source code for the project.
- `notebooks/`: Contains Jupyter notebooks with experiments and analysis.
- `results/`: Contains the results of the experiments, including generated samples and evaluation metrics.

## Getting Started

To get started with the project, clone the repository and navigate to the project directory:

```bash
git clone <repository_url>
cd generative-models
```

## Dependencies

Make sure you have the following dependencies installed:

- Python 3.x
- NumPy
- TensorFlow or PyTorch
- Matplotlib
- Jupyter

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

To train a generative model, run the following command:

```bash
python src/train.py --config configs/model_config.yaml
```

To generate new samples, run:

```bash
python src/generate.py --model checkpoints/model_checkpoint.pth
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact [your_email@example.com].
