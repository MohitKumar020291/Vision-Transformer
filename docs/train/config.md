# Configuration File Specification

This document outlines the structure and usage of the YAML configuration file used for defining model parameters, training behavior, and output paths. The configuration file provides a centralized and flexible way to control experimental settings without modifying source code.

---

## File Format

The configuration must be written in [YAML](https://yaml.org/spec/) format. It is divided into the following top-level sections:

* [`model`](#model)
* [`training`](#training)
* [`paths`](#paths)

Each section contains relevant hyperparameters and directives used by the training pipeline.

---

## Section: `model`

Defines the model architecture and input specifications.

```yaml
model:
  type: ViT               # Model architecture to use. Currently supports 'ViT' (Vision Transformer).
  image_size: 32          # Input image resolution (assumes square images).
  patch_size: 4           # Size of the image patches used in patch embedding.
  num_classes: 10         # Number of output classes for classification.
```

**Notes:**

* `type`: This must correspond to a model implementation available in the codebase.
* Additional model types may be added in future releases.

---

## Section: `training`

Specifies the training procedure and optimization strategy.

```yaml
training:
  epochs: 1                     # Total number of training epochs.
  batch_size: 32                # Number of samples per training batch.
  learning_rate: 0.001          # Initial learning rate.
  optimizer: adam               # Optimization algorithm. E.g., 'adam', 'sgd'.
  scheduler: sequential-lr      # Learning rate scheduler to use. E.g., 'step-lr', 'cosine', 'sequential-lr'.
  criterion: cross-entropy      # Loss function. E.g., 'cross-entropy', 'mse'.
  use_cuda: True                # Whether to enable GPU acceleration if available.
```

**Guidelines:**

* Ensure that the optimizer, scheduler, and criterion specified are supported by the training script or framework.
* `use_cuda` must be set to `True` only if CUDA-compatible hardware is available.

---

## Section: `paths`

Defines directories used for storing experiment outputs.

```yaml
paths:
  save_dir: experiments/runs    # Directory where model checkpoints, logs, and other artifacts are saved.
```

**Best Practices:**

* Use a dedicated output directory for each experiment to avoid overwriting previous runs.
* Relative paths are supported, but ensure that the parent directories exist or are created by the training script.

---

## Complete Example

```yaml
model:
  type: ViT
  image_size: 32
  patch_size: 4
  num_classes: 10

training:
  epochs: 1
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  scheduler: sequential-lr
  criterion: cross-entropy
  use_cuda: True

paths:
  save_dir: experiments/runs
```

---

## Usage

To run training using a configuration file:

```bash
python train.py --config config.yaml
```

Ensure that the training script includes logic to parse and apply the configuration file. This typically involves loading the YAML file into a dictionary and passing it to the model and training components.

---

## Validation

Prior to execution, confirm the following:

* The YAML file is syntactically valid.
* All fields are correctly spelled and properly indented.
* The values match the accepted types and constraints of the implementation.

You may use tools such as [`yamllint`](https://github.com/adrienverge/yamllint) or online YAML validators to check formatting.

---

## Extensibility

This configuration format is designed to be modular and extensible. Additional sections (e.g., for data loading, logging, or augmentation) can be introduced as needed, provided they are properly handled by the consuming code.

For example:

```yaml
data:
  dataset: CIFAR10
  augmentations:
    - random_crop
    - horizontal_flip
```
