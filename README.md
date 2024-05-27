# MoRA: Matrix of Rank Adaptation for Parameter-Efficient Fine-Tuning

MoRA is a parameter-efficient fine-tuning technique for large language models (LLMs) that utilizes a high-rank matrix adaptation approach. It enables effective learning and memorization of new knowledge during fine-tuning while maintaining a small number of trainable parameters.

## Features

- Parameter efficiency: MoRA achieves parameter efficiency by using a square matrix with a high rank instead of low-rank matrices used in other methods like LoRA (Low-Rank Adaptation).
- Improved performance: The implementation includes various enhancements and best engineering practices that improve the performance and reliability of MoRA.
- Easy integration: The code provides utility functions like `apply_mora` and `apply_mora_linear` that make it easy to integrate MoRA into existing models.
- Flexibility: MoRA offers flexibility in configuring the rank of the adaptation matrix, the group type for compression and decompression, and the merging and resetting steps.

## Installation

To use MoRA in your project, simply copy the `mora.py` file into your project directory and import the necessary classes and functions.

```python
from mora import MoRA, apply_mora, MoRALinear, apply_mora_linear, merge_and_reset_mora_linear, update_model_mora_linear
```

## Usage

### Applying MoRA to a Model

To apply MoRA to a model, use the `apply_mora` function:

```python
model = YourModel()
hidden_size = 768
rank = 128
merge_steps = 1000

update_model = apply_mora(model, hidden_size, rank, merge_steps)
```

The `apply_mora` function replaces the linear layers in the model with a sequential combination of `MoRA` and the original linear layer.

### Training with MoRA

During training, call the `update_model` function returned by `apply_mora` at each step:

```python
for step in range(num_steps):
    # Training code...
    update_model(step)
```

The `update_model` function periodically merges the square matrix of each `MoRA` instance into the corresponding linear layer's weights and resets the square matrix.

### Using MoRALinear

Alternatively, you can use the `MoRALinear` class as a drop-in replacement for `nn.Linear`:

```python
model = YourModel()
rank = 128
group_type = 0

apply_mora_linear(model, rank, group_type)
```

The `apply_mora_linear` function replaces the linear layers in the model with `MoRALinear` layers.

During training, use the `update_model_mora_linear` function to create an update function:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
merge_steps = 1000

update_model = update_model_mora_linear(model, optimizer, merge_steps)

for step in range(num_steps):
    # Training code...
    update_model(step)
```

The `update_model` function returned by `update_model_mora_linear` periodically merges the square matrix of each `MoRALinear` instance into the linear layer's weights and resets the square matrix.

## Examples

Example code and usage can be found in the `examples` directory.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgements

MoRA is based on the research paper "MoRA: High-Rank Matrix Adaptation for Parameter-Efficient Fine-Tuning of Large Language Models"

## Contact

For questions or inquiries, please contact [San] at [sanowl98@mail.com].
