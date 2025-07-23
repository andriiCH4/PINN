# PINN
Physics-Informed Neural Network (PINN) in Python

## Methane Combustion Example

The file `methane_combustion_pinn.py` provides a simple PINN implementation
using [DeepXDE](https://github.com/lululxvi/deepxde) to model methane
combustion inside a gas turbine combustion chamber. The model solves a
reaction--diffusion system for methane mass fraction and temperature. Run the
example with:

```bash
python methane_combustion_pinn.py
```

Training logs and checkpoints are saved in the current directory.
