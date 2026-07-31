# Model weights

`best_model.pth` holds the trained UNet++ / EfficientNet-B3 weights used for every
detection result in the thesis. The optimizer state has been stripped, so the file
carries `model_state_dict` plus the `epoch`, `val_dice` and `loss` records, and is
about a third of the size of the raw training checkpoint.

Load it with `load_trained_model()` in `../code/predict_pipeline.py`, or pass it
directly as `--model-path`. See [`../code/MODEL_USAGE.md`](../code/MODEL_USAGE.md).

To reproduce this file from a full training checkpoint:

```bash
python ../code/strip_checkpoint.py /path/to/full_checkpoint.pth best_model.pth
```
