# Training and experiment notebooks

The notebooks that produced the segmentation model, kept for the record of the
empirical search reported in Chapter 3 of the thesis (dataset versions, encoder
and loss comparison, augmentation policy).

**Cell outputs have been cleared.** The originals embed rendered figures and run
logs and total roughly 480 MB, which is impractical to distribute; all reported
metrics live in the thesis and in `Results_Analysis/`. The code is unchanged.

## Which notebook is which

| Notebook | Role |
|---|---|
| `Almound_Trees_Seg_09_08.ipynb` | **Production run.** Roboflow dataset version 8, UNet++ / EfficientNet-B3, Focal Tversky loss. This is the model used throughout the thesis. |
| `Almond_Trees_Seg_Unet_1024_DiceLoss*.ipynb` | Earlier iterations at 1024 px on dataset versions 3 to 6, Dice loss |
| `Almond_Trees_Seg_Unet_1536_Improved*.ipynb`, `*_2048_*.ipynb` | Input-resolution trials (1536 and 2048 px) |
| `Almond_Trees_Seg_Unet_Attention-2048.ipynb` | Attention-augmented U-Net variant |
| `Almond_Trees_Seg_Unet_deeper_DiceLoss.ipynb` | Deeper decoder variant |
| `augmentation_comparison_unet_EfficientNetB3.ipynb` | Augmentation-policy comparison behind Table 4 |
| `loading_model_and_post_proccesing.ipynb` | Inference and watershed post-processing |
| `CoCo_Unet_first_test.ipynb` | Initial COCO-format segmentation test |
| `FlightLogMatcher.ipynb` | Matches drone flight logs to images for GPS recovery (2021 survey) |

## Running them

The dataset is pulled from Roboflow. Supply the key through the environment;
no credentials are stored in this repository.

```bash
export ROBOFLOW_API_KEY="your-key-here"
```

Dependencies: `torch`, `segmentation-models-pytorch`, `albumentations`,
`pycocotools`, `opencv-python`, `roboflow`.
