import torch
import argparse
import os

# Custom modules
from data import COCOSegmentationDataset, get_train_transform, get_val_test_transform
from train import train_model
from predict_pipeline import predict_folder

# Fix for Intel MKL duplicate lib error on some Macs/Environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    parser = argparse.ArgumentParser(description="Train or predict almond tree segmentation")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True, help="Run mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--model-path", type=str, default="best_model.pth", help="Path to model checkpoint")
    
    # Train arguments
    parser.add_argument("--train-img-dir", type=str, help="Training images directory")
    parser.add_argument("--train-ann-path", type=str, help="Training COCO annotation JSON")
    parser.add_argument("--val-img-dir", type=str, help="Validation images directory")
    parser.add_argument("--val-ann-path", type=str, help="Validation COCO annotation JSON")
    
    # Predict arguments
    parser.add_argument("--predict-dir", type=str, help="Input images directory for prediction")
    parser.add_argument("--output-dir", type=str, help="Output directory for prediction results")
    parser.add_argument("--thresh", type=float, default=0.5, help="Prediction threshold")
    
    args = parser.parse_args()
    
    # Handle MPS (Mac Metal Performance Shaders) if available
    if args.device == 'cuda' and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
            
    device = torch.device(args.device)
    print(f"Active Device: {device}")

    if args.mode == "train":
        if not all([args.train_img_dir, args.train_ann_path, args.val_img_dir, args.val_ann_path]):
            raise ValueError("Training mode requires --train-img-dir, --train-ann-path, --val-img-dir, and --val-ann-path")
        
        train_dataset = COCOSegmentationDataset(args.train_img_dir, args.train_ann_path, get_train_transform())
        val_dataset = COCOSegmentationDataset(args.val_img_dir, args.val_ann_path, get_val_test_transform())
        
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
        
        train_model(train_loader, val_loader, device, save_path=args.model_path, epochs=20)

    elif args.mode == "predict":
        if not all([args.predict_dir, args.output_dir]):
            raise ValueError("Predict mode requires --predict-dir and --output-dir")
        
        predict_folder(
            folder=args.predict_dir,
            model_path=args.model_path,
            device=device,
            output_root=args.output_dir,
            transform=get_val_test_transform(),
            thresh=args.thresh
        )

if __name__ == "__main__":
    main()