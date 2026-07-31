import torch
from tqdm import tqdm
from model import build_model, FocalTverskyLoss, dice_coef, iou_score

def train_model(train_loader, val_loader, device, save_path="best_model.pth", epochs=20):
    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004)
    loss_fn = FocalTverskyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_val_dice = 0
    patience = 5
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': [], 'train_iou': [], 'val_iou': []}

    for epoch in range(epochs):
        model.train()
        train_loss, train_dice, train_iou = 0, 0, 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_dice += dice_coef(outputs, masks).item()
            train_iou += iou_score(outputs, masks).item()

        model.eval()
        val_loss, val_dice, val_iou = 0, 0, 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += loss_fn(outputs, masks).item()
                val_dice += dice_coef(outputs, masks).item()
                val_iou += iou_score(outputs, masks).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_dice = train_dice / len(train_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_val_iou = val_iou / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Dice: {avg_train_dice:.4f} | Val Dice: {avg_val_dice:.4f} | Train IoU: {avg_train_iou:.4f} | Val IoU: {avg_val_iou:.4f}")
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': avg_val_dice,
                'val_loss': avg_val_loss
            }, save_path)
            print("✅ Saved new best model with optimizer and metrics")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement for {epochs_no_improve} epochs")
        if epochs_no_improve >= patience:
            print("⛔ Early stopping triggered")
            break
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_dice'].append(avg_train_dice)
        history['val_dice'].append(avg_val_dice)
        history['train_iou'].append(avg_train_iou)
        history['val_iou'].append(avg_val_iou)
    return history
