import torch
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(str(device) == 'cuda')):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes=2, ignore_index=-100):
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    total_intersection = torch.zeros(num_classes, device=device)
    total_union = torch.zeros(num_classes, device=device)
    total_target = torch.zeros(num_classes, device=device)
    total_pred = torch.zeros(num_classes, device=device)

    pbar = tqdm(dataloader, desc="Evaluating")
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(str(device) == 'cuda')):
            logits = model(images)
            loss = criterion(logits, masks)
        running_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        valid = (masks != ignore_index)  # esclude i pixel "no-data"
        correct_pixels += ((preds == masks) & valid).sum().item()
        total_pixels += valid.sum().item()

        for c in range(num_classes):
            if c == ignore_index:
                continue
            pred_c = (preds == c) & valid
            mask_c = (masks == c) & valid
            total_intersection[c] += (pred_c & mask_c).sum()
            total_union[c] += (pred_c | mask_c).sum()
            total_target[c] += mask_c.sum()
            total_pred[c] += pred_c.sum()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct_pixels / max(total_pixels, 1)

    iou_per_class = total_intersection / (total_union + 1e-6)
    dice_per_class = (2 * total_intersection) / (total_pred + total_target + 1e-6)

    valid_classes = total_union > 0  # la classe ignorata ha union=0 -> automaticamente esclusa
    mIoU = iou_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
    mdice = dice_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0

    return avg_loss, accuracy, mIoU, mdice, iou_per_class.cpu().numpy()
