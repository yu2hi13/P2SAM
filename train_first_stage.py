import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tensorboardX import SummaryWriter
from utils import l2_regularisation, calculate_dice_loss, calculate_sigmoid_focal_loss
from load_LIDC_data import LIDC_IDRI, RandomGenerator
import torch.nn as nn
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
import numpy as np
import os
import logging
import argparse

# Configure logging
logging.basicConfig(filename='training_log_first_stage.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

class MaskWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(7, 1, requires_grad=True) / 8)

def inference(batched_input, lora_sam):
    img_size = 128
    input_images = lora_sam.sam.preprocess(batched_input)
    image_embeddings = lora_sam.sam.image_encoder(input_images)
    sparse_embeddings, dense_embeddings = lora_sam.sam.prompt_encoder(
        points=None, boxes=None, masks=None
    )
    low_res_masks, iou_predictions = lora_sam.sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=lora_sam.sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True
    )
    masks = lora_sam.sam.postprocess_masks(
        low_res_masks,
        input_size=(img_size, img_size),
        original_size=(img_size, img_size)
    )

    return {
        'masks': masks,
        'iou_predictions': iou_predictions,
        'low_res_logits': low_res_masks
    }

def evaluate(model, data_loader, device, mask_weights):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for sampled_batch in data_loader:
            image_batch, label_batch = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
            outputs = inference(image_batch, model)
            output_masks = outputs['masks']
            logits_high = output_masks.to(device)
            weights_eight = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0).to(device)
            logits_high = logits_high * weights_eight.unsqueeze(-1)
            logits_high_res = logits_high.sum(1).unsqueeze(1)
            gt_mask = label_batch.unsqueeze(1)
            dice_loss = calculate_dice_loss(logits_high_res, gt_mask[:].long())
            focal_loss = calculate_sigmoid_focal_loss(logits_high_res, gt_mask[:].float())
            total_loss += (dice_loss + focal_loss).item()
    return total_loss / len(data_loader)

def main(args):
    # Set global device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    os.makedirs("best", exist_ok=True)

    ckpt = args.checkpoint
    img_size = 128
    sam, img_embedding_size = sam_model_registry["vit_b"](
        image_size=img_size,
        num_classes=8,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
        checkpoint=ckpt
    )
    low_res = img_embedding_size * 4

    db = LIDC_IDRI(dataset_location='data/', transform=transforms.Compose([
        RandomGenerator(output_size=[128, 128], low_res=[low_res, low_res], test=True)
    ]))
    dataset_size = len(db)
    indices = list(range(dataset_size))
    train_split = int(np.floor(0.6 * dataset_size))
    validation_split = int(np.floor(0.8 * dataset_size))
    train_indices = indices[:train_split]
    validation_indices = indices[train_split:validation_split]
    test_indices = indices[validation_split:]

    train_dataset = Subset(db, train_indices)
    validation_dataset = Subset(db, validation_indices)
    test_dataset = Subset(db, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Total dataset size: {dataset_size}")
    print(f"Training set size: {len(train_indices)}")
    print(f"Validation set size: {len(validation_indices)}")
    print(f"Test set size: {len(test_indices)}")

    mask_weights = MaskWeights().to(device)
    lora_sam = LoRA_Sam(sam, 4).to(device)

    for param in lora_sam.sam.prompt_encoder.parameters():
        param.requires_grad = True
    for param in lora_sam.sam.mask_decoder.parameters():
        param.requires_grad = True

    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, lora_sam.parameters()), lr=args.lr, weight_decay=0)
    optimizer2 = torch.optim.AdamW(mask_weights.parameters(), lr=args.lr, eps=1e-4)

    writer = SummaryWriter('tf-logs/train_first_stage')
    max_epoch = args.epochs
    best_val_loss = float('inf')
    best_epoch = 0
    start_epoch = 1

    # Check for the latest checkpoint
    latest_checkpoint = f"checkpoint/last_model_epoch_{max_epoch-1}.pth"
    if os.path.exists(latest_checkpoint):
        lora_sam.load_lora_parameters(latest_checkpoint)
        weights_eight = torch.load(f"checkpoint/last_mask_weights_epoch_{max_epoch-1}.pt")
        start_epoch = max_epoch

    try:
        for epoch_num in range(start_epoch, max_epoch + 1):
            lora_sam.train()
            mask_weights.train()
            loss_epoch = 0.0
            print(f"Epoch {epoch_num}")

            for i_batch, sampled_batch in enumerate(train_loader):
                image_batch, label_batch = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
                assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

                outputs = inference(image_batch, lora_sam)
                output_masks = outputs['masks']
                logits_high = output_masks.to(device)
                weights_eight = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0).to(device)
                logits_high = logits_high * weights_eight.unsqueeze(-1)
                logits_high_res = logits_high.sum(1).unsqueeze(1)

                cel = torch.nn.CrossEntropyLoss()
                loss1 = cel(logits_high, label_batch[:].long())
                loss_epoch += loss1.item()

                gt_mask = label_batch.unsqueeze(1)
                dice_loss = calculate_dice_loss(logits_high_res, gt_mask[:].long())
                focal_loss = calculate_sigmoid_focal_loss(logits_high_res, gt_mask[:].float())
                loss2 = dice_loss + focal_loss
                loss = loss1 + loss2

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()

            avg_train_loss = loss_epoch / len(train_loader)
            writer.add_scalar("Train/Loss", avg_train_loss, epoch_num)
            print(f"Average Training Loss: {avg_train_loss}")
            logging.info(f"Epoch {epoch_num}: Average Training Loss: {avg_train_loss}")

            # Validation evaluation
            val_loss = evaluate(lora_sam, validation_loader, device, mask_weights)
            writer.add_scalar("Validation/Loss", val_loss, epoch_num)
            print(f"Validation Loss: {val_loss}")
            logging.info(f"Epoch {epoch_num}: Validation Loss: {val_loss}")

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch_num
                
                # Remove previous best model files
                for file in os.listdir("checkpoint"):
                    if file.startswith("best_mask_weights_epoch_") or file.startswith("best_model_epoch_"):
                        os.remove(os.path.join("checkpoint", file))
                
                # Save new best weights
                torch.save(weights_eight, f"checkpoint/best_mask_weights_epoch_{best_epoch}.pt")
                file_name = f"checkpoint/best_model_epoch_{best_epoch}.pth"
                lora_sam.save_lora_parameters(file_name)

            print(f"Best Validation Loss: {best_val_loss} at epoch {best_epoch}")
            logging.info(f"Best Validation Loss: {best_val_loss} at epoch {best_epoch}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Save the current checkpoint
        torch.save(weights_eight, f"checkpoint/last_mask_weights_epoch_{epoch_num}.pt")
        file_name = f"checkpoint/last_model_epoch_{epoch_num}.pth"
        lora_sam.save_lora_parameters(file_name)
        print(f"Saved checkpoint at epoch {epoch_num}")
        logging.info(f"Saved checkpoint at epoch {epoch_num}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model with specified parameters.')
    parser.add_argument('--checkpoint', type=str, default='sam_vit_b_01ec64.pth', help='Path to the checkpoint file.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=101, help='Number of epochs to train.')
    parser.add_argument('--gpu_id', type=int, default=4, help='GPU ID to use for training.')

    args = parser.parse_args()
    main(args)