import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from p2sam import P2SAM
from load_LIDC_data import LIDC_IDRI, RandomGenerator
from utils import l2_regularisation, calculate_dice_loss, calculate_sigmoid_focal_loss
from tensorboardX import SummaryWriter
import argparse
import logging

# Configure logging
logging.basicConfig(filename='training_log_second_stage.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def evaluate(model, data_loader, device, weight_eight):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for sampled_batch in data_loader:
            image_batch, label_batch = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
            image_batch_oc = sampled_batch['image_oc'].to(device)
            outputs = model.forward(image_batch, image_batch_oc)
            output_masks = outputs['masks']
            logits_high = output_masks * weight_eight.unsqueeze(-1)
            logits_high_res = logits_high.sum(1).unsqueeze(1)
            gt_mask = label_batch.unsqueeze(1)
            dice_loss = calculate_dice_loss(logits_high_res, gt_mask[:].long())
            focal_loss = calculate_sigmoid_focal_loss(logits_high_res, gt_mask[:].float())
            total_loss += (dice_loss + focal_loss).item()
    return total_loss / len(data_loader)

def save_non_lora_weights(model, epoch, combined_weights):
    non_lora_state_dict = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith('lora')}
    combined_weights[f'epoch_{epoch}'] = non_lora_state_dict
    print(f"Non-LoRA weights for epoch {epoch} added to combined weights.")

def main(args):
    # Set global device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Initialize model
    print(f"Loading best model from {args.model_path}")
    net = P2SAM(device=device, lora_ckpt=args.model_path).to(device)
    for param in net.lora_sam.parameters():
        param.requires_grad = False

    # Load weights
    print(f"Loading best weights from {args.weight_path}")
    weight_eight = torch.load(args.weight_path).to(device)
    low_res = net.img_embedding_size * 4

    # Prepare dataset
    db = LIDC_IDRI(dataset_location='data/', transform=transforms.Compose([
        RandomGenerator(output_size=[128, 128], low_res=[low_res, low_res], test=True)
    ]))
    dataset_size = len(db)
    indices = list(range(dataset_size))
    train_split = int(np.floor(0.6 * dataset_size))
    validation_split = int(np.floor(0.8 * dataset_size))
    train_indices = indices[:500]
    validation_indices = indices[train_split:validation_split]
    test_indices = indices[validation_split:]

    train_dataset = Subset(db, train_indices)
    validation_dataset = Subset(db, validation_indices)
    test_dataset = Subset(db, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=0)
    writer = SummaryWriter('tf-logs/train_second_stage')
    max_epoch = args.epochs
    start_epoch = 1

    # Dictionary to store all saved weights
    combined_weights = {}

    try:
        for epoch_num in range(start_epoch, max_epoch + 1):
            net.train()
            loss_epoch = 0.0
            reg_loss_epoch = 0.0
            print(f"Epoch {epoch_num}")

            for i_batch, sampled_batch in enumerate(train_loader):
                image_batch, label_batch = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
                image_batch_oc = sampled_batch['image_oc'].to(device)
                assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

                # Forward pass
                outputs = net.forward(image_batch, image_batch_oc)
                output_masks = outputs['masks']
                logits_high = output_masks * weight_eight.unsqueeze(-1)
                logits_high_res = logits_high.sum(1).unsqueeze(1)

                # Calculate loss
                cel = torch.nn.CrossEntropyLoss()
                cel_loss = cel(logits_high, label_batch[:].long())
                reg_loss = l2_regularisation(net.prior_dense) + l2_regularisation(net.fcomb.layers)
                gt_mask = label_batch.unsqueeze(1)
                dice_loss = calculate_dice_loss(logits_high_res, gt_mask[:].long())
                focal_loss = calculate_sigmoid_focal_loss(logits_high_res, gt_mask[:].float())
                loss = cel_loss + args.reg_weight * reg_loss + dice_loss + focal_loss
                loss_epoch += loss.item()
                reg_loss_epoch += reg_loss.item()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = loss_epoch / len(train_loader)
            writer.add_scalar("Train/Loss", avg_train_loss, epoch_num)
            print(f"Average Training Loss: {avg_train_loss}")
            logging.info(f"Epoch {epoch_num}: Average Training Loss: {avg_train_loss}")

            # Save model every 10 epochs
            if epoch_num % 10 == 0:
                save_non_lora_weights(net, epoch_num, combined_weights)
                print(f"Checkpoint saved at epoch {epoch_num}")
                logging.info(f"Checkpoint saved at epoch {epoch_num}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Save weights corresponding to epochs
        torch.save(combined_weights, args.save_path)
        print(f"Epoch weights saved to {args.save_path}.")
        logging.info(f"Epoch weights saved to {args.save_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the second stage model with specified parameters.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--gpu_id', type=int, default=2, help='GPU ID to use for training.')
    parser.add_argument('--model_path', type=str, default='checkpoint/last_model_epoch_101.pth', help='Path to the best model file.')
    parser.add_argument('--weight_path', type=str, default='checkpoint/last_mask_weights_epoch_101.pt', help='Path to the best weight file.')
    parser.add_argument('--save_path', type=str, default='checkpoint/final_weights.pth', help='Path to save the final weights.')
    parser.add_argument('--reg_weight', type=float, default=1e-5, help='Regularization weight.')

    args = parser.parse_args()
    main(args)