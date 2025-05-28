import glob
import os
import argparse

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

from model import MultiViewFusionModel, HeadpointCoordDataset

def hungarian_loss(pred_coords, gt_coords, num_people):
    """
    pred_coords, gt_coords: (B, max_people, 2)
    num_people: (B,) 실제 사람 수
    """
    loss = 0.0
    batch_size = pred_coords.shape[0]

    for i in range(batch_size) :
        n = int(num_people[i])
        if n == 0 :
            continue
        pred = pred_coords[i, :n]  # (n, 2)
        gt = gt_coords[i, :n]      # (n, 2)
        cost = torch.cdist(pred, gt, p=2)  # (n, n)  (torch tensor)
        row_ind, col_ind = linear_sum_assignment(cost.cpu().detach().numpy())
        pairwise_loss = cost[row_ind, col_ind].sum() / n  # ← torch tensor!
        loss += pairwise_loss
        
    return loss / batch_size

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    logits: [N,] raw predictions (before sigmoid)
    targets: [N,] binary ground truth (0 or 1)
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probas = torch.sigmoid(logits)
    pt = probas * targets + (1 - probas) * (1 - targets)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


def hungarian_loss_p2pnet(conf_map, offset_map, gt_coords, image_size=(256, 256)):
    """
    Args:
        conf_map: Tensor of shape (B, 1, Hf, Wf) — raw logits
        offset_map: Tensor of shape (B, 2, Hf, Wf)
        gt_coords: Tensor of shape (B, max_people, 2) — normalized coords [0–1]
        image_size: tuple — (H, W) of input image for scaling back to pixels
    Returns:
        total_loss: (float tensor) averaged over batch
    """
    batch_size = conf_map.shape[0]
    Hf, Wf = conf_map.shape[2:]

    total_coord_loss = 0.0
    total_conf_loss = 0.0

    cell_h = image_size[0] / Hf
    cell_w = image_size[1] / Wf

    for b in range(batch_size):
        # [1] Get ground truth coords (remove padding)
        gt = gt_coords[b]
        gt = gt[~(gt == 0).all(dim=1)]  # remove zero rows
        n_gt = gt.shape[0]

        if n_gt == 0:
            # If no ground truth, apply negative loss on all predictions
            conf_b = conf_map[b, 0]  # shape [Hf, Wf]
            total_conf_loss += focal_loss(conf_b, torch.zeros_like(conf_b))
            continue

        # [2] Compute predicted positions and confidences
        y_range = torch.arange(Hf, device=conf_map.device)
        x_range = torch.arange(Wf, device=conf_map.device)
        yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
        # Grid centers
        grid_x = (xx + 0.5) * cell_w
        grid_y = (yy + 0.5) * cell_h

        dx = offset_map[b, 0] * cell_w
        dy = offset_map[b, 1] * cell_h

        pred_x = grid_x + dx
        pred_y = grid_y + dy
        pred_points = torch.stack([pred_x, pred_y], dim=-1).reshape(-1, 2)  # (Hf*Wf, 2)

        # [3] Ground truth points → pixel coordinates
        gt_pixel = gt * torch.tensor(image_size, device=gt.device)

        # [4] Compute cost matrix and Hungarian match
        cost_matrix = torch.cdist(pred_points, gt_pixel, p=1).cpu()
        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

        matched_pred = pred_points[row_ind]  # (n_matched, 2)
        matched_gt = gt_pixel[col_ind]       # (n_matched, 2)

        # [5] Coordinate loss (L1)
        coord_loss = F.l1_loss(matched_pred, matched_gt, reduction='sum') / n_gt
        total_coord_loss += coord_loss

        # [6] Confidence loss
        pred_conf = conf_map[b, 0].reshape(-1)  # (Hf*Wf,)
        target_conf = torch.zeros_like(pred_conf)
        target_conf[row_ind] = 1.0

        conf_loss = focal_loss(pred_conf, target_conf)
        total_conf_loss += conf_loss

    # Average losses over batch
    coord_loss_avg = total_coord_loss / batch_size
    conf_loss_avg = total_conf_loss / batch_size
    total_loss = coord_loss_avg + conf_loss_avg

    return total_loss


def parse_arguments() :

    parser = argparse.ArgumentParser(description="Multiview Crowd Counting")
    
    # environmental settings
    parser.add_argument("--seed", type=int, default=821)
    parser.add_argument("--data_path", type=str, default='./dataset/')
    parser.add_argument("--save_path", type=str, default='./output_p2p')
    
    # hyper-parameters
    parser.add_argument("--train_split_rate", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_people", type=int, default=200)

    args = parser.parse_args()

    return args


def save_checkpoint(model, optimizer, epoch, loss, filename='ckpt.pth') :
    checkpoint = {  
        'epoch': epoch,
        'model_state_dict': model.state_dict(),  
        'optimizer_state_dict': optimizer.state_dict(),  
        'loss': loss,
    }  
    torch.save(checkpoint, filename)  
    print(f"Checkpoint saved to {filename}.")


if __name__ == "__main__" :

    args = parse_arguments()

    if not os.path.exists(args.save_path) :
        os.makedirs(args.save_path)

    # =========================================================
    # Dataset Prep
    # =========================================================
    
    left_paths = sorted(glob.glob(os.path.join(args.data_path, 'screenshot/left_*.png')))
    right_paths = sorted(glob.glob(os.path.join(args.data_path, 'screenshot/right_*.png')))
    csv_paths = sorted(glob.glob(os.path.join(args.data_path, 'headpoint_csv/headpoint_*.csv')))

    print(f"Left: {len(left_paths)}, Right: {len(right_paths)}, CSV: {len(csv_paths)}")

    # 전체 80% 학습, 20% 테스트
    dataset = HeadpointCoordDataset(left_paths, right_paths, csv_paths, max_people=args.max_people, W=256, H=256)
    
    n_total = len(dataset)
    n_train = int(args.train_split_rate * n_total)
    n_test = n_total - n_train

    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Dataset length : {}".format(len(dataset)))

    # =========================================================
    # Model Prep
    # =========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device : ".format(device))
    model = MultiViewFusionModel().to(device)   
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # =========================================================
    # Model Train
    # =========================================================

    print("Start Training...")
    for epoch in range(args.epochs) :
        
        train_loss = 0
        test_loss = 0

        # train loss and backprop
        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar :
            for left_img, right_img, gt_coords, num_people in pbar:

                left_img = left_img.to(device)
                right_img = right_img.to(device)
                gt_coords = gt_coords.to(device)
                # num_people = num_people.to(device)
                
                optimizer.zero_grad()
                conf_map, offset_map = model(left_img, right_img)
                loss = hungarian_loss_p2pnet(conf_map, offset_map, gt_coords)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        # validation loss
        model.eval()
        for left_img, right_img, gt_coords, num_people in tqdm(test_loader, desc="Validation : ") :
            
            left_img = left_img.to(device)
            right_img = right_img.to(device)
            gt_coords = gt_coords.to(device)
            # num_people = num_people.to(device)

            conf_map, offset_map = model(left_img, right_img)
            loss = hungarian_loss_p2pnet(conf_map, offset_map, gt_coords)
            test_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.epochs}]\tTrain Loss : {train_loss/len(train_loader):.4f}\tTest Loss : {test_loss/len(test_loader):.4f}")
        save_checkpoint(model, optimizer, epoch, train_loss, os.path.join(args.save_path, f'epoch_{epoch+1}.pth'))