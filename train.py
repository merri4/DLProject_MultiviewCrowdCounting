import glob
import os
import argparse

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment

from model import MultiviewFusionModel, HeadpointCoordDataset

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


def parse_arguments() :

    parser = argparse.ArgumentParser(description="Multiview Crowd Counting")
    
    # environmental settings
    parser.add_argument("--seed", type=int, default=821)
    parser.add_argument("--data_path", type=str, default='./dataset/')
    parser.add_argument("--save_path", type=str, default='./output')
    
    # hyper-parameters
    parser.add_argument("--train_split_rate", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=128)
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
    model = MultiviewFusionModel().to(device)   
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
                num_people = num_people.to(device)
                
                optimizer.zero_grad()
                pred_coords = model(left_img, right_img)
                loss = hungarian_loss(pred_coords, gt_coords, num_people)
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
            num_people = num_people.to(device)

            pred_coords = model(left_img, right_img)
            loss = hungarian_loss(pred_coords, gt_coords, num_people)
            test_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.epochs}]\tTrain Loss : {train_loss/len(train_loader):.4f}\tTest Loss : {test_loss/len(test_loader):.4f}")
        save_checkpoint(model, optimizer, epoch, train_loss, os.path.join(args.save_path, f'epoch_{epoch+1}.pth'))