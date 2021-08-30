import random
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pseudo_labeling(threshold, data_loadr, model):
    idx_list = []
    pseudo_target = []
    pseudo_idx = []
    gt_target = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets, indexs) in enumerate(target_val_loader):
            out_prob = []
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                targets = targets.to(device)
    
            outputs,_ = model(inputs)
            out_prob.append(F.softmax(outputs, dim = 1))
            out_prob = torch.stack(out_prob)
            out_prob = torch.mean(out_prob, dim = 0)

            max_valu, max_idx = torch.max(out_prob,dim =1)

            selected_idx = max_valu >= threshold
  
            pseudo_target.extend(max_idx[selected_idx].cpu().numpy().tolist())
            pseudo_idx.extend(indexs[selected_idx].numpy().tolist())
            gt_target.extend(targets[selected_idx].cpu().numpy().tolist())

    pseudo_target = np.array(pseudo_target)
    gt_target = np.array(gt_target)
    pseudo_idx = np.array(pseudo_idx)

    pseudo_labeling_acc = (pseudo_target == gt_target)*1
    pseudo_labeling_acc = (sum(pseudo_labeling_acc)/len(pseudo_labeling_acc))*100

    print(f'Pseudo-Labeling Accuracy (positive): {pseudo_labeling_acc}, Total Selected: {len(pseudo_idx)}')

    return pseudo_idx