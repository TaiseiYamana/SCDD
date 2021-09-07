import numpy as np

np.random.seed(args.seed)
def split_dataset(dataset, split_pro = 0.8):
  targets = []
  for _, target in dataset.samples:
    targets.append(target)
  
  num_per_class = np.zeros(dataset.num_classes)
  for i in targets:
    num_per_class[i] = num_per_class[i] + 1

  split_per_class = np.zeros(dataset.num_classes)
  for i in range(dataset.num_classes):
    split_per_class[i] = np.round(num_per_class[i] * split_pro)
  
  targets =  np.array(targets)

  split_idx = []
  for i in range(dataset.num_classes):
    idx = np.where(targets == i)[0]

    np.random.shuffle(idx)
    split_idx.extend(idx[:int(split_per_class[i])])
  
  return split_idx