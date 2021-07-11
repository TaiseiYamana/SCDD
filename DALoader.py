import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import common.vision.datasets as datasets

__all__ = ['Office31_loader','OfficeCaltech_loader','OfficeHome_loader','VisDA2017_loader','DomainNet_loader']

def Office31_loader(root, task, batch_size, num_workers, train:bool):

    # Computed with compute_mean_std.py
    mean_std = {
        'A': {
            'mean': [0.7497086 , 0.74227816, 0.7399158 ],
            'std':  [0.3219877 , 0.3242278 , 0.32624903]
        },
        'D': {
            'mean': [0.4530981 , 0.43283814, 0.39659035],
            'std':  [0.21102072, 0.19876367, 0.20394132]
        },
        'W': {
            'mean': [0.5944873, 0.603156, 0.6096597],
            'std':  [0.2514419, 0.25771737, 0.2617499]
        }
    }
    if train == True:
      data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[task]['mean'],
                                 std=mean_std[task]['std'])
        ])
      dataset = datasets.office31.Office31(root=root, task=task, download=True, transform=data_transform)
      dataset_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers, drop_last=True)
    else:
      data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[task]['mean'],
                                 std=mean_std[task]['std'])
        ])
      dataset = datasets.office31.Office31(root=root, task=task, download=True, transform=data_transform)
      dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataset,dataset_loader

def OfficeCaltech_loader(root, task, batch_size, num_workers, train:bool):
    # Computed with compute_mean_std.py
    mean_std = {
        'A': {
            'mean': [0.7289755 , 0.7262561 , 0.72922254],
            'std':  [0.33858246, 0.33773202, 0.33638695]
        },
        'D': {
            'mean': [0.43920696, 0.4179544 , 0.36827588],
            'std':  [0.21082632, 0.19928727, 0.20038427]
        },
        'W': {
            'mean': [0.58090585, 0.59661317, 0.60144824],
            'std':  [0.2669448 , 0.2722725 , 0.27037242]
        },
        'C': {
            'mean': [0.5866699, 0.5762263, 0.5780217],
            'std':  [0.32587516, 0.3239491 , 0.32507262]
            }
    }
    if train == True:
      data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[task]['mean'],
                                 std=mean_std[task]['std'])
        ])
      dataset = datasets.officecaltech.OfficeCaltech(root=root, task=task, download=True, transform=data_transform)
      dataset_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers, drop_last=True)
    else:
      data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[task]['mean'],
                                 std=mean_std[task]['std'])
        ])
      dataset = datasets.office31.OfficeCaltech(root=root, task=task, download=True, transform=data_transform)
      dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataset,dataset_loader


def OfficeHome_loader(root, task, batch_size, num_workers, train:bool):

    # Computed with compute_mean_std.py
    mean_std = {
        'Ar': {
            'mean': [0.52283245, 0.48524195, 0.4446748 ],
            'std':  [0.31315085, 0.30861387, 0.3110282 ]
        },
        'Cl': {
            'mean': [0.5977058 , 0.5730141 , 0.54277223],
            'std':  [0.39597276, 0.391109  , 0.40348607]
        },
        'Pr': {
            'mean': [0.68958294, 0.67496663, 0.66532934],
            'std':  [0.3365528 , 0.3380124 , 0.34408984]
        },
        'Rw': {
            'mean': [0.59091556, 0.5561959 , 0.52175593],
            'std':  [0.3059498 , 0.3053527 , 0.31786376]
        }
    }

    if train == True:
      data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[task]['mean'],
                                 std=mean_std[task]['std'])
        ])
      dataset = datasets.officecaltech.OfficeCaltech(root=root, task=task, download=True, transform=data_transform)
      dataset_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers, drop_last=True)
    else:
      data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[task]['mean'],
                                 std=mean_std[task]['std'])
        ])
      dataset = datasets.officehome.OfficeHome(root=root, task=task, download=True, transform=data_transform)
      dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataset,dataset_loader

def VisDA2017_loader(root, task, batch_size, num_workers, train:bool):

    # Computed with compute_mean_std.py
    mean_std = {
        'T': {
            'mean': [0.7984475 , 0.79600173, 0.7906679 ],
            'std':  [0.2224036 , 0.22520256, 0.23131843]
        },
        'V': {
            'mean': [0.40844184, 0.3882431 , 0.36937693],
            'std':  [0.27102378, 0.26400164, 0.26523244]
        }
    }

    if train == True:
      data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[task]['mean'],
                                 std=mean_std[task]['std'])
        ])
      dataset = datasets.officechome.OfficeCaltech(root=root, task=task, download=True, transform=data_transform)
      dataset_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers, drop_last=True)
    else:
      data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[task]['mean'],
                                 std=mean_std[task]['std'])
        ])
      dataset = datasets.officehome.OfficeHome(root=root, task=task, download=True, transform=data_transform)
      dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataset,dataset_loader

def DomainNet_loader(root, task, batch_size, num_workers, train:bool):

    # Computed with compute_mean_std.py
    mean_std = {
        'c': {
            'mean': [0.71909213, 0.6920118 , 0.6511557 ],
            'std':  [0.3496526 , 0.3510136 , 0.37529978]
        },
        'i': {
            'mean': [0.6880168 , 0.6934753 , 0.65844494],
            'std':  [0.31298026, 0.28956342, 0.31055406]
        },
        'p': {
            'mean': [0.5743671 , 0.5400756 , 0.49719965],
            'std':  [0.28724572, 0.27813205, 0.29283443]
        },
          'q': {
            'mean': [0.942863, 0.942863, 0.942863],
            'std':  [0.21338843, 0.21338843, 0.21338843]
        },
          'r': {
            'mean': [0.597666 , 0.57264  , 0.5351411],
            'std':  [0.3016261 , 0.29861847, 0.3155561 ]
        },
          's': {
            'mean': [0.81809413, 0.81204367, 0.8027624 ],
            'std':  [0.25711504, 0.2593651 , 0.2650984 ]
        }
    }



    dataset = datasets.domainnet.DomainNet(root=root, task=task, evaluate = evaluate, download=True, transform=data_transform)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train)
    return dataset,dataset_loader
