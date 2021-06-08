# import library


sys.path.append('..')


architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name]))


# define parser
parser = argparse.ArgumentParser(description='pretrain Teacher net')

# dataset & domain
dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )
parser.add_argument('--dataset', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
parser.add_argument('--source', default = 'A', help='source domain(s)')
#parser.add_argument('--target', default = 'W', help='target domain(s)')

# network
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')

# various path
parser.add_argument('--save_root', type=str, default='./results/Teacher', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets/DA', help='path name of image dataset')

args, unparsed = parser.parse_known_args()

# main code
def main():

if __name__ == '__main__':
	main()
