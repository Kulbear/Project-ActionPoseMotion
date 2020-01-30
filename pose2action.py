from torch import nn
from core.data.data_utils import ntu_fetch
import os
import torch
from core.data.generators import PoseGenerator
from core.data.ntu_dataset import NTUDataset
from torch.utils.data import DataLoader
from pose2motion_arguments import parse_args
from core.models.pose_action_model import Net, Model
from pose2action_utils import train, evaluate


def main(config):
    dataset = NTUDataset('/data/ntu_rgbd_60/nturgb+d_skeletons')
    print('==> Initializing dataloaders...')
    train_data = PoseGenerator(*ntu_fetch(['P001', 'P002', 'P003', 'P004', 'P005'], dataset), dataset_name='ntu')
    train_loader = DataLoader(train_data,
                              batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    data = ntu_fetch(['P004', 'P005'], dataset)
    test_data = PoseGenerator(*data, dataset_name='ntu')
    test_loader = DataLoader(test_data,
                             batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    print('==> Dataloaders initialized ...')

    graph_args = {
        'layout': 'ntu-rgb+d',
        'strategy': 'spatial'
    }
    model = Model(3, 60, graph_args, edge_importance_weighting=True).to(config.device)
    criterion = nn.MSELoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(config.start_epoch, 1000):
        print('\nEpoch: %d' % (epoch + 1))
        # Train for one epoch
        epoch_loss = train(train_loader, model, criterion, optimizer, config.device)
        print(epoch_loss)

        a = evaluate(test_loader, model, config.device)
        print('Accuracy is %8f' % a)

    """
    model = Net().to(config.device)
    criterion = nn.MSELoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(config.start_epoch, 1000):
        print('\nEpoch: %d' % (epoch + 1))
        # Train for one epoch
        epoch_loss = train(train_loader, model, criterion, optimizer, config.device)
        print(epoch_loss)

        a = evaluate(test_loader, model, config.device)
        print('Accuracy is %8f' % a)
    """

if __name__ == "__main__":
    config = parse_args()

    # os setting
    os.environ['OMP_NUM_THREAD'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.visible_devices

    # a simple work-around for the dataloader multithread bug
    torch.set_num_threads(1)
    main(config)
