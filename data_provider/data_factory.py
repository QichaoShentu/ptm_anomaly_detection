from torch.utils.data import ConcatDataset, DataLoader
from .data_loader_onlyTrain import ASDSegLoader, MSLSegLoader, PSMSegLoader, SKABSegLoader, SWATSegLoader
from .data_loader import SMDSegLoader, SMAPSegLoader, GECCOSegLoader
from .batch_scheduler import BatchSchedulerSampler

train_data_dict = {
    'ASD': ASDSegLoader,
    'MSL': MSLSegLoader,
    'PSM': PSMSegLoader,
    'SKAB': SKABSegLoader,
    'SWAT': SWATSegLoader,
}

data_dict = {
    'SMD': SMDSegLoader,
    'SMAP': SMAPSegLoader,
    'NIPS_TS_GECCO': GECCOSegLoader,
}

def train_data_provider(args):
    '''shuffle=False, 考虑重写DataLoader的sampler

    Args:
        args (_type_): args.train_datasets, args.root_path, args.batch_size, args.win_size, args.num_workers

    Returns:
        concat_dataset: 多个数据集合并后的数据集
        dataloader: 
    '''
    concat_dataset = []
    # config
    batch_size = args.batch_size
    drop_last = False
    dataset_list = args.train_datasets.split(',')
    for dataset_name in dataset_list:
        factory = train_data_dict[dataset_name]
        dataset = factory(
            root_path=args.root_path,
            win_size=args.win_size,
        )
        print(f'{dataset_name} len: ', len(dataset))
        concat_dataset.append(dataset)

    concat_dataset = ConcatDataset(concat_dataset)

    data_loader = DataLoader(
        dataset=concat_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=drop_last,
        sampler=BatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size))
    
    return concat_dataset, data_loader

def data_provider(args, flag, finetune=False):
    '''

    Args:
        args (_type_): args.data, args.root_path, args.batch_size, args.win_size, args.num_workers
    '''
    factory = data_dict[args.data]
    if flag == 'test':
        shuffle_flag = False
    else:
        shuffle_flag = True
    batch_size = args.batch_size 
    drop_last = False

    data_set = factory(
        root_path=args.root_path,
        win_size=args.win_size,
        flag=flag,
        finetune=finetune
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_datasets', type=str, default='ASD,MSL,PSM,SKAB,SWAT', help='') # ASD,MSL,PSM,SKAB,SWAT
    parser.add_argument('--data', type=str, default='SMD', help='') # SMD,SMAP,NIPS_TS_GECCO
    parser.add_argument('--root_path', type=str, default='/workspace/ptm_anomaly_detection/dataset', help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--win_size', type=int, default=100, help='')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    args = parser.parse_args()
    print(args)

    # print('N datasets')
    # train_datasets, train_dataloader = train_data_provider(args)
    # print(len(train_datasets))
    # print(len(train_dataloader))
    
    # for i, (data, labels) in enumerate(train_dataloader):
    #     print(i)
    #     print(data.shape)
    #     print(labels.shape)
    #     if i==4: break

    print('\n(n+1)th dataset')
    dataset, dataloader = data_provider(args, 'train', finetune=True)
    for i, (data, labels) in enumerate(dataloader):
        print(data.shape)
        print(labels.shape)
        break



