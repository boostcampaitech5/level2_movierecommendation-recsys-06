from torch.utils.data import DataLoader


def get_dataloader(dataset, settings: dict):
    """
    gets dataloader using dataset and settings

    Parameters:
        dataset: Contains dataset
        settings: Contains settings
    Returns:
        dataloader(DataLoader): dataloader
    """

    dataloader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=True)

    return dataloader
