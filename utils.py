from torch.utils.data import DataLoader
from torch import nn, sqrt


def compute_channel_stats(dataset, batch_size=32):
    """
    Compute the channel-wise mean and standard deviation of all images in a dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): A dataset object that returns (image, label) pairs.
        batch_size (int): Batch size for processing the dataset.
    
    Returns:
        mean (tensor): A tensor containing the channel-wise mean.
        std (tensor): A tensor containing the channel-wise standard deviation.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    mean_sum = 0.0
    std_sum = 0.0
    total_images = 0

    for images, _ in dataloader:
        images = images.view(images.size(0), images.size(1), -1)
        
        total_images += images.size(0)
        
        mean_sum += images.mean(dim=2).sum(dim=0)
        std_sum += images.var(dim=2, unbiased=False).sum(dim=0)
    
    mean = mean_sum / total_images
    std = sqrt(std_sum / total_images)
    
    return mean, std

def get_dim_before_first_linear(layers, in_dim, in_channels, brain=False):
    """
    Assume square in dimensions, square kernels, cuz I'm lazy
    Also assume kernel numbers and channels match up, because that's trivial enough
    """

    current_dim = in_dim
    current_channels = in_channels
    for layer in layers:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
            # If the layer padding is same we do not need to change the dimension of the input...
            if layer.padding == 'same':
                if isinstance(layer, nn.Conv2d):
                    current_channels = layer.out_channels
                continue
            vals = {
                'kernel_size': layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0],
                'stride': layer.stride if isinstance(layer.stride, int) else layer.stride[0],
                'padding': layer.padding if isinstance(layer.padding, int) else layer.padding[0],
                'dilation': layer.dilation if isinstance(layer.dilation, int) else layer.dilation[0]
            }
            current_dim = (current_dim + 2*vals['padding'] - vals['dilation']*(vals['kernel_size'])) // vals['stride'] + 1
        if isinstance(layer, nn.Conv2d):
            current_channels = layer.out_channels
    
        if isinstance(layer, nn.Linear):
            if brain:
                return current_dim, current_channels
            else:
                return current_dim * current_dim * current_channels
    if brain:
        return current_dim, current_channels
    else:
        return current_dim * current_dim * current_channels

    # raise ValueError("No linear layer found in layers! Why are you even asking me?")