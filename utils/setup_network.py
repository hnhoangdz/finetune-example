from models.hub import (model)
from utils.checkpoint import restore
from utils.logger import Logger
from utils.helper import init_weights, count_parameters

nets = {
    'model': model.Model
}

def setup_network(network, in_channels, num_classes=7):
    
    print('model: ', network)
    net = nets[network](in_channels=in_channels, num_classes=num_classes)
    if network != 'inception_resnet' or network != 'efficientnet':
        net.apply(init_weights)
    print(f'total trainable parameters: {count_parameters(net)}')

    # Prepare logger
    logger = Logger()

    return logger, net