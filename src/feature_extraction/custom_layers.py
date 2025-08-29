import torch.nn as nn


class CustomFeatureExtractor(nn.Module):
    def __init__(self, layer_config):
        super().__init__()
        layers = []

        for layer_info in layer_config:
            layer_type = layer_info['type']

            if layer_type == "conv2d":
                layers.append(nn.Conv2d(
                    in_channels=layer_info['in_channels'],
                    out_channels=layer_info['out_channels'],
                    kernel_size=layer_info['kernel_size'],
                    padding=layer_info.get('padding', 0)
                ))
            elif layer_type == "relu":
                layers.append(nn.ReLU())
            elif layer_type == "maxpool2d":
                layers.append(nn.MaxPool2d(
                    kernel_size=layer_info['kernel_size']
                ))
            elif layer_type == "adaptive_avg_pool2d":
                layers.append(nn.AdaptiveAvgPool2d(
                    output_size=layer_info['output_size']
                ))
            elif layer_type == "flatten":
                layers.append(nn.Flatten())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)