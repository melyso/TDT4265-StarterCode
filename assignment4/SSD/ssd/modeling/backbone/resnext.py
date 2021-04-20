import torch
from torch import nn
import torchvision.models as models

class ResNext(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        if cfg.MODEL.BACKBONE.RESNEXT_TYPE == 50:
            resnext = models.resnext50_32x4d(pretrained=True)
        elif cfg.MODEL.BACKBONE.RESNEXT_TYPE == 101:
            resnext = models.resnext101_32x8d(pretrained=True)
        else:
            print("Resnext model number not recognized. Using resnext50.")
            resnext = models.resnext50_32x4d(pretrained=True)
        # Freeze first half of network
        for c in list(resnext.children())[:-5]:
            for p in c.parameters():
                p.requires_grad = False
        
        #given  input size 600x600:
        if cfg.INPUT.IMAGE_SIZE == [600, 600]:
            #ouput size 37    
            self.conv1 = nn.Sequential(
                resnext.conv1,
                resnext.bn1,
                resnext.relu,
                resnext.maxpool,
                resnext.layer1,
                resnext.layer2,
                resnext.layer3
            )
            
            #output size 18
            self.conv2 = nn.Sequential(
                resnext.layer4
            )

            #output size 9
            self.conv3 = nn.Sequential(
                nn.BatchNorm2d(self.output_channels[1]),
                nn.ReLU(),
                nn.Conv2d(self.output_channels[1], 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, self.output_channels[2], 3, stride=2, padding=1)
            )
        #given  input size 300x300:
        else:
            #ouput size 37    
            self.conv1 = nn.Sequential(
                resnext.conv1,
                resnext.bn1,
                resnext.relu,
                resnext.maxpool,
                resnext.layer1,
                resnext.layer2
            )
            
            #output size 18
            self.conv2 = nn.Sequential(
                resnext.layer3
            )

            #output size 9
            self.conv3 = nn.Sequential(
                resnext.layer4
            )
        #output res 5x5
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(self.output_channels[2]),
            nn.ReLU(),
            nn.Conv2d(self.output_channels[2], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.output_channels[3], 3, stride=2, padding=1)
        )
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(self.output_channels[3]),
            nn.ReLU(),
            nn.Conv2d(self.output_channels[3], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.output_channels[4], 3, stride=2, padding=1)
        )
        self.conv6 = nn.Sequential(
            nn.BatchNorm2d(self.output_channels[4]),
            nn.ReLU(),
            nn.Conv2d(self.output_channels[4], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.output_channels[5], 3, padding=0)
        )





    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """

        output_0 = self.conv1(x)
        output_1 = self.conv2(output_0)
        output_2 = self.conv3(output_1)
        output_3 = self.conv4(output_2)
        output_4 = self.conv5(output_3)
        output_5 = self.conv6(output_4)
        
        
        out_features = [output_0, output_1, output_2, output_3, output_4, output_5]
        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
