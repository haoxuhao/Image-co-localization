import torch.nn as nn
import torch 

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG19(nn.Module):

    def __init__(self, out_features=True, num_classes=1000, init_weights=True):
        super(VGG19, self).__init__()
        self.features = make_layers(cfgs['E'])

        # self.stage1 = self.features[:4]
        # self.stage2 = self.features[5:8]
        # self.stage3 = self.features[9:14]
        # self.stage4 = self.features[15:20]
        # self.stage5 = self.features[21:26]

        # self.stage2 = make_layers([128, 128], in_channels=64)
        # self.stage3 = make_layers([256, 256, 256, 256], in_channels=128)
        # self.stage4 = make_layers([512, 512, 512, 512], in_channels=256)
        # self.stage5 = make_layers([512, 512, 512, 512], in_channels=512)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.feature_dim = 512

        self.out_features = out_features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        #x = self.features(x)
        out1 = self.pool_2(self.features[:4](x))
        out2 = self.pool_2(self.features[5:9](out1))
        out3 = self.features[10:18](out2)
        out3 = self.pool_2(out3)

        out4 = self.features[19:27](out3)
        out4_add = out4
        out4 = self.pool_2(out4)
        out5 = self.features[28:36](out4)
        #print(self.features[28:36])
        
        if self.out_features:
            #ret = torch.cat((self.upsample(out5), out4_add),1)
            #ret = self.upsample(out5)
            ret = out5
            return ret
            #return self.upsample(out5)+out4_add
        else:
            out5 = self.pool2(out5)
            x = self.avgpool(out5)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    #in_channels = 3

    #this two param control the dilate rate and poolling rate
    idx=0
    dia=1
    
    for idx, v in enumerate(cfg):
        if idx==25:
            layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            continue
        if idx>25:
            dia = 2

        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, dilation=dia)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

if __name__ == "__main__":
    model = VGG19(out_features=True)
    model.load_state_dict(torch.load("/root/.cache/torch/checkpoints/vgg19-dcbb9e9d.pth"))
    input_data = torch.rand(2, 3, 224, 224)
    output = model(input_data)
    print(output.shape)