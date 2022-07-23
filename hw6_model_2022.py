'''
Skeleton model class. You will have to implement the classification and regression layers, along with the forward definition.
'''


from torch import nn
from torchvision import models


class RCNN(nn.Module):
    def __init__(self, size):
        super(RCNN, self).__init__()

        # Pretrained backbone. If you are on the cci machine then this will not be able to automatically download
        #  the pretrained weights. You will have to download them locally then copy them over.
        #  During the local download it should tell you where torch is downloading the weights to, then copy them to 
        #  ~/.cache/torch/checkpoints/ on the supercomputer.
        resnet = models.resnet18(pretrained=True)

        # Remove the last fc layer of the pretrained network.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze backbone weights. 
        for param in self.backbone.parameters():
            param.requires_grad = False

        # TODO: Implement the fully connected layers for classification and regression.
        """
        For RCNN, features increase as depth increases. (Larger area has more features)
        What exactly this means not sure yet.
        At least 2 layers. Use RelU and Max pooling at each layer 
        """
        num_ftrs = resnet.fc.in_features
        self.label = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, size+1),
            nn.Softmax(dim=1)
        )
        self.bounds = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 4*size)
        )

    def forward(self, x):
        # TODO: Implement forward. Should return a (batch_size x num_classes) tensor for classification
        #           and a (batch_size x num_classes x 4) tensor for the bounding box regression. 
        labels = self.label(self.backbone(x))
        bounds = self.bounds(self.backbone(x))
        return labels, bounds
