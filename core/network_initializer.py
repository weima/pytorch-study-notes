"""
Initialize and Reshape the Networks
Now to the most interesting part. Here is where we handle the reshaping of
each network. Note, this is not an automatic procedure and is unique to
each model. Recall, the final layer of a CNN model, which is often times
an FC layer, has the same number of nodes as the number of output classes
in the dataset. Since all of the models have been pre-trained on Imagenet,
they all have output layers of size 1000, one node for each class.
The goal here is to reshape the last layer to have the same number
of inputs as before, AND to have the same number of outputs as the number
of classes in the dataset. In the following sections we will discuss
how to alter the architecture of each model individually.
But first, there is one important detail regarding the difference between
fine-tuning and feature-extraction.
When feature extracting, we only want to update the parameters of the
last layer, or in other words, we only want to update the parameters
for the layer(s) we are reshaping. Therefore, we do not need to compute the
gradients of the parameters that we are not changing, so for efficiency we
set the .requires_grad attribute to False. This is important because by default,
this attribute is set to True. Then, when we initialize the new layer and
by default the new parameters have .requires_grad=True so only the new layer’s
parameters will be updated. When we are fine-tuning we can leave all of the
'.required_grad'  set to the default of True.

Finally, notice that inception_v3 requires the input size to be (299,299),
whereas all of the other models expect (224,224).
Resnet
Resnet was introduced in the paper Deep Residual Learning for Image Recognition.
There are several variants of different sizes, including Resnet18, Resnet34,
Resnet50, Resnet101, and Resnet152, all of which are available from
torchvision models. Here we use Resnet18, as our dataset is small and only has
two classes. When we print the model, we see that the last layer is a fully
connected layer as shown below:
  (fc): Linear(in_features=512, out_features=1000, bias=True)

Thus, we must reinitialize model.fc to be a Linear layer with 512 input features
and 2 output features with:
  model.fc = nn.Linear(512, num_classes)



Alexnet
Alexnet was introduced in the paper ImageNet Classification with Deep
Convolutional Neural Networks and was the first very successful CNN on the
ImageNet dataset. When we print the model architecture, we see the model output
comes from the 6th layer of the classifier
(classifier): Sequential(
    ...
    (6): Linear(in_features=4096, out_features=1000, bias=True)
 )

To use the model with our dataset we reinitialize this layer as
   model.classifier[6] = nn.Linear(4096,num_classes)


VGG
VGG was introduced in the paper Very Deep Convolutional Networks for
Large-Scale Image Recognition. Torchvision offers eight versions of VGG with
various lengths and some that have batch normalizations layers. Here we use
VGG-11 with batch normalization. The output layer is similar to Alexnet, i.e.

(classifier): Sequential(
    ...
    (6): Linear(in_features=4096, out_features=1000, bias=True)
 )

Therefore, we use the same technique to modify the output layer
   model.classifier[6] = nn.Linear(4096,num_classes)



Squeezenet
The Squeezenet architecture is described in the paper SqueezeNet: AlexNet-level
accuracy with 50x fewer parameters and <0.5MB model size and uses a different
output structure than any of the other models shown here. Torchvision has
two versions of Squeezenet, we use version 1.0. The output comes from a 1x1
convolutional layer which is the 1st layer of the classifier:

(classifier): Sequential(
    (0): Dropout(p=0.5)
    (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
    (2): ReLU(inplace)
    (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
 )

To modify the network, we reinitialize the Conv2d layer to have an output
feature map of depth 2 as
   model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))


Densenet
Densenet was introduced in the paper Densely Connected Convolutional Networks.
Torchvision has four variants of Densenet but here we only use Densenet-121.
The output layer is a linear layer with 1024 input features:
(classifier): Linear(in_features=1024, out_features=1000, bias=True)

To reshape the network, we reinitialize the classifier’s linear layer as
   model.classifier = nn.Linear(1024, num_classes)


Inception v3
Finally, Inception v3 was first described in Rethinking the Inception
Architecture for Computer Vision. This network is unique because it has
two output layers when training. The second output is known as an auxiliary
output and is contained in the AuxLogits part of the network. The primary
output is a linear layer at the end of the network. Note, when testing we only
consider the primary output. The auxiliary output and primary output of the
loaded model are printed as:

(AuxLogits): InceptionAux(
    ...
    (fc): Linear(in_features=768, out_features=1000, bias=True)
 )
 ...
(fc): Linear(in_features=2048, out_features=1000, bias=True)

To fine-tune this model we must reshape both layers. This is accomplished with
the following
   model.AuxLogits.fc = nn.Linear(768, num_classes)
   model.fc = nn.Linear(2048, num_classes)

Notice, many of the models have similar output structures, but each must
be handled slightly differently. Also, check out the printed model
architecture of the reshaped network and make sure the number of output
features is the same as the number of classes in the dataset.
"""
from __future__ import print_function

from typing import Optional

from core.imgnet_models.alexnet_model import AlexnetModel
from core.imgnet_models.densenet_model import DensenetModel
from core.imgnet_models.inception_model import InceptionModel
from core.imgnet_models.resnet_model import ResnetModel
from core.imgnet_models.squeezenet_model import SqueezenetModel
from core.imgnet_models.vgg_model import VggModel
from core.types import ModelName

model_map = {
    ModelName.Resnet: ResnetModel(),
    ModelName.Alexnet: AlexnetModel(),
    ModelName.Vgg: VggModel(),
    ModelName.Squeezenet: SqueezenetModel(),
    ModelName.Inception: InceptionModel(),
    ModelName.Densenet: DensenetModel(),
}


def initialize_model(
        model_name: ModelName,
        num_classes: int,
        feature_extract: bool,
        device,
        use_pretrained: Optional[bool] = True
):
    try:
        model = model_map[model_name]
        model.init(
            device=device,
            num_classes=num_classes,
            feature_extract=feature_extract,
            use_pretrained=use_pretrained
        )
        return model
    except KeyError:
        raise ValueError(f"Not supported model name {model_name}")
