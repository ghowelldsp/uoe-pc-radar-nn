import torch
import torch.nn as nn
import numpy as np

def loadModel(modelType: str, 
              device: torch.device,
              signalLen: int,
              nOutFeatures: int,
              verboseOutput: bool
              ) -> nn.Module:
    """ Load an instance of a model

    Parameters
    ----------
    modelType : str
        Model to load
    device : torch.device
        Device the model is trained on.
    signalLen : int
        Length of the input signal.
    nOutFeatures: int
        Number of output features.
    verboseOutput : bool
        Verbose output information.

    Returns
    -------
    nn.Module
        Instance of the model.

    Raises
    ------
    ValueError
        If the model type is not recognised.
    """
    
    # create an instance of the model
    match modelType:
        case "testing":
            model = modelTesting(nInFeatures=signalLen,
                                 nOutFeatures=nOutFeatures)
        case "basic0":
            model = modelBasic0(nInFeatures=signalLen,
                                nOutFeatures=nOutFeatures,
                                apKernelSize=32,
                                apStride=1,
                                apPadding=0)
        case "basic1":
            model = modelBasic1(nInFeatures=signalLen,
                                nOutFeatures=nOutFeatures,
                                apKernelSize=16,
                                apStride=8,
                                apPadding=0)
        case "twoThirdsRule":
            model = modelTwoThirdsRule(nInFeatures=signalLen,
                                       nOutFeatures=nOutFeatures,
                                       apKernelSize=16,
                                       apStride=8,
                                       apPadding=0)
        case _:
            raise ValueError(f"model {modelType} not recognised")
        
    # send model to device
    model = model.to(device)
    
    if verboseOutput:
        print("\nMODELS\n")
        print(f"Input Features (Signal Length): {signalLen}")
        print(f"Model: {modelType}")
        print(f"Device: {device}")
        # TODO - fix this
        # print("Model Summary:")
        # from torchinfo import summary
        # summary(model, input_size=[2, signalLen])
        
    return model

def calcApOutLen(inLen: int,
                 kernelSize: int,
                 stride: int,
                 padding: int,
                 ) -> int:
    """ Calculate average pooling output length

    Parameters
    ----------
    inLen : int
        Length of input tensor to the average pooling layer.
    kernelSize : int
        Kernel size.
    stride : int
        Stride length.
    padding : int
        Padding length.

    Returns
    -------
    int
        Length of output tensor after average pooling.
    """
    
    return int((inLen + 2 * padding - kernelSize) / stride + 1)

class modelTesting(nn.Module):
    """ Testing model
    
    Basic model testing used for testing purposes.
    
    """ 
    
    def __init__(self,
                 nInFeatures: int,
                 nOutFeatures: int
                 ) -> None:
    
        # initialise nn.Module class
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Linear(in_features=nInFeatures, out_features=10),
            nn.Linear(in_features=10, out_features=nOutFeatures)
            )
        
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        
        x = self.block_1(x)
        
        return x
    
class modelBasic0(nn.Module):
    """ Basic model 0
    """ 
    
    def __init__(self,
                 nInFeatures: int,
                 nOutFeatures: int,
                 apKernelSize: int,
                 apStride: int,
                 apPadding: int,
                 ) -> None:
    
        # initialise nn.Module class
        super().__init__()
        
        # calculate the length of the output of the average pooling layer
        apOutLen = calcApOutLen(inLen=nInFeatures, # type: ignore
                                kernelSize=apKernelSize,
                                stride=apStride,
                                padding=apPadding
                                )
    
        self.block_1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=apKernelSize, 
                         stride=apStride,
                         padding=apPadding),
            nn.Linear(in_features=apOutLen, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=nOutFeatures)
        )
        
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        
        x = self.block_1(x)
        
        return x
    
class modelBasic1(nn.Module):
    """ Basic model 1
    
    Basic model uses in the dissertation.
    """ 
    
    def __init__(self,
                 nInFeatures: int,
                 nOutFeatures: int,
                 apKernelSize: int,
                 apStride: int,
                 apPadding: int,
                 ) -> None:
    
        # initialise nn.Module class
        super().__init__()
        
        # calculate the length of the output of the average pooling layer
        apOutLen = calcApOutLen(inLen=nInFeatures, # type: ignore
                                kernelSize=apKernelSize,
                                stride=apStride,
                                padding=apPadding
                                )
    
        self.block_1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=apKernelSize, 
                         stride=apStride,
                         padding=apPadding),
            nn.Linear(in_features=apOutLen, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=nOutFeatures)
        )
        
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        
        x = self.block_1(x)
        
        return x

class modelTwoThirdsRule(nn.Module):
    """ Two third rule model
    
    This model works by creating a linear and relu layer where the input features of each consecutive layer reduces 
    by two thirds each time. 
    """ 
    
    def __init__(self,
                 nInFeatures: int,
                 nOutFeatures: int,
                 apKernelSize: int,
                 apStride: int,
                 apPadding: int,
                 ) -> None:
    
        # initialise nn.Module class
        super().__init__()
        
        # calculate the length of the output of the average pooling layer
        apOutLen = calcApOutLen(inLen=nInFeatures, # type: ignore
                                kernelSize=apKernelSize,
                                stride=apStride,
                                padding=apPadding
                                )
        
        # create of all layers starting with the average pooling layer
        allLayers = [nn.AvgPool1d(kernel_size=apKernelSize, 
                                  stride=apStride,
                                  padding=apPadding)]
        
        # calculate the final number of features, given that the last layer is simply a linear layer
        finalValue = int(np.ceil(nOutFeatures * 3/2))
        
        # create all the layers using the 2/3 rule
        rlInFeatures = apOutLen
        while True:
            # output of repeated layer is 2/3 of input
            rlOutFeatures = int(np.floor(rlInFeatures * 2/3))
            if rlOutFeatures <= finalValue:
                break
            # create the repeated layer
            repeatedLayer = [nn.Linear(in_features=rlInFeatures, out_features=rlOutFeatures),
                             nn.ReLU()]
            # concatonate the repeated layer with the rest of the layers
            allLayers += repeatedLayer
            # update the repeated layers input value
            rlInFeatures = rlOutFeatures
        
        # add final layer
        allLayers += [nn.Linear(in_features=rlInFeatures, out_features=nOutFeatures)]
        
        # create the full sequential block
        self.block_1 = nn.Sequential(*allLayers)
        
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        
        x = self.block_1(x)
        
        return x  