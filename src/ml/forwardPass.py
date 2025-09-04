import torch

def forwardPass(dataloader: torch.utils.data.DataLoader,
                model: torch.nn.Module,
                verbose: bool = True
                ) -> None:
    """ Run a forward pass on a model.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Pytorch dataloader
    model : torch.nn.Module
        Pytorch model.
    verbose : bool, optional
        Print output information, by default True.
    """
    
    # get a sample of the data
    data, label = next(iter(dataloader))
    
    # perform a forward pass on the model
    model.eval()
    with torch.inference_mode():
        sampleLogits = model(data)
        
    # determine probabilities
    sampleProbs = torch.softmax(sampleLogits, dim=1)
    
    # determine prediction class
    dataClass = torch.argmax(sampleProbs, dim=1)
    
    if verbose:
        print(f"Data Shape: {data.shape}")
        print(f"Label: {label}")
        print(f"Output logits: {sampleLogits}")
        print(f"Output prediction probabilities: {sampleProbs}")
        print(f"Output prediction label: {dataClass} (Actual Label: {label})")