import torch
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def trainStep(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              lossFn: torch.nn.Module, 
              optimizer: torch.optim.Optimizer,
              device: torch.device
              ) -> Tuple[float, float]:
    """ Training step for a single epoch
    
    Turns a target PyTorch model to training mode and then runs through all of the required training steps (forward 
    pass, loss calculation, optimizer step).

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to be trained.
    dataloader : torch.utils.data.DataLoader
        DataLoader instance for the model to be trained on.
    lossFn : torch.nn.Module
        PyTorch loss function to minimize.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer to help minimize the loss function.
    device : torch.device
        Target device to compute on (e.g. "cuda" or "cpu").

    Returns
    -------
    Tuple[float, float]
        A tuple of training loss and training accuracy metrics.
    """
  
    # put model in train mode
    model.train()

    # setup train loss and train accuracy values
    trainLoss, trainAcc = 0, 0

    # loop through data loader data batches
    for batch, (data, actualClass) in enumerate(dataloader):
        
        # send data to target device
        data, actualClass = data.to(device), actualClass.to(device)

        # run a forward pass on the model
        predLogits = model(data)

        # calculate and accumulate loss
        loss = lossFn(predLogits, actualClass)
        trainLoss += loss.item()

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # calculate and accumulate accuracy metric across all batches
        predClass = torch.argmax(torch.softmax(predLogits, dim=1), dim=1)
        trainAcc += (predClass == actualClass).sum().item() / len(predLogits)

    # adjust metrics to get average loss and accuracy per batch 
    trainLoss /= len(dataloader)
    trainAcc /= len(dataloader)
    
    return trainLoss, trainAcc

def testStep(model: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             lossFn: torch.nn.Module,
             device: torch.device
             ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """ Testing step for a single epoch

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to be trained.
    dataloader : torch.utils.data.DataLoader
        DataLoader instance for the model to be trained on.
    lossFn : torch.nn.Module
        PyTorch loss function to minimize.
    device : torch.device
        Target device to compute on (e.g. "cuda" or "cpu").

    Returns
    -------
    Tuple[float, float]
        A tuple of testing loss and testing accuracy metrics. In the form (testLoss, testAccuracy)
    """
    
    # put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    testLoss, testAcc = 0, 0
    
    # create probabilities array
    testProbs = np.array([])
    testIndxs = np.array([])
    targets = np.array([])

    # turn on inference context manager
    with torch.inference_mode():
        
        # loop through dataloader batches
        for batch, (data, actualClass) in enumerate(dataloader):
            
            # send data to target device
            data, actualClass = data.to(device), actualClass.to(device)

            # forward pass
            predLogits = model(data)
            
            # calculate probabilities
            predProbs = torch.sigmoid(predLogits) # logits -> prediction probabilities
            predProbsMax = predProbs.max(dim=1, keepdim=False)
            testProbs = np.append(testProbs, predProbsMax.values.cpu().numpy())
            testIndxs = np.append(testIndxs, predProbsMax.indices.cpu().numpy())         
            targets = np.append(targets, actualClass.cpu().numpy())

            # calculate and accumulate loss
            loss = lossFn(predLogits, actualClass)
            testLoss += loss.item()

            # calculate and accumulate accuracy
            predClass = predLogits.argmax(dim=1)
            testAcc += ((predClass == actualClass).sum().item() / len(predClass))

    # calculate binary probabilities
    probs = np.abs((1 - testIndxs) - testProbs)
    
    # adjust metrics to get average loss and accuracy per batch 
    testLoss /= len(dataloader)
    testAcc /= len(dataloader)
    
    return testLoss, testAcc, probs, targets

def train(model: torch.nn.Module, 
          trainDL: torch.utils.data.DataLoader, 
          testDL: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          lossFn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          ) -> Dict[str, List]:
    """ Trains and tests a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to be trained.
    trainDL : torch.utils.data.DataLoader
        DataLoader instance for the model to be trained on.
    testDL : torch.utils.data.DataLoader
        DataLoader instance for the model to be tested on.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer to help minimize the loss function.
    lossFn : torch.nn.Module
        PyTorch loss function to minimize.
    epochs : int
        An integer indicating how many epochs to train for.
    device : torch.device
        A target device to compute on (e.g. "cuda" or "cpu").

    Returns
    -------
    Dict[str, List]
        A dictionary of training and testing loss as well as training and testing accuracy metrics. Each metric has a 
        value in a list for each epoch.
            In the form: 
                {trainLoss: [...],
                trainAcc: [...],
                testLoss: [...],
                testAcc: [...]} 
            For example if training for nEpochs = 2: 
                {trainLoss: [2.0616, 1.0537],
                trainAcc: [0.3945, 0.3945],
                testLoss: [1.2641, 1.5706],
                testAcc: [0.3400, 0.2973]}
    """
  
    # create empty results dictionary
    results = {"trainLoss": [],
               "trainAcc": [],
               "testLoss": [],
               "testAcc": []
               }

    # loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs), desc="Processing"):
        
        # run training step
        trainLoss, trainAcc = trainStep(model=model,
                                            dataloader=trainDL,
                                            lossFn=lossFn,
                                            optimizer=optimizer,
                                            device=device)
        
        # run testing step
        testLoss, testAcc, _, _ = testStep(model=model,
                                           dataloader=testDL,
                                           lossFn=lossFn,
                                           device=device)

        # print results
        tqdm.write(f"Epoch {epoch+1} | "
                   f"Train Loss: {trainLoss:.4f} | "
                   f"Test Loss: {testLoss:.4f} | "
                   f"Train Acc: {trainAcc*100:.2f} | "
                   f"Test Acc: {testAcc*100:.2f}")

        # update results
        results["trainLoss"].append(trainLoss)
        results["trainAcc"].append(trainAcc)
        results["testLoss"].append(testLoss)
        results["testAcc"].append(testAcc)

    return results

def eval(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         lossFn: torch.nn.Module,
         device: torch.device
         ) -> Dict[str, np.ndarray]:
    """ Evaluates the model

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to be trained.
    dataloader : torch.utils.data.DataLoader
        DataLoader instance for the model to be evaluated on.
    lossFn : torch.nn.Module
        PyTorch loss function to minimize.
    device : torch.device
        A target device to compute on (e.g. "cuda" or "cpu").

    Returns
    -------
    Dict[str, List]
        A dictionary of probabilities and targets. Each metric has a value in a numpy array for each epoch.
    """
    
    evalLoss, evalAcc, probs, targets = testStep(model=model,
                                                 dataloader=dataloader,
                                                 lossFn=lossFn,
                                                 device=device)

    # print results
    print(
        f"Results | "
        f"Eval Loss: {evalLoss:.4f} | "
        f"Eval Acc: {evalAcc*100:.2f}\n"
    )
    
    results = {"probs": probs,
               "targets": targets
              }

    return results
