import argparse
import logging
import numpy as np
from pathlib import Path
from os.path import exists
import os
import glob
import torch
import pandas as pd
import transformers
from transformers import BertModel, BertTokenizer
import math
from tqdm import tqdm
import pickle as pkl
from datasets import load_dataset, Dataset
import csv

DATASETS = {
    "rte": "yangwang825/rte",
    "sst2":"gpt3mix/sst2",
}

NUM_CLASSES = {
    "rte": 2,
    "sst2": 2,
}

MODELS = {
    "BERT": {
        "tiny": "prajjwal1/bert-tiny",
        "mini": "prajjwal1/bert-mini"
    }
}

SIZES = set({})
for k in MODELS.keys():
    SIZES.update((MODELS[k].keys()))

MAX_LENGTH = 512

parser = argparse.ArgumentParser()

parser.add_argument(
    "-info",
    action="store_true",
    help="Boolean flag to enable info mode"
)

parser.add_argument(
    "-log",
    "--logFile",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-model",
    choices=list(MODELS.keys()),
    help="Name of the HF model to use",
    default="BERT"
)

parser.add_argument(
    "-size",
    choices=SIZES,
    help="Size of the HF model to use",
    required=True
)

parser.add_argument(
    "-dataset",
    choices=list(DATASETS.keys()),
    help="Name of HF dataset to use",
    required=True
)

parser.add_argument(
    "-test",
    type=str,
    help="Path to file containing test instances; Must be set when -testOnly is set."
)

parser.add_argument(
    "-out",
    help="Path to directory where models should be saved after training",
    default="./models/"
)

parser.add_argument(
    "-save",
    type=str,
    help="Path to directory where outputs are to be saved after testing",
    default="./outputs/"
)

parser.add_argument(
    "-load",
    type=str,
    help="[Optional] Path to saved PyTorch model to load"
)

parser.add_argument(
    "-seed",
    type=int,
    help="Seed for torch/numpy",
    default=13
)

parser.add_argument(
    "-numEpochs",
    type=int,
    help="Number of epochs to train model for",
    default=10
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size of dataloader",
    default=64
)

parser.add_argument(
    "-learningRate",
    type=float,
    nargs="+",
    help="Learning rate(s) for optimizer",
    default=[0.0001, 0.00001, 0.000001]
)

parser.add_argument(
    "-weightDecay",
    type=float,
    help="Weight Decay for optimizer",
    default=0
)

parser.add_argument(
    "-maxSteps",
    type=int,
    help="Maximum number of optimization steps allowed",
    default=-1
)

parser.add_argument(
    "-optimizer",
    choices=["adam", "adagrad"],
    help="Choice of optimizer to use for training",
    default="adagrad"
)

parser.add_argument(
    "-amsgrad",
    action="store_true",
    help="Boolean flag to enable amsgrad in optimizer"
)

parser.add_argument(
    "-lrScheduler",
    action="store_true",
    help="Boolean flag to employ learning rate scheduler during training"
)

parser.add_argument(
    "-testOnly",
    action="store_true",
    help="Boolean flag to only perform testing; Model load path must be provided when this is set to true"
)

parser.add_argument(
    "-cacheDir",
    type=str,
    help="Path to cache directory",
    default="/scratch/general/vast/u1419542/huggingface_cache"
)

parser.add_argument(
    "-skipFinetuning",
    action="store_true",
    help="When set, skips finetuning BERT; For use when -testOnly is not set.",
)
#---------------------------------------------------------------------------
def checkIfExists(path, isDir=False, createIfNotExists=False): 
    if isDir and not path.endswith("/"):
        path += "/"
    pathExists = exists(path)
    if not pathExists:
        if createIfNotExists:
            os.makedirs(path) 
        else:
            raise ValueError(f"{path} is an invalid path!")
    if not isDir:
        filePath = Path(path)
        if not filePath.is_file():
            raise ValueError(f"{path} is not a file!")   
    return path
#---------------------------------------------------------------------------
def checkFile(fileName, fileExtension=None):
    if fileExtension:
        if not fileName.endswith(fileExtension):
            raise ValueError(f"[checkFile] {fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        raise RuntimeError(f"[checkFile] {fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        raise RuntimeError(f"[checkFile] {fileName} is not a file!")
#---------------------------------------------------------------------------
def readFile(fileName):
    data=None
    if fileName.endswith(".csv"):
        with open(fileName, "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    else: 
        raise RuntimeError("Unsupported file: {}".format(fileName))
    return data
#---------------------------------------------------------------------------
class BERTforClassification(torch.nn.Module):
    def __init__(self, modelPath, numClasses, cacheDir="/scratch/general/vast/u1419542/huggingface_cache"):
        super(BERTforClassification, self).__init__()
        self.modelPath = modelPath
        self.numClasses = numClasses
        self.model = BertModel.from_pretrained(modelPath, cache_dir=cacheDir)
        self.classifier = torch.nn.Linear(
            in_features=self.model.config.hidden_size,
            out_features=numClasses,
            bias=True
        )

    def freezeBERT(self):
        for param in self.model.parameters():
            param.requires_grad = False 
    
    def forward(self, tokenizedInputs):
        return self.classifier(self.model(**tokenizedInputs)["pooler_output"])
#---------------------------------------------------------------------------
class CustomDataset():
    def __init__(self, data, task):
        self.data = data
        self.task = task

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        if self.task in "sst2":
            input = self.data[item]["text"]
            target = None 
            if "label" in self.data[item].keys():
                target = torch.tensor(self.data[item]["label"])
            return input, target
        elif self.task in "rte":
            input1, input2 = self.data[item]["text1"], self.data[item]["text2"]
            target = None 
            if "label" in self.data[item].keys():
                target = torch.tensor(self.data[item]["label"])
            return input1, input2, target
        else: 
            raise ValueError("Unrecognized task: {}".format(self.task))
#---------------------------------------------------------------------------
def tokenizeInstances(inputs, modelPath):
    tokenizer = BertTokenizer.from_pretrained(modelPath)
    tokenizedInput = tokenizer(*inputs, truncation=True, padding=True, return_tensors="pt", max_length=MAX_LENGTH)
    return tokenizedInput
#---------------------------------------------------------------------------
class collateBatch:
    def __init__(self, task, modelPath):
        self.task = task 
        self.modelPath = modelPath
    
    def __call__(self, batch):
        if self.task in "sst2":
            inputs, labels = zip(*batch)
            inputs = tokenizeInstances((inputs,), self.modelPath)
            if labels[0] != None:
                labels = torch.stack(labels)
            return inputs, labels
        elif self.task in "rte":
            input1s, input2s, labels = zip(*batch)
            inputs = tokenizeInstances((input1s, input2s), self.modelPath)
            if labels[0] != None:
                labels = torch.stack(labels)
            return inputs, labels
        else: 
            raise ValueError("Unrecognized task: {}".format(self.task))
#---------------------------------------------------------------------------
def createDataLoader(dataset, task, modelPath, batchSize, shuffle=True):
    ds = CustomDataset(
        data=dataset,
        task=task,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batchSize,
        num_workers=0,
        shuffle=shuffle,
        collate_fn=collateBatch(task, modelPath)
    )
#---------------------------------------------------------------------------
def trainModel(model, dataLoader, lossFunction, optimizer, device="cpu", scheduler=None, maxSteps=-1, logSteps=1000, dataDesc="Train data"):
    model.to(device)
    model.train()

    losses = []
    corrPreds = 0
    numExamples = 0
    numBatch = 0
    numSteps = 0
    for inputs, targets in tqdm(dataLoader, desc=dataDesc):
        inputs = inputs.to(device)
        targets = targets.to(device)

        numBatch += 1
        numExamples += len(targets)
        outputs = model(inputs)
        
        _, preds = torch.max(outputs, dim=-1)

        loss = lossFunction(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))

        if numSteps%logSteps == 0:
            logging.info(f"\nBatch: {numBatch}/{len(dataLoader)}, Loss: {loss.item()}")

        corrPreds += torch.sum(preds.reshape(-1) == targets.reshape(-1))
        losses.append(loss.item())
        #Zero out gradients from previous batches
        optimizer.zero_grad()
        #Backwardpropagate the losses
        loss.backward()
        # #Avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #Perform a step of optimization
        optimizer.step()
        numSteps += 1
        if maxSteps and numSteps >= maxSteps:
            break
    if scheduler:
        scheduler.step()
    return corrPreds.double()/numExamples, np.mean(losses)
#---------------------------------------------------------------------------
def evalModel(model, lossFunction, dataLoader, device="cpu", dataDesc="Test batch"):
    model.to(device)
    model.eval()
    with torch.no_grad():
        losses = []
        corrPreds = 0
        numExamples = 0
        numBatch = 0
        numSteps = 0
        perplexity = 0
        for inputs, targets in tqdm(dataLoader, desc=dataDesc):
            inputs = inputs.to(device)
            targets = targets.to(device)

            numBatch += 1
            numExamples += len(targets)
            outputs = model(inputs)

            _, preds = torch.max(outputs, dim=-1)

            loss = lossFunction(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))

            corrPreds += torch.sum(preds.reshape(-1) == targets.reshape(-1))
            losses.append(loss.mean().item())
            numSteps += 1
    return corrPreds.double()/numExamples, np.mean(losses)
#---------------------------------------------------------------------------
def testModel(model, dataLoader, device="cpu", dataDesc="Test data"):
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = []
        for inputs, _ in tqdm(dataLoader, desc=dataDesc):
            inputs = inputs.to(device)

            outputs = model(inputs)

            predictions.extend(outputs.cpu())
    predictions = torch.stack(predictions)
    return predictions
#---------------------------------------------------------------------------
def main(errTrace="main"):
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.logFile:
        checkFile(args.logFile)
        logging.basicConfig(filename=args.logFile, filemode='w', level=logging.INFO)
    elif args.info:
        logging.basicConfig(filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(filemode='w', level=logging.ERROR)

    if not args.testOnly:
        args.out = checkIfExists(args.out, isDir=True, createIfNotExists=True)
    if args.testOnly:
        args.save = checkIfExists(args.save, isDir=True, createIfNotExists=True)
        if args.test:
            checkFile(args.test, fileExtension=".csv")
        else: 
            raise RuntimeError("[{}] -test must be set when -testOnly is set!".format(errTrace))

    if torch.cuda.is_available:
        device = "cuda"
    else: 
        device = "cpu"
    logging.info("Using device:{}".format(device))

    if args.load:
        model = torch.load(args.load)
        logging.info("Loaded BERTforClassification model from {}".format(args.load)) 
    else: 
        model = BERTforClassification(
            modelPath=MODELS[args.model][args.size],
            numClasses=NUM_CLASSES[args.dataset],
            cacheDir=args.cacheDir,
        )
        logging.info("Instantiated a BERTforClassification model") 
    model.to(device)

    dataset = load_dataset(DATASETS[args.dataset], cache_dir=args.cacheDir)

    if not args.testOnly:
        trainDataLoader = createDataLoader(
            dataset=dataset["train"], 
            task=args.dataset,
            modelPath=MODELS[args.model][args.size],
            batchSize=args.batchSize
        )
        valDataLoader = createDataLoader(
            dataset=dataset["validation"], 
            task=args.dataset,
            modelPath=MODELS[args.model][args.size],
            batchSize=args.batchSize
        )
        testDataLoader = createDataLoader(
            dataset=dataset["test"], 
            task=args.dataset,
            modelPath=MODELS[args.model][args.size],
            batchSize=args.batchSize
        )
        _, counts = np.unique(dataset["test"]["label"], return_counts=True)
        randomBaselineAcc = np.max(counts/(dataset["test"].num_rows))
        logging.info("Random baseline classifier's best expected test accuracy: {:0.2f}%".format(randomBaselineAcc*100))
    else:
        testData = readFile(args.test)
        testDataset = Dataset.from_pandas(pd.DataFrame(data=testData))
        testDataLoader = createDataLoader(
            dataset=testDataset, 
            task=args.dataset,
            modelPath=MODELS[args.model][args.size],
            batchSize=args.batchSize,
            shuffle=False
        )

    logging.info(args)

    if not args.testOnly:
        lossFunction = torch.nn.CrossEntropyLoss().to(device)
        modelSavedAs = "{}{}_{}_{}.pt".format(args.out, args.model, args.size, args.dataset)
        bestValAcc = None
        for expInd, learningRate in enumerate(args.learningRate):
            numTrainingSteps = args.numEpochs * len(trainDataLoader)
            maxSteps = args.maxSteps
            if maxSteps == -1:
                maxSteps = numTrainingSteps
            elif maxSteps > 0:
                maxSteps = math.ceil(maxSteps/len(trainDataLoader))
            else: 
                raise ValueError(f"Maximum no. of steps (maxSteps) has to be positive!")
            
            if expInd:
                if args.load:
                    model = torch.load(args.load)
                    logging.info("Loaded BERTforClassification model from {}".format(args.load)) 
                else: 
                    model = BERTforClassification(
                        modelPath=MODELS[args.model][args.size],
                        numClasses=NUM_CLASSES[args.dataset],
                        cacheDir=args.cacheDir,
                    )
                    logging.info("Instantiated a BERTforClassification model") 
                model.to(device)

            if args.skipFinetuning:
                model.freezeBERT()

            if args.optimizer == "adam":
                optimizer = torch.optim.Adam(
                    model.parameters(), 
                    lr=learningRate, 
                    weight_decay=args.weightDecay,
                    amsgrad=args.amsgrad,
                )
            elif args.optimizer == "adagrad":
                optimizer = torch.optim.Adagrad(
                    model.parameters(), 
                    lr=learningRate, 
                    weight_decay=args.weightDecay,                
                )
            else:
                raise ValueError("[main] Invalid input to -optimizer: {}".format(args.optimizer))
            totalSteps = args.numEpochs
            if args.lrScheduler:
                scheduler = transformers.get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0.1*totalSteps,
                    # num_warmup_steps=2000,
                    # num_warmup_steps=0,
                    num_training_steps=totalSteps
                )
            else:
                scheduler = None
            lossFunction = torch.nn.CrossEntropyLoss().to(device)

            logging.info("Learning Rate: {}".format(learningRate))
            
            for epoch in range(args.numEpochs):
                curAcc, curLoss = trainModel(
                    model=model, 
                    dataLoader=trainDataLoader, 
                    lossFunction=lossFunction, 
                    optimizer=optimizer, 
                    device=device, 
                    scheduler=scheduler, 
                    maxSteps=maxSteps,
                )
                maxSteps -= len(trainDataLoader)
                valAcc, valLoss = evalModel(
                    model=model, 
                    dataLoader=valDataLoader,
                    lossFunction=lossFunction,
                    device=device,
                    dataDesc="Validation batch", 
                )

                logging.info("Epoch {}/{}\nTraining Loss: {:0.2f}\nTrain Accuracy: {:0.2f}%\nValidation Loss: {:0.2f}\nValidation Accuracy: {:0.2f}%".format(epoch+1, args.numEpochs, curLoss, curAcc*100, valLoss, valAcc*100))
                logging.info("*****")

                if bestValAcc == None or valAcc >= bestValAcc:
                    bestValAcc = valAcc
                    bestLearningRate = learningRate
                    torch.save(model, modelSavedAs)
                    logging.info("Model saved at '{}'".format(modelSavedAs))
                if maxSteps <= 0:
                    logging.info("Max steps reached!")
                    break
        logging.info("Best learning rate: {}".format(bestLearningRate))
        logging.info("Best model's validation accuracy: {:0.2f}%".format(bestValAcc*100))

        model = torch.load(modelSavedAs)
        logging.info("Loaded best model from {}".format(modelSavedAs))
            
        testAcc, _ = evalModel(
            model=model, 
            dataLoader=testDataLoader,
            lossFunction=lossFunction,
            device=device,
            dataDesc="Test batch", 
        )
        logging.info("Test Accuracy: {:0.2f}%".format(testAcc*100))
    else:
        predictions = testModel(model, testDataLoader, device)
        predictionsDF = pd.DataFrame(testData)
        for i in range(NUM_CLASSES[args.dataset]):
            predictionsDF.insert(i, f"probab_{i}", predictions[:, i], False)
        predictionsDF.insert(0, "prediction", np.argmax(predictions, axis=-1), False)
        predictions = predictionsDF.to_dict("records")
        saveAs = "{}{}_{}_{}.csv".format(args.save, ".".join(args.test.split("/")[-1].split(".")[:-1]), args.model, args.size)
        with open(saveAs, "w") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
            writer.writeheader()
            writer.writerows(predictions)
#---------------------------------------------------------------------------
if __name__=="__main__":
    main()