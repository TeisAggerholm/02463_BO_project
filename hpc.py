import torch
import PIL
from torch import nn
from torch.utils.data.dataloader import default_collate
import pickle
import torchvision.models
import numpy as np
from tqdm import tqdm
import pickle
from skopt import gp_minimize
from sklearn.model_selection import ParameterSampler

##
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device:',device)

# Set a random seed for everything important
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

# Set a seed with a random integer, in this case, I choose my verymost favourite sequence of numbers
seed_everything(123)

##
class TumorDataset(torch.utils.data.Dataset): 
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx): 
        target = torch.tensor(self.data[idx][0], dtype=torch.long)  # Ensure target is long for classification
        image = torch.tensor(self.data[idx][1], dtype=torch.float32)  # Convert image to float32
        return image, target

    def __len__(self):
        return len(self.data)  # Use len() instead of shape[0] for lists

def collate_fn(batch):
    return tuple(x_.to(device) for x_ in default_collate(batch))

def get_dataset(test_size, val_size, v=True): 
    with open("dataset.pkl", "rb") as f: 
        rawdata = pickle.load(f)
    
    mean = np.mean(np.stack(rawdata[:,1]))
    std = np.std(np.stack(rawdata[:,1]))

    rawdata[:,1]  = (rawdata[:,1] - mean)/std
    
    labels = {
        "Not cancer": 0, 
        "Cancer": 1
    }

    dataset = TumorDataset(data=rawdata)

    test_amount, val_amount = int(dataset.__len__() * test_size), int(dataset.__len__() * val_size)

    # this function will automatically randomly split your dataset but you could also implement the split yourself
    train_set, test_set = torch.utils.data.random_split(dataset, [
                (dataset.__len__() - (test_amount)), 
                test_amount, 
    ])
    
    print(f"There are {len(train_set)} examples in the training set")
    print(f"There are {len(test_set)} examples in the test set \n")
    print(f"Image shape is: {train_set[0][0].shape}, label example is {train_set[0][1]}")
    
    return train_set, test_set

train_set, test_set = get_dataset(test_size=0.15, val_size=0)

# Make dataloaders
batch_size=16

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

class VGG16(torch.nn.Module):
    def __init__(self, num_classes, in_channels=1, features_fore_linear=25088, dataset=None, dropout_probs = 0.5, final_layer = 4096):
        super().__init__()
        
        # Helper hyperparameters to keep track of VGG16 architecture
        pool_stride = 2
        conv_kernel = 3
        pool_kernel = 2
        optim_momentum = 0.9
        weight_decay = 5e-4
        learning_rate = 1e-5

        # Define features and classifier each individually, this is how the VGG16-D model is orignally defined
        self.features = torch.nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=conv_kernel, padding=1), 
            nn.BatchNorm2d(64),  # Batch Normalization
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=conv_kernel, padding=1), 
            nn.BatchNorm2d(64),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=conv_kernel),
            nn.BatchNorm2d(128),  # Batch Normalization
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=conv_kernel),
            nn.BatchNorm2d(128),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=conv_kernel),
            nn.BatchNorm2d(256),  # Batch Normalization
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=conv_kernel),
            nn.BatchNorm2d(256),  # Batch Normalization
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=conv_kernel),
            nn.BatchNorm2d(256),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
        ).to(device)
        
        self.classifier = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=features_fore_linear, out_features=final_layer),
            nn.ReLU(),
            nn.Dropout(p=dropout_probs),
            nn.Linear(in_features=final_layer, out_features=final_layer),
            nn.ReLU(),
            nn.Dropout(p=dropout_probs),
            nn.Linear(in_features=final_layer, out_features=num_classes),
        ).to(device)
        
        # In the paper, they mention updating towards the 'multinomial logistic regression objective'
        # As can be read in Bishop p. 159, taking the logarithm of this equates to the cross-entropy loss function.
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer - For now just set to Adam to test the implementation
        self.optim = torch.optim.Adam(list(self.features.parameters()) + list(self.classifier.parameters()), lr=learning_rate)
        #self.optim = torch.optim.SGD(list(self.features.parameters()) + list(self.classifier.parameters()), lr=learning_rate, momentum=optim_momentum, weight_decay=weight_decay)

        self.dataset = dataset

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.classifier(self.features(x))

    def train_model(self, train_dataloader, epochs=1, val_dataloader=None):
        
        # Call .train() on self to turn on dropout
        self.train()

        # To hold accuracy during training and testing
        train_accs = []
        test_accs = []

        for epoch in range(epochs):
            
            epoch_acc = 0

            for inputs, targets in tqdm(train_dataloader):
                logits = self(inputs)
                loss = self.criterion(logits, targets)
                loss.backward()
        
                self.optim.step()
                self.optim.zero_grad()

                # Keep track of training accuracy
                epoch_acc += (torch.argmax(logits, dim=1) == targets).sum().item()
            train_accs.append(epoch_acc / len(train_dataloader.dataset))

            # If val_dataloader, evaluate after each epoch
            if val_dataloader is not None:
                # Turn off dropout for testing
                self.eval()
                acc = self.eval_model(val_dataloader)
                test_accs.append(acc)
                print(f"Epoch {epoch} validation accuracy: {acc}, train accuracy: {epoch_acc / len(train_dataloader.dataset)}")
                # turn on dropout after being done
                self.train()
        
        return train_accs, test_accs

    def eval_model(self, test_dataloader):
        
        self.eval()
        total_acc = 0

        for input_batch, label_batch in test_dataloader:
            logits = self(input_batch)

            total_acc += (torch.argmax(logits, dim=1) == label_batch).sum().item()

        total_acc = total_acc / len(test_dataloader.dataset)

        return total_acc

    def predict(self, img_path):
        img = PIL.Image.open(img_path)
        img = self.dataset.dataset.transform(img)
        classification = torch.argmax(self(img.unsqueeze(dim=0)), dim=1)
        return img, classification
    
    def predict_random(self, num_predictions=16):
        """
        Plot random images from own given dataset
        """
        random_indices = np.random.choice(len(self.dataset)-1, num_predictions, replace=False)
        classifcations = []
        labels = []
        images = []
        for idx in random_indices:
            img, label = self.dataset.__getitem__(idx)
            device = next(self.parameters()).device  # Get model's device
            img = img.to(device).unsqueeze(0)  # Ensure correct shape: (1, C, H, W)
            # Move image to same device
            
            classifcation = torch.argmax(self(img), dim=1)

            classifcations.append(classifcation)
            labels.append(label)
            images.append(img)

        return classifcations, labels, images

def get_vgg_weights(model):
    """
    Loads VGG16-D weights for the classifier to an already existing model
    Also sets training to only the classifier
    """
    # Load the complete VGG16 model
    temp = torchvision.models.vgg16(weights='DEFAULT')

    # Get its state dict
    state_dict = temp.state_dict()

    # Change the last layer to fit our, smaller network
    state_dict['classifier.6.weight'] = torch.randn(10, 4096)
    state_dict['classifier.6.bias'] = torch.randn(10)

    # Apply the state dict and set the classifer (layer part) to be the only thing we train
    model.load_state_dict(state_dict)

    for param in model.features.parameters():
        param.requires_grad = False

    model.optim = torch.optim.Adam(model.classifier.parameters())


    return model


### BAYESIAN OPTIMIZATION: 
runs = 1

domain = {
    "dropout_probs": np.arange(0, 0.91, 0.01),
    "final_layer": range(3000,6000)
}


def objective_function(x: list):
    dropout_probs = x[0]
    final_layer = x[1]
    in_channels = 1
    CNN_model = VGG16(num_classes=2, in_channels=in_channels, features_fore_linear=36864, dataset=test_set, dropout_probs=dropout_probs, final_layer=final_layer) 
    train_epochs = 1
    train_accs, val_accs = CNN_model.train_model(train_dataloader, epochs=train_epochs)
    acc = CNN_model.eval_model(test_dataloader)
    
    return - acc

## FOR random
param_list = list(ParameterSampler(domain, n_iter=runs, random_state=32))

best_score = 10000
best_params = None
scores = []
for params in param_list:
    params_formatted = [params["dropout_probs"], params["final_layer"]]
    score = objective_function(params_formatted)
    
    if score < best_score:
        best_score = score
        best_params = params_formatted
    scores.append(best_score)

with open("rnd_scores.pkl", "wb") as f: 
    pickle.dump(scores, f)

with open("rnd_best_params.pkl", "wb") as f: 
    pickle.dump(best_params, f)

print("Completed random selection")

## FOR Bayesian opt
dropout_probs = (0, 0.9)
final_layer = (3000, 6000)
x0 = [param_list[0]["dropout_probs"], param_list[0]["final_layer"]]
y0 = objective_function(x0)

opt = gp_minimize(objective_function,
            [dropout_probs, final_layer],
            acq_func= "EI",
            n_initial_points= 0,
            n_calls= runs-1,
            x0= [x0,],
            y0 =[y0, ],
            xi= 0.005
            )

with open("opt.pkl", "wb") as f: 
    pickle.dump(opt, f)
print("Completed Bayesian OPT")
