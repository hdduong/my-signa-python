import torch # torch will allow us to create tensors.
import torch.nn as nn # torch.nn allows us to create a neural network.
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.
from torch.optim import Adam # optim contains many optimizers. This time we're using Adam

import lightning as L # lightning has tons of cool tools that make neural networks easier
from torch.utils.data import TensorDataset, DataLoader # these are needed for the training data

import pandas as pd # We'll use pandas to read in the data and normalize it
from sklearn.model_selection import train_test_split # We'll use this to create training and testing datasets

## NOTE: If you get an error running this block of code, it is probably
##       because you installed a new package earlier and forgot to
##       restart your session for python to find the new module(s).
##
##       To restart your session:
##       - In Google Colab, click on the "Runtime" menu and select
##         "Restart Session" from the pulldown menu
##       - In a local jupyter notebook, click on the "Kernel" menu and select
##         "Restart Kernel" from the pulldown menu

## We'll read in the dataset with the pandas function read_table()
## read_table() can read in various text files including, comma-separated and tab-delimted.
url = "https://raw.githubusercontent.com/StatQuest/signa/main/chapter_03/iris.txt"
df = pd.read_table(url, sep=",", header=None)
## NOTE: If the data were tab-delimted, we would set sep="\t".
## print out the first handful of rows using the head() method
## To name each column, we assign a list of column names to `columns`
df.columns = ["sepal_length",
              "sepal_width",
              "petal_length",
              "petal_width",
              "class"]

## To verify we did that correctly, let's print out the first few rows
print(df.head())

print(df.shape) ## shape returns the rows and columns...

## To determine the number of iris species in the dataset,
## we'll count the number of unique values in the column called `class`.
print(df['class'].nunique())
print(df['class'].unique())

for class_name in df['class'].unique(): # for each unique class name...

    ## ...print out the number of rows associated with it
    print(class_name, ": ", sum(df['class'] == class_name), sep="")

## Print out the first few rows of just the `petal_width` and `sepal_width` columns
print(df[['petal_width', 'sepal_width']].head())

input_values = df[['petal_width', 'sepal_width']]
print(input_values.head())

label_values = df['class']
print(label_values.head())


## Convert the strings in the 'class' column into numbers with factorize()
classes_as_numbers = label_values.factorize()[0] ## NOTE: factorize() returns a list of lists,
                                                 ## and since we only need the first list of values,
                                                 ## we index the output of factorize() with [0].
print(classes_as_numbers) ## print out the numbers


input_train, input_test, label_train, label_test = train_test_split(input_values,
                                                                    classes_as_numbers,
                                                                    test_size=0.25,
                                                                    stratify=classes_as_numbers)

print(input_train.shape)
print(label_train.shape)

print(input_test.shape)
print(label_test.shape)

## Now create a new tensor with one-hot encoded rows for each row in the original dataset.
one_hot_label_train = F.one_hot(torch.tensor(label_train)).type(torch.float32)
print(one_hot_label_train)


## First, determine the maximum values in input_train...
max_vals_in_input_train = input_train.max()
## Now print them out...
print(max_vals_in_input_train)

## Second, determine the minimum values in input_train
min_vals_in_input_train = input_train.min()
## Now print them out...
print(min_vals_in_input_train)


## Now normalize input_train with the maximum and minimum values from input_train
input_train = (input_train - min_vals_in_input_train) / (max_vals_in_input_train - min_vals_in_input_train)
print(input_train.head())


## Convert the DataFrame input_train into tensors
input_train_tensors = torch.tensor(input_train.values).type(torch.float32)

## now print out the first 5 rows to make sure they are what we expect.
print(input_train_tensors[:5])

train_dataset = TensorDataset(input_train_tensors, one_hot_label_train)
train_dataloader = DataLoader(train_dataset)

input_test_tensors = torch.tensor(input_test.values).type(torch.float32)
## now print out the first 5 rows to make sure they are what we expect.
print(input_test_tensors[:5])

class MultipleInsOuts(L.LightningModule):

    def __init__(self):

        super().__init__() ## We call the __init__() for the parent, LightningModule, so that it
                           ## can initialize itself as well.

        ## Now we the seed for the random number generorator.
        ## This ensures that when you create a model from this class, that model
        ## will start off with the exact same random numbers that I started out with when
        ## I created this demo. At least, I hope that is what happens!!! :)
        L.seed_everything(seed=42)

        ############################################################################
        ##
        ## Here is where we initialize the Weights and Biases for the neural network
        ##
        ############################################################################

        ## If you look at the drawing of the network we want to build (above),
        ## you see that we have 2 inputs that lead to 2 activation functions.
        ## We create these connections and initialize their Weights and Biases
        ## with the nn.Linear() function by setting in_features=2 and out_features=2.
        self.input_to_hidden = nn.Linear(in_features=2, out_features=2, bias=True)

        ## Next, we see that the 2 activation functions are connected to 3 outputs.
        ## We create these connections and initialize their Weights and Biases
        ## with the nn.Linear() function by setting in_features=2 and out_features=3.
        self.hidden_to_output = nn.Linear(in_features=2, out_features=3, bias=True)

        ## We'll use Cross Entropy to calculate the loss between what the
        ## neural network's predictions and actual, or known, species for
        ## each row in the dataset.
        ## To learn more about Cross Entropy, see: https://youtu.be/6ArSys5qHAU
        ## NOTE: nn.CrossEntropyLoss applies a SoftMax function to the values
        ## we give it, so we don't have to do that oursevles. However,
        ## when we use this neural network (after it has been trained), we'll
        ## have to remember to apply a SoftMax function to the output.
        # self.loss = nn.CrossEntropyLoss()
        self.loss = nn.MSELoss(reduction='sum')


    def forward(self, input):
        ## First, we run the input values to the activation functions
        ## in the hidden layer
        hidden = self.input_to_hidden(input)
        ## Then we run the values through a ReLU activation function
        ## and then run those values to the output.
        output_values = self.hidden_to_output(torch.relu(hidden))

        return(output_values)


    def configure_optimizers(self):
        ## In this example, configuring the optimizer
        ## consists of passing it the weights and biases we want
        ## to optimize, which are all in self.parameters(),
        ## and setting the learning rate with lr=0.001.
        return Adam(self.parameters(), lr=0.001)


    def training_step(self, batch, batch_idx):
        ## The first thing we do is split 'batch'
        ## into the input and label values.
        inputs, labels = batch

        ## Then we run the input through the neural network
        outputs = self.forward(inputs)

        ## Then we calculate the loss.
        loss = self.loss(outputs, labels)

        ## Lastly, we could add the loss a log file
        ## so that we can graph it later. This would
        ## help us decide if we have done enough training
        ## Ideally, if we do enough training, the loss
        ## should be small and not getting any smaller.
        # self.log("loss", loss)

        return loss
    
model = MultipleInsOuts() # First, make model from the class

## Now print out the name and value for each named parameter
## parameter in the model. Remember parameters are variables,
## like Weights and Biases, that we can train.
for name, param in model.named_parameters():
    print(name, torch.round(param.data, decimals=2))

model = MultipleInsOuts()
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_dataloaders=train_dataloader)

# Run the input_test_tensors through the neural network
predictions = model(input_test_tensors)

print(predictions[0:4,])

## Select the output with highest value...
predicted_labels = torch.argmax(predictions, dim=1) ## dim=0 applies argmax to rows, dim=1 applies argmax to columns
print(predicted_labels[0:4]) # print out the first 4 predictions

## Now compare predicted_labels with test_labels to calculate accuracy
## NOTE: torch.eq() computes element-wise equality between two tensors.
##       label_test, however, is just an array, so we convert it to a tensor
##       before passing it in. torch.sum() then adds up all of the "True"
##       output values to get the number of correct predictions.
##       We then divide the number of correct predictions by the number of predicted values,
##       obtained with len(predicted_labels), to get the percentage of correct predictions
torch_sum = torch.sum(torch.eq(torch.tensor(label_test), predicted_labels)) / len(predicted_labels)
print(torch_sum) # print out the accuracy;
path_to_checkpoint = trainer.checkpoint_callback.best_model_path ## By default, "best" = "most recent"
## First, create a new Lightning Trainer
trainer = L.Trainer(max_epochs=100) # Before, max_epochs=10, so, by setting it to 100, we're adding 90 more.

## Then call trainer.fit() using the path to the most recent checkpoint files
## so that we can pick up where we left off.
trainer.fit(model, train_dataloaders=train_dataloader, ckpt_path=path_to_checkpoint)
# Run the input_test_tensors through the neural network
predictions = model(input_test_tensors)

## Select the output with highest value...
predicted_labels = torch.argmax(predictions, dim=1) ## dim=0 applies softmax to rows, dim=1 applies softmax to columns

## Now compare predicted_labels with test_labels to calculate accuracy
## NOTE: torch.eq() computes element-wise equality between two tensors.
##       label_test, however, is just an array, so we convert it to a tensor
##       before passing it in. torch.sum() then adds up all of the "True"
##       output values to get the number of correct predictions.
##       We then divide the number of correct predictions by the number of predicted values,
##       obtained with len(predicted_labels), to get the percentage of correct predictions
torch_sum =torch.sum(torch.eq(torch.tensor(label_test), predicted_labels)) / len(predicted_labels)
print(torch_sum) # print out the accuracy;