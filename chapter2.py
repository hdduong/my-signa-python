import torch ## torch let's us create tensors and also provides helper functions
import torch.nn as nn ## torch.nn gives us nn.Module(), nn.Embedding() and nn.Linear()
import torch.nn.functional as F # This gives us relu()
from torch.optim import SGD # SGD is short of Stochastic Gradient Descent, but
                            # the way we'll use it, passing in all of the training
                            # data at once instead of passing it random subsets,
                            # it will act just like plain old Gradient Descent.

import lightning as L ## Lightning makes it easier to write, optimize and scale our code
from torch.utils.data import TensorDataset, DataLoader ## We'll store our data in DataLoaders

import matplotlib.pyplot as plt ## matplotlib allows us to draw graphs.
import seaborn as sns ## seaborn makes it easier to draw nice-looking graphs.

## NOTE: If you get an error running this block of code, it is probably
##       because you installed a new package earlier and forgot to
##       restart your session for python to find the new module(s).
##
##       To restart your session:
##       - In Google Colab, click on the "Runtime" menu and select
##         "Restart Session" from the pulldown menu
##       - In a local jupyter notebook, click on the "Kernel" menu and select
##         "Restart Kernel" from the pulldown menu

## The inputs are the x-axis coordinates for each data point
## These values represent different doses
training_inputs = torch.tensor([0.0, 0.5, 1.0])

## The labels are the y-axis coordinates for each data point
## These values represent the effectiveness
training_labels = torch.tensor([0.0, 1.0, 0.0])

## Now let's package everything up into a DataLoader...
training_dataset = TensorDataset(training_inputs, training_labels)
dataloader = DataLoader(training_dataset)

class myNN(L.LightningModule):

    def __init__(self):

        super().__init__()

        ## Create all of the weights and biases for the network.
        ## However, this time they are initialized with random values.
        ## We are also wrapping the tensors up in nn.Parameter() objects.
        ## PyTorch will only optimize parameters. There are a lot of
        ## different ways to create parameters, and we'll see those
        ## in later examples, but nn.Parameter() is the most basic.
        self.w1 = nn.Parameter(torch.tensor(0.06))
        self.b1 = nn.Parameter(torch.tensor(0.0))

        self.w2 = nn.Parameter(torch.tensor(3.49))
        self.b2 = nn.Parameter(torch.tensor(0.0))

        self.w3 = nn.Parameter(torch.tensor(-4.11))
        self.w4 = nn.Parameter(torch.tensor(2.74))

        self.loss = nn.MSELoss(reduction='sum')


    def forward(self, input_values):
        ## The forward() method is identical to what we used in Chapter 1.

        top_x_axis_values = (input_values * self.w1) + self.b1
        bottom_x_axis_values = (input_values * self.w2) + self.b2

        top_y_axis_values = F.relu(top_x_axis_values)
        bottom_y_axis_values = F.relu(bottom_x_axis_values)

        output_values = (top_y_axis_values * self.w3) + (bottom_y_axis_values * self.w4)

        return output_values


    def configure_optimizers(self): # this configures the optimizer we want to use for backpropagation.
        return SGD(self.parameters(), lr=0.01)
        ## NOTE: PyTorch doesn't have a Gradient Descent optimizer, just a
        ## Stochastic Gradient Descent (SGD) optimizer. However, since we
        ## are running all 3 doses through the NN each time, rather than a
        ## random subset, we are essentially doing Gradient Descent instead of
        ## SGD.


    def training_step(self, batch, batch_idx): # take a step during gradient descent.
        ## NOTE: When training_step() is called it calculates the loss with the code below...
        inputs, labels = batch # collect input
        outputs = self.forward(inputs) # run input through the neural network
        loss = self.loss(outputs, labels) ## the `loss` quantifies the difference between
                                          ## the observed drug effectiveness in `labels`
                                          ## and the outputs created by the neural network

        return loss
model = myNN() # First, make model from the class

## Now print out the name and value for each named parameter
## parameter in the model. Remember parameters are variables,
## like Weights and Biases, that we can train.
for name, param in model.named_parameters():
    print(name, torch.round(param.data, decimals=2))


 ## now run different doses through the neural network.
output_values = model(training_inputs)
torch.round(output_values, decimals=2)

## Create the different doses we want to run through the neural network.
## torch.linspace() creates the sequence of numbers between, and including, 0 and 1.
input_doses = torch.linspace(start=0, end=1, steps=11)
# now print out the doses to make sure they are what we expect...
print(input_doses)

output_values = model(input_doses)
print(output_values)

## Now draw a graph that shows how well, or poorly, the model
## predicts the training data. At this point, since the
## model is untrained, there should be a big difference between
## the model's output and the training data.

## First, set the style for seaborn so that the graph looks cool.
sns.set(style="whitegrid")

## First, draw the individual output points
sns.scatterplot(x=input_doses,
                y=output_values.detach().numpy(),
                color='green',
                s=200)

## Now connect those points with a line
sns.lineplot(x=input_doses,
             y=output_values.detach().numpy(), ## NOTE: We call .detatch() because...
             color='green',
             linewidth=2.5)

## Add the values in the training dataset
sns.scatterplot(x=training_inputs,
                y=training_labels,
                color='orange',
                s=200)

## now label the y- and x-axes.
plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show()


model = myNN()
## Now train the model...
trainer = L.Trainer(max_epochs=500, # how many times to go through the training data
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False)

trainer.fit(model, train_dataloaders=dataloader)
## Now that we've trained the model, let's print out the
## new values for each Weight and Bias.
for name, param in model.named_parameters():
    print(name, torch.round(param.data, decimals=3))

## now run the different doses through the neural network.
output_values = model(input_doses)
torch.round(output_values, decimals=2)

## Now draw a graph that shows how well, or poorly, the model
## predicts the training data. At this point, since we just
## trained th model, the training data should overlap the
## model's output

## First, set the style for seaborn so that the graph looks cool.
sns.set(style="whitegrid")

## First, draw the individual output points
sns.scatterplot(x=input_doses,
                y=output_values.detach().numpy(),
                color='green',
                s=200)

## Now connect those points with a line
sns.lineplot(x=input_doses,
             y=output_values.detach().numpy(), ## NOTE: We call .detatch() because...
             color='green',
             linewidth=2.5)

## Add the values in the training dataset
sns.scatterplot(x=training_inputs,
                y=training_labels,
                color='orange',
                s=200)

## now label the y- and x-axes.
plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show()
