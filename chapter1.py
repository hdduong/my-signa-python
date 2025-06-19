import torch ## torch let's us create tensors and also provides helper functions
import torch.nn as nn ## torch.nn gives us nn.Module(), nn.Embedding() and nn.Linear()
import torch.nn.functional as F # This gives us relu()

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


class myNN(nn.Module):

    def __init__(self):
        ## The __init__() method is called when we create an object
        ## from this class. This is where we create and initialize the
        ## weights and biases in the neural network.

        ## When you create a class that inherits from another class
        ## then you always call the parent's __init__() method.
        ## Otherwise, there is no point in inheriting...
        super().__init__()

        ## Now we create and initialize all of the Weights and Biases
        ## in the model with pre-trained values. Each Weight and Bias
        ## is a torch.tensor() object.
        ##
        ## NOTE: w1 = weight 1, b1 = bias 1 etc. (as seen in the
        ## figure above).
        self.w1 = torch.tensor(1.43)
        self.b1 = torch.tensor(-0.61)

        self.w2 = torch.tensor(2.63)
        self.b2 = torch.tensor(-0.27)

        self.w3 = torch.tensor(-3.89)
        self.w4 = torch.tensor(1.35)


    def forward(self, input_values):
        ## The forward() method is called by default when we pass
        ## values to an object created from this class.
        ## This is where we do the math associated with running
        ## data through the neural network.

        top_x_axis_values = (input_values * self.w1) + self.b1
        bottom_x_axis_values = (input_values * self.w2) + self.b2

        top_y_axis_values = F.relu(top_x_axis_values)
        bottom_y_axis_values = F.relu(bottom_x_axis_values)

        output_values = (top_y_axis_values * self.w3) + (bottom_y_axis_values * self.w4)

        return output_values
    
## First, let's create an instance of our neural network.
## We'll call it "model" since that is the standard
## terminology in the field.
model = myNN()

## Now let's see what the neural network outputs
## for Doses = 0.0, 0.5, and 1.0.
##
## Create a tensor with input doses
doses = torch.tensor([0.0, 0.5, 1.0])

## Pass the doses to the model to compute
## the output.
## NOTE: By default, the forward() method will be called on the input
m = model(doses)
print(m)

mr = torch.round(model(doses), decimals=2)
print(mr)

## Create the different doses we want to run through the neural network.
## torch.linspace() creates the sequence of numbers between, and including, 0 and 1.
input_doses = torch.linspace(start=0, end=1, steps=11)
input_model = model(input_doses)
print(input_model)

# now print out the doses to make sure they are what we expect...
print(input_doses)


top_x_axis_values = (model.w1 * input_doses) + model.b1
print(top_x_axis_values)

top_y_axis_values = F.relu(top_x_axis_values)
print(top_y_axis_values)


# Now draw a graph of the input doses and the y-axis output values from the ReLU

## First, set the style for seaborn so that the graph looks cool.
sns.set(style="whitegrid")

## First, draw the individual points
sns.scatterplot(x=input_doses,
                y=top_y_axis_values,
                color='blue',
                s=200)

## Now connect those points with a line
sns.lineplot(x=input_doses,
             y=top_y_axis_values,
             color='blue',
             linewidth=2.5)

## now label the y- and x-axes.
#plt.ylabel('Upper ReLU Output')
#plt.xlabel('Dose')
#plt.show()

## now multiply the doses by the weight (w2) and add the bias (b2) on
## the connection from the input to the bottom activation function.
bottom_x_axis_values = (model.w2 * input_doses) + model.b2
bottom_x_axis_values

## now run those x-axis values through the ReLU...
bottom_y_axis_values = F.relu(bottom_x_axis_values)
bottom_y_axis_values

## Now draw a graph of the input doses and the y-axis output values from the ReLU

## First, set the style for seaborn so that the graph looks cool.
#sns.set(style="whitegrid")

## First, draw the individual points
sns.scatterplot(x=input_doses,
                y=bottom_y_axis_values,
                color='orange',
                s=200)

## Now connect those points with a line
sns.lineplot(x=input_doses,
             y=bottom_y_axis_values,
             color='orange',
             linewidth=2.5)
## now label the y- and x-axes.
plt.ylabel('Bottom ReLU Output')
plt.xlabel('Dose')
plt.show()

final_top_y_axis_values = top_y_axis_values * model.w3
final_top_y_axis_values

final_bottom_y_axis_values = bottom_y_axis_values * model.w4
final_bottom_y_axis_values



sns.set(style="whitegrid")

## Draw the individual points (top)
sns.scatterplot(x=input_doses,
                y=final_top_y_axis_values,
                color='blue',
                s=200)

## Connect those points with a line (top)
sns.lineplot(x=input_doses,
             y=final_top_y_axis_values,
             color='blue',
             linewidth=2.5)

## Draw the individual points (bottom)
sns.scatterplot(x=input_doses,
                y=final_bottom_y_axis_values,
                color='orange',
                s=200)

## Connect those points with a line (bottom)
sns.lineplot(x=input_doses,
             y=final_bottom_y_axis_values,
             color='orange',
             linewidth=2.5)

## now label the y- and x-axes.
plt.ylabel('Final Bent Shapes for Top and Bottom')
plt.xlabel('Dose')
plt.show()

final_bent_shape = final_top_y_axis_values + final_bottom_y_axis_values
final_bent_shape
print(final_bent_shape)

## Now put both bent shapes on the same graph...

## First, set the style for seaborn so that the graph looks cool.
sns.set(style="whitegrid")

## Draw the individual points
sns.scatterplot(x=input_doses,
                y=final_bent_shape,
                color='green',
                s=200)

## Connect those points with a line
sns.lineplot(x=input_doses,
             y=final_bent_shape,
             color='green',
             linewidth=2.5)

## now label the y- and x-axes.
plt.ylabel('Final Bent Shape')
plt.xlabel('Dose')
plt.show()