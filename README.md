# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

## Neural Network Model
![Uploading image.png…]()


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Pharsheen Rahuman M

### Register Number:2122242302193

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
 def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x





# Initialize the Model, Loss Function, and Optimizer



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    #Include your code here
 for epoch in range (epochs):
      optimizer.zero_grad() # Clear gradients
      outputs = ai_brain(X_train)
      loss = criterion(outputs, y_train)
      loss.backward() # Perform backpropagation
      optimizer.step() # Update model parameters

      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
         print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```

### Dataset Information
<img width="257" height="263" alt="image" src="https://github.com/user-attachments/assets/b84a3227-01d3-497e-a081-367c393bfcb7" />


### OUTPUT

### Training Loss Vs Iteration Plot
<img width="742" height="587" alt="image" src="https://github.com/user-attachments/assets/54b8a823-1cc8-4a93-bcd9-9720e0d5271a" />


### New Sample Data Prediction
<img width="351" height="33" alt="image" src="https://github.com/user-attachments/assets/e2e860b2-8bc7-42c8-b1ed-ac6f5864708a" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
