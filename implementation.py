import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class CyberPublicHealthSystem:
    def __init__(self, population_size, infection_rate, intervention_effectiveness):
        self.population_size = population_size
        self.infection_rate = infection_rate
        self.intervention_effectiveness = intervention_effectiveness
        self.population = np.zeros(population_size)  # 0: healthy, 1: infected
        self.infected_count = 0

    def initialize_infection(self, initial_infected):
        indices = np.random.choice(self.population_size, initial_infected, replace=False)
        self.population[indices] = 1
        self.infected_count = initial_infected

    def spread_infection(self):
        new_infections = 0
        for i in range(self.population_size):
            if self.population[i] == 0:  # If healthy
                if np.random.rand() < self.infection_rate:
                    self.population[i] = 1
                    new_infections += 1
        self.infected_count += new_infections

    def apply_intervention(self):
        for i in range(self.population_size):
            if self.population[i] == 1:  # If infected
                if np.random.rand() < self.intervention_effectiveness:
                    self.population[i] = 0
                    self.infected_count -= 1

    def simulate(self, steps, intervention_step=None):
        history = []
        for step in range(steps):
            if intervention_step is not None and step == intervention_step:
                self.apply_intervention()
            self.spread_infection()
            history.append(self.infected_count)
        return history

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def train_model(data, labels, input_size, hidden_size, output_size, epochs=100, lr=0.01):
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        inputs = torch.tensor(data, dtype=torch.float32)
        targets = torch.tensor(labels, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model

if __name__ == '__main__':
    # Simulate Cyber Public Health System
    population_size = 1000
    infection_rate = 0.01
    intervention_effectiveness = 0.5
    initial_infected = 10
    steps = 50
    intervention_step = 25

    system = CyberPublicHealthSystem(population_size, infection_rate, intervention_effectiveness)
    system.initialize_infection(initial_infected)
    history = system.simulate(steps, intervention_step)

    print("Infection history:", history)

    # Train a simple neural network on dummy data
    np.random.seed(42)
    torch.manual_seed(42)

    data = np.random.rand(100, 10)  # 100 samples, 10 features
    labels = np.random.randint(0, 2, size=100)  # Binary classification (0 or 1)

    input_size = 10
    hidden_size = 16
    output_size = 2
    epochs = 50
    lr = 0.01

    model = train_model(data, labels, input_size, hidden_size, output_size, epochs, lr)

    # Test the trained model on a dummy input
    test_input = torch.tensor(np.random.rand(1, 10), dtype=torch.float32)
    prediction = model(test_input)
    print("Test input prediction:", prediction)