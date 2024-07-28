import sys
import torch
import random
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        output = self.softmax(x)
        return output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


def accuracy(predictions, targets):
    predicted_labels = torch.argmax(predictions, dim=1)
    true_labels = torch.argmax(targets, dim=1)
    correct = (predicted_labels == true_labels).sum().item()
    total = targets.size(0)
    return correct / total

def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = len(test_loader)

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_accuracy += accuracy(outputs, targets) * inputs.size(0)

    average_loss = total_loss / len(test_loader.dataset)
    average_accuracy = total_accuracy / len(test_loader.dataset)
    return average_loss, average_accuracy


def load_model(model_path, input_size, hidden_size, output_size):
    model = NeuralNetwork(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_sentence(sentence, words):
    sentence_words = sentence.lower().split()
    sentence_words = [word for word in sentence_words if word in words]
    return sentence_words

def sentence_to_features(sentence_words, words):
    features = [1 if word in sentence_words else 0 for word in words]
    return torch.tensor(features).float().unsqueeze(0)

def generate_response(sentence, model, words, classes, data):
    sentence_words = preprocess_sentence(sentence, words)
    if len(sentence_words) == 0:
        return "I'm sorry, but I don't understand. Can you please rephrase or provide more information?"

    features = sentence_to_features(sentence_words, words)
    with torch.no_grad():
        outputs = model(features)

    probabilities, predicted_class = torch.max(outputs, dim=1)
    confidence = probabilities.item()
    predicted_tag = classes[predicted_class.item()]

    if confidence > 0.5:
        for intent in data['intents']:
            if intent['tag'] == predicted_tag:
                return random.choice(intent['responses'])

    return "I'm sorry, but I'm not sure how to respond to that."
     