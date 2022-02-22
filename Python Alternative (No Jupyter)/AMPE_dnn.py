import torch
from torch import nn
import torch.nn.functional as F

TRAINED_MODEL_NAME = "DigitRecognizer_trained_model.pt"

# Modelo de nuestra red neuronal para reconocimiento de dígitos
class DigitRecognizerModel(nn.Module):
    # Constructor
    def __init__(self):
        super().__init__()

        # Parámetros del modelo
        self.is_trained   = False
        self.input_size   = 784
        self.hidden_sizes = [128, 64]
        self.output_size  = 10

        # Definición de capas
        self.fc1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.fc2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.fc3 = nn.Linear(self.hidden_sizes[1], self.output_size)

    # Proceso de inferencia (avance en capas y funciones de activación)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    # Guarda los pesos del modelo en un fichero
    def save_trained_model(self):
        torch.save(self.state_dict(), TRAINED_MODEL_NAME)

    # Carga desde un fichero los pesos (pre-entrenados) del modelo
    def load_trained_model(self):
        try:
            self.load_state_dict(torch.load(TRAINED_MODEL_NAME))
            self.is_trained = True
            return True
        except:
            return False