import matplotlib.pyplot as plt
import numpy as np
from time import time
from torchvision import datasets, transforms
import torch
from torch import nn, optim
import AMPE_dnn

BATCH_SIZE = 64
NUM_EPOCHS = 15
transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)), ])

# Muestra en pantalla la imagen inferida y la predicción realizada para cada clase [0-9]
def show_infer_results(image, output):
    ps = torch.exp(output)
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    plt.tight_layout() # ajusta espaciado entre subplots

    ax1.imshow(image.resize_(1, 28, 28).numpy().squeeze(), cmap='plasma') # muestra la imagen
    ax1.axis('off') # elimina ejes

    ax2.barh(np.arange(len(ps)), ps) # muestra los resultados en forma de diagrama
    ax2.set_aspect(0.1) # fija el tamaño del diagrama
    ax2.set_yticks(np.arange(len(ps))) # líneas del eje Y
    ax2.set_yticklabels(np.arange(len(ps))) # etiquetas del eje Y
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1) # define el rango del eje X

    plt.show()


# Imprime un subconjunto de 60 elementos del dataset
def print_dataset(datasetloader):
    # Obtenemos un batch de imágenes
    images, labels = next(iter(datasetloader))

    # Bucle para imprimir todas las imágenes de un batch
    num_of_images = (BATCH_SIZE // 10) * 10
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    plt.show()


# Inicializa los pesos de la DNN si esta no está ya entrenada
def initialize_model(model, datasetloader):
    if not model.is_trained:
        print('Initializing model...')

        # Obtenemos una imagen del dataset
        images, labels = next(iter(datasetloader))
        images = images.view(images.shape[0], -1)

        # Inferimos la imagen en el modelo y calculamos la pérdida
        logps = model(images)
        loss = nn.NLLLoss()(logps, labels)  # negative log-likelihood loss

        # Realizamos un ajuste inicial de los pesos a partir de esa diferencia
        print('Before backward pass: \n', model.fc1.weight.grad)
        loss.backward()
        print('After backward pass: \n', model.fc1.weight.grad)


# Entrena el modelo a partir de un conjunto de entrenamiento y un número de etapas
def train_model(model, trainloader, epochs):
    # Función de pérdida y de optimización
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    # Marca de tiempo inicial antes del entrenamiento
    start_time = time()

    # Bucle de entrenamiento que se repite el número de etapas indicado
    for e in range(1, epochs + 1):
        running_loss = 0
        for images, labels in trainloader:
            # Convierte las imágenes del dataset MNIST a un tensor bidimensional de 784 valores
            images = images.view(images.shape[0], -1)

            # Paso de entrenamiento
            optimizer.zero_grad()
            # Inferencia y cálculo de la pérdida
            output = model(images)
            loss = criterion(output, labels)

            # El modelo aprende mediante backpropagation
            loss.backward()
            # Se optimizan los pesos
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"Epoch {e} - Training loss: {running_loss / len(trainloader)} - Accuracy: {100 - running_loss / len(trainloader) * 100}%")

    print("\nTraining Time (in minutes) =", (time() - start_time) / 60)
    model.is_trained = True

# Prueba la precisión del modelo frente a un conjunto de datos de validación
def test_model(model, valloader):
    correct_count, all_count = 0, 0
    # Iteramos sobre el conjunto de validación
    for images, labels in valloader:
        # Inferimos cada imagen del batch individualmente
        for i in range(len(labels)):
            # Obtiene una imagen del batch
            img = images[i].view(1, model.input_size)

            # Infiere la imagen y comprueba el resultado
            predicted_label, output = infer_image(model, img)

            if predicted_label == labels.numpy()[i]:
                correct_count += 1
            all_count += 1

    print(f"Number Of Images Tested = {all_count}")
    print(f"Model Accuracy = {(correct_count / all_count) * 100}%")


# Infiere una imagen en el modelo
def infer_image(model, image):
    # Infiere la imagen en el modelo (deshabilitando el cálculo del gradiente)
    with torch.no_grad():
        output = model(image)

    # Lista de valores con la probabilidad de cada valor
    probabilities = list(torch.exp(output).numpy()[0])

    # Retorna del resultado inferido (label) y la salida del modelo
    return probabilities.index(max(probabilities)), output


# Realiza la inferencia de una imagen en el modelo
def test_image(model, image, label):
    image = image.view(1, 784)

    # Infiere la imagen y muestra la predicción
    predicted_label, output = infer_image(model, image)
    print(f"Predicted Digit = {predicted_label}")
    print(f"True Digit = {label}")

    # Comprueba si la predicción es correcta
    if predicted_label == label:
        print("\nCorrect prediction!")
    else:
        print("\nWrong prediction :/")

    # Muestra la probabilidad de cada valor
    show_infer_results(image.view(1, 28, 28), output)


if __name__ == '__main__':
    # Creamos el modelo de DNN
    model = AMPE_dnn.DigitRecognizerModel()

    # Si no hay un modelo preentrenado, lo inicializa y lo entrena 
    if not model.load_trained_model():
        # Descarga del dataset de entrenamiento
        trainset = datasets.MNIST('trainset', download=True, train=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        # Entrenamiento del modelo
        initialize_model(model, trainloader)
        train_model(model, trainloader, NUM_EPOCHS)
        model.save_trained_model()

    # Descarga del dataset de validación
    valset = datasets.MNIST('valset', download=True, train=False, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)

    # Pone a prueba la validez del modelo frente al conjunto de datos de validación
    test_model(model, valloader)