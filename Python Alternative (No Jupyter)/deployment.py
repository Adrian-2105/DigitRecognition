import AMPE_dnn
import entrenamiento_evaluacion
from torchvision import datasets
import torch
import sys
from PIL import Image
from PIL import ImageOps

# Carga una imagen y le aplica todo el preprocesamiento necesario para que la imagen
# pueda ser correctamente inferida en el reconocedor de dígitos
def load_image(path):
    image = Image.open(path)
    image = image.resize((28, 28), Image.ANTIALIAS) # escala la imagen (sin mantener el ratio de aspecto)
    image = ImageOps.grayscale(image)
    
    thresh = 145
    image = image.convert('L').point(lambda x : 255 if x > thresh else x - x // 2, mode='L')

    image = ImageOps.invert(image) # invertimos los colores de la imagen (para adecuarlo al formato del trainset)
    return image

# Realiza la inferencia de una imagen y muestra el resultado de la predicción
def test_image(image, label):
    # Creamos el modelo de DNN
    model = AMPE_dnn.DigitRecognizerModel()
    model.load_trained_model()
    if not model.is_trained:
        print("Model is not trained. Please, train it before the execution")
        exit(1)

    # Si no es una imagen del conjunto de validación, entonces le aplica la transformación a tensor
    if type(image) is not torch.Tensor:
        image = entrenamiento_evaluacion.transform(image)

    # Infiere la imagen en el modelo
    entrenamiento_evaluacion.test_image(model, image, label)


if __name__ == '__main__':
    if len(sys.argv) == 1: # se coge una imagen del conjunto de validación
        # Descarga del dataset de validación
        valset = datasets.MNIST('valset', download=True, train=False, transform=entrenamiento_evaluacion.transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
        # Extraemos una imagen de conjunto
        image, label = next(iter(valloader))
        image = image[0]
        label = label[0]

    elif len(sys.argv) == 2 and sys.argv[1] == 'dataset': # imprime un subconjunto del conjunto de evaluación
        # Descarga del dataset de validación
        valset = datasets.MNIST('valset', download=True, train=False, transform=entrenamiento_evaluacion.transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
        entrenamiento_evaluacion.print_dataset(valloader)
        exit(0)
        
    elif len(sys.argv) == 3: # imagen pasada como argumento
        image = load_image(sys.argv[1])
        label = int(sys.argv[2])
       
    else: # menú de ayuda, uso incorreto
        print(f'Use:')
        print(f'- python {sys.argv[0]}                      # infers an image from the validation dataset')
        print(f'- python {sys.argv[0]} dataset              # shows a subset of the validation dataset')
        print(f'- python {sys.argv[0]} <filepath> <label>   # infers the specified image')
        print(f'  -> Example: python deployment.py ./MyNumberIs6.jpg 6')
        exit(1)

    # Inferimos la imagen y mostramos el resultado
    test_image(image, label)



    