
from ultralytics import YOLO
import torch
import os
import shutil
import time



def separar_dados_train_test_pasta(source_dir, treinamento_dir, validacao_dir, ratio=0.7):
    # Lista todos os arquivos no diretório de origem
    cont_treinamento_images = 0
    cont_validacao_images = 0
    if not os.path.exists(validacao_dir):
        os.makedirs(validacao_dir)
    if not os.path.exists(treinamento_dir):
        os.makedirs(treinamento_dir)


    image_files = [file for file in os.listdir(source_dir)]

    # Calcula o número de imagens para treinamento e teste
    num_images = len(image_files)
    num_train = int(num_images * ratio)
    num_test = num_images - num_train

    # Seleciona as imagens igualmente espaçadas
    step = num_images // num_train
    train_images = [image_files[i] for i in range(0, num_images, step)]
    test_images = [image for image in image_files if image not in train_images]

    # Copia as imagens para os diretórios de treinamento e teste
    for image in train_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(validacao_dir, image))
        cont_validacao_images += 1

    for image in test_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(treinamento_dir, image))
        cont_treinamento_images += 1

    # Exibe as informações sobre a quantidade de dados
    print("Diretorio: ", source_dir)
    print("Quantidade total de dados: ", num_images)
    print("Quantidade de dados de treinamento: ", cont_treinamento_images)
    print("Quantidade de dados de validacao: ",cont_validacao_images)



    # Load the YOLOv8 model
if __name__ == '__main__':
    print("Tutorial para treinamento na documentação yolo: https://docs.ultralytics.com/datasets/classify/")
    print("Tutorial nesse video: https://www.youtube.com/watch?v=ZeLg5rxLGLg&ab_channel=Computervisionengineer")
    print("dataset utilizado: https://www.kaggle.com/datasets/juniorbueno/neural-networks-homer-and-bart-classification?resource=download")
    print("Setar o cuda: ")
    #torch.cuda.set_device(0)
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"A gpu está sendo usada ? {device}")
    #time.sleep(10)

    separar_dados_train_test_pasta(
        "C:\\Users\\igor8\\PycharmProjects\\yolov8\\yolov8_classifier_homer_bart\\dados\\bart",
        "C:\\Users\\igor8\\PycharmProjects\\yolov8\yolov8_classifier_homer_bart\\dados\\test\\bart",
        "C:\\Users\\igor8\\PycharmProjects\\yolov8\\yolov8_classifier_homer_bart\\dados\\train\\bart",
        0.3)

    separar_dados_train_test_pasta(
        "C:\\Users\\igor8\\PycharmProjects\\yolov8\\yolov8_classifier_homer_bart\\dados\\homer",
        "C:\\Users\\igor8\\PycharmProjects\\yolov8\yolov8_classifier_homer_bart\\dados\\test\\homer",
        "C:\\Users\\igor8\\PycharmProjects\\yolov8\\yolov8_classifier_homer_bart\\dados\\train\\homer",
        0.3)



    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)




    model.train(data="C:\\Users\\igor8\\PycharmProjects\\yolov8\\yolov8_classifier_homer_bart\\dados", epochs=4, imgsz=224) #Train the model
    model.export()
#Configurar modelo





