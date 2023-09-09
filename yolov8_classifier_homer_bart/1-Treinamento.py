
from ultralytics import YOLO
import torch
import os
import shutil
import time
import random
from torchvision import datasets, transforms

def separar_dados_train_test_pasta(source_dir, treinamento_dir, teste_dir, validacao_dir, ratio=(0.2, 0.2, 0.6), seed=42):
    # Define a semente para garantir reproducibilidade
    random.seed(seed)

    # Lista todos os arquivos no diretório de origem
    image_files = [file for file in os.listdir(source_dir)]

    # Embaralha a lista de arquivos
    random.shuffle(image_files)

    # Calcula o número de imagens para treinamento, validação e teste
    num_images = len(image_files)
    num_train = int(num_images * ratio[0])
    num_val = int(num_images * ratio[1])
    num_test = num_images - num_train - num_val

    # Divide a lista de imagens em conjuntos de treinamento, validação e teste
    train_images = image_files[:num_train]
    val_images = image_files[num_train:num_train + num_val]
    test_images = image_files[num_train + num_val:]

    # Cria os diretórios de treinamento, validação e teste, se não existirem
    for directory in [treinamento_dir, validacao_dir, teste_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Copia as imagens para os diretórios correspondentes
    for image in train_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(treinamento_dir, image))

    for image in val_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(validacao_dir, image))

    for image in test_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(teste_dir, image))

    # Exibe as informações sobre a quantidade de dados
    print("Diretorio: ", source_dir)
    print("Quantidade total de dados: ", num_images)
    print("Quantidade de dados de treinamento: ", len(train_images))
    print("Quantidade de dados de validacao: ", len(val_images))
    print("Quantidade de dados de teste: ", len(test_images))




    # Load the YOLOv8 model
if __name__ == '__main__':


    separar_dados_train_test_pasta(
        "./dados/Bom",
        "./dados/dados/train/Bom",
        "./dados/dados/test/Bom",
        "./dados/dados/val/Bom",
        ratio=(0.2, 0.2, 0.6)
        )

    separar_dados_train_test_pasta(
        "./dados/Ruim",
        "./dados/dados/train/Bom",
        "./dados/dados/test/Ruim",
        "./dados/dados/val/Ruim",
        ratio=(0.4, 0.2, 0.4)
        )


    model = YOLO('yolov8n-cls.pt', )  # load a pretrained model (recommended for training)
    
    


# Load a model
    #model = YOLO('yolov8n-cls.pt')  # build a new model from YAML
    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
    #model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='coco128.yaml', epochs=100, imgsz=640)




    model.train(data="./dados/dados", epochs=20, imgsz=224, device=0, workers =32, save_period =1) #Train the model
    model.export()
#Configurar modelo





