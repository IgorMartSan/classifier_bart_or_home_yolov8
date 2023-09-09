from ultralytics import YOLO
import torch
import os
import shutil
import time



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


    # Load the YOLOv8 model


if __name__ == '__main__':

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

    # load a pretrained model (recommended for training)
    model = YOLO(model='C:\\Users\\igor8\\OneDrive\Área de Trabalho\\classification-yolov8\\yolov8\\yolov8_classifier_homer_bart\\runs\\classify\\train\\weights\\last.pt')  

    model.train(data="C:\\Users\\igor8\\PycharmProjects\\yolov8\\yolov8_classifier_homer_bart\\dados", epochs=4, imgsz=224, device=0) #Train the model
    model.export()
#Configurar modelo