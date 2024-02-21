import json
from datetime import datetime

import dill
import pandas as pd
from pathlib import Path
import os

if os.name == 'nt':
    path = os.environ.get('PROJECT_PATH', '.')
    # Пути относительно корневой директории проекта:
    MODEL_PATH = '.\\data\\models\\'  # Директория с pkl файлом модели
    TESTS_PATH = '.\\data\\tes\\'  # Директория с json файлами тестов
    RESULTS_CSV_PATH = '.\\data\\predictions\\'
else:
    root_path = '/home/airflow/airflow_hw'
    path = os.environ.get('PROJECT_PATH', root_path)
    # Пути относительно корневой директории проекта:
    MODEL_PATH = root_path + '/data/models/'  # Директория с pkl файлом модели
    TESTS_PATH = root_path + '/data/test/'  # Директория с json файлами тестов
    RESULTS_CSV_PATH = root_path + '/data/predictions/'


def directory_exists(path: str) -> str:
    """
    Функция проверки наличия директории и коррекции пути к ней в случае необходимости
    """
    if not Path(path).exists():
        print('Current dir:', os.getcwd())
        print(f'Directory {path} not exist')
        exit()
    if path[-1] == '/' or path[-1] == '\\':
        path = path[:-1]

    return path


def get_model_name(path=MODEL_PATH, ext='pkl') -> str:
    """
    Функция получения имени самого позднего файла по расширению из указанной директории
    """
    # Проверка наличия директории
    path = directory_exists(path)

    if path[-1] == '/' or path[-1] == '\\':
        path = path[:-1]

    # Вычитывание всех файлов из директории
    model_counts = 0
    files = os.listdir(path)
    for i, file in enumerate(files):
        # Определение файлов с нужным расширением
        if os.path.splitext(file)[1][1:] != ext:
            files.pop(i)
        else:
            model_counts += 1

    if model_counts == 0:
        print(f'Files not found in directory "{path}"')
        return 'Error'

    # Список файлов в список файлов с путями
    files = [os.path.join(path, file) for file in files]

    # Самый последний файл
    last_model = max(files, key=os.path.getctime)

    return last_model


def get_dicts_tests(path=TESTS_PATH) -> list:
    """
    Функция получения списка словарей из файлов json в указанной директории
    """
    # Проверка наличия директории
    path = directory_exists(path)

    list_dict = []
    for file in os.listdir(path):
        filename = f'{path}/{file}'
        if os.path.isfile(filename) and os.path.splitext(filename)[1] == '.json':
            with open(filename, 'r') as file_json:
                list_dict.append(json.load(file_json))

    return list_dict


def predict():
    model_filename = get_model_name()

    # Загрузка обученной модели из файла
    with open(model_filename, 'rb') as file:
        model = dill.load(file)

    # Формирование списка словарей из файлов json
    tests = get_dicts_tests()
    # Создание датафрейма с тестами и результатами предсказания
    df = pd.DataFrame(tests)
    df.insert(len(df.columns), 'pred', model.predict(df))
    # Удаление ненужных колонок
    df = df[['id', 'pred']]
    # И сохранение в CSV
    preds_filename = f'{RESULTS_CSV_PATH}preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df.to_csv(preds_filename, sep=',', index=False, encoding='utf-8')


if __name__ == '__main__':
    predict()
