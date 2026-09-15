# Решение задачи NER на датасете NEREL

Извлечение именованных сущностей из русского текста с использованием тонкой настройки RuBERT.

## Результаты

| Метрика | Значение |
|---------|----------|
| **F1 Score** | **0.76** |
| Precision | 0.75 |
| Recall | 0.76 |
| Accuracy | 0.88 |

## Структура проекта

```
NEREL_NER_project/
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── model_utils.py
│   ├── experiments.py
│   └── inference.py
├── notebooks/
│   └── experiments.ipynb
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .dockerignore
└── README.md
```

## Требования

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Docker (опционально)

## Установка и запуск (локально)

```bash
# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Создание окружения
uv venv
.venv\Scripts\activate

# Установка зависимостей
uv pip install -r requirements.txt

# Запуск Jupyter
jupyter notebook
```

Открой `notebooks/experiments.ipynb` и выполни все ячейки.

## Запуск через Docker

```bash
docker build -t nerel-ner .
docker run -p 8888:8888 nerel-ner
```
