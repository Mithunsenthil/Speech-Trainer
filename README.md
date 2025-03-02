# Speech-Trainer

## Overview
Speech-Trainer is a web application designed to help users improve their vocal variety and expression through various exercises. The application processes audio files and provides feedback and suggestions for improvement.

## Tools Used
- Python
- Django
- Pydub
- Whisper
- Groq
- NLTK
- Sentence Transformers
- Joblib

## Minimum Requirements
- Python 3.8 or higher
- Django 3.2 or higher
- Pydub
- Whisper
- Groq
- NLTK
- Sentence Transformers
- Joblib
- MySQL

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Mithunsenthil/Speech-Trainer.git
    cd Speech-Trainer
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Install MySQL:
    - On Windows, download and install MySQLclient 
        ``` 
        pip install mysqlclient
        ```
    - On macOS, use Homebrew:
        ```sh
        brew install mysqlclient
        ```
    - On Linux, use your package manager:
        ```sh
        sudo apt-get install mysql-server
        ```

5. Start the MySQL service:
    - On Windows, start MySQL from the Services panel.
    - On macOS and Linux:
        ```sh
        sudo service mysql start
        ```

6. Update the Django settings to use MySQL:
    Edit `settings.py` and configure the `DATABASES` setting:
    ```python
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': 'speech_trainer',
            'USER': 'speech_user',
            'PASSWORD': 'your_password',
            'HOST': 'localhost',
            'PORT': '3306',
        }
    }
    ```

7. Apply migrations:
    ```sh
    python manage.py migrate
    ```

## Running the Project

1. Start the Django development server:
    ```sh
    python manage.py runserver
    ```

2. Open your web browser and navigate to `http://127.0.0.1:8000/dashboard` to access the application.

## Usage

1. Log in with your credentials.
2. Do the exercises and receive feedback on your vocal performance.
3. View charts and statistics on your progress.

