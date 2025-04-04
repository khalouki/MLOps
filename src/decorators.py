
import time
import logging

# Configurer le logging
logging.basicConfig(filename="../logs/app.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def log_call(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Appel de la fonction {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"Fin de {func.__name__}")
        return result
    return wrapper

def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} exécuté en {end_time - start_time:.2f} secondes")
        return result
    return wrapper
