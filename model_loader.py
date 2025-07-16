import threading
from concurrent.futures import ThreadPoolExecutor
import transformers
import os

class ModelLoader:
    def __init__(self, model_configs):
        self.model_configs = model_configs  # {'codellama': path_or_id, 'sqlcoder': path_or_id}
        self.models = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=len(model_configs))
        self.futures = {}

    def _load_model(self, model_name, model_path, hf_token=None):
        # Загрузка модели через transformers с поддержкой HF токена
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, token=hf_token)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, token=hf_token)
        return {'model': model, 'tokenizer': tokenizer}

    def load_all(self, hf_token=None):
        for name, path in self.model_configs.items():
            self.futures[name] = self.executor.submit(self._load_model, name, path, hf_token)

    def get_model(self, model_name):
        if model_name not in self.futures:
            raise ValueError(f"Model {model_name} not configured")
        result = self.futures[model_name].result()  # блокирует поток, если модель еще не загружена
        with self.lock:
            if model_name not in self.models:
                self.models[model_name] = result
        return self.models[model_name]

if __name__ == "__main__":
    hf_token = os.getenv('HF_TOKEN')
    configs = {
        'codellama': input('Введите путь к модели CodeLlama: '),
        'sqlcoder': input('Введите путь к модели SQLCoder: ')
    }
    loader = ModelLoader(configs)
    loader.load_all(hf_token if hf_token else None)
    print('Выберите модель: 1 - codellama, 2 - sqlcoder')
    choice = input('Введите номер модели: ')
    if choice == '1':
        model = loader.get_model('codellama')
        print('CodeLlama загружена!')
    elif choice == '2':
        model = loader.get_model('sqlcoder')
        print('SQLCoder загружена!')
    else:
        print('Неверный выбор!')
