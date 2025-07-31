# 추론(inference) 전용 스크립트
# 이 스크립트는 학습된 모델을 이용해 test 데이터를 요약합니다.
import pandas as pd
import os
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration
from datetime import datetime, timedelta, timezone

# 데이터 전처리 및 데이터셋 클래스 (train.py와 동일)
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
    @staticmethod
    def make_set_as_df(file_path, is_train=True):
        df = pd.read_csv(file_path)
        if is_train:
            return df[['fname','dialogue','summary']]
        else:
            return df[['fname','dialogue']]
    def make_input(self, dataset, is_test=False):
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x : str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()

class DatasetForInference(torch.utils.data.Dataset):
    def __init__(self, encoder_input, test_id, length):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.len = length
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item
    def __len__(self):
        return self.len

def prepare_test_dataset(config, preprocessor, tokenizer):
    test_file_path = os.path.join(config['general']['data_path'],'test.csv')
    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data['fname']
    encoder_input_test, decoder_input_test = preprocessor.make_input(test_data, is_test=True)
    test_tokenized_encoder_inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
    return test_data, test_encoder_inputs_dataset

def load_tokenizer_and_model_for_test(config, device):
    model_name = config['general']['model_name']
    ckt_path = config['inference']['ckt_path']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)
    generate_model = BartForConditionalGeneration.from_pretrained(ckt_path)
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    return generate_model, tokenizer

def inference(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generate_model, tokenizer = load_tokenizer_and_model_for_test(config, device)
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config, preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])
    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(input_ids=item['input_ids'].to(device),
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping'],
                max_length=config['inference']['generate_max_length'],
                num_beams=config['inference']['num_beams'],
            )
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)
    # 불필요한 토큰 제거
    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token," ") for sentence in preprocessed_summary]
    output = pd.DataFrame({
        "fname": test_data['fname'],
        "summary": preprocessed_summary,
    })
    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    # 실험명(wandb name)별로 output 폴더 생성
    experiment_name = config.get('wandb', {}).get('name', 'default_exp')
    output_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "output", experiment_name))
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"output_{timestamp}.csv")
    output.to_csv(filename, index=False)
    print(f"Saved inference results to {filename}")
    return output

if __name__ == "__main__":
    # config.yaml 경로
    config_path = "./config.yaml"
    with open(config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    # 사용할 checkpoint 경로 지정
    loaded_config['inference']['ckt_path'] = "./output/bart_decay_0016_202507301204/checkpoint-1500"
    inference(loaded_config)
