# **💁🏻🗨️💁🏻‍♂️대화 요약 Baseline code (Sweep 실험 전용 최종 버전)**
import pandas as pd
import os
import yaml
from pprint import pprint
import torch
from rouge import Rouge
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
import wandb

# 3) Configuration 불러오기
config_path = "./config.yaml"
with open(config_path, "r") as file:
    loaded_config = yaml.safe_load(file)
pprint(loaded_config)

# 1. 데이터 가공 및 데이터셋 클래스 구축
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
    @staticmethod
    def make_set_as_df(file_path, is_train = True):
        df = pd.read_csv(file_path)
        if is_train:
            return df[['fname','dialogue','summary']]
        else:
            return df[['fname','dialogue']]
    def make_input(self, dataset,is_test = False):
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x : str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()

class CustomDataset(Dataset):
    def __init__(self, encoder_input, decoder_input, labels):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item.update({f"decoder_{key}": val[idx].clone().detach() for key, val in self.decoder_input.items()})
        item['labels'] = self.labels['input_ids'][idx]
        return item
    def __len__(self):
        return len(self.encoder_input['input_ids'])

def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_file_path = os.path.join(data_path,'train.csv')
    val_file_path = os.path.join(data_path,'dev.csv')
    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)
    encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val, decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)

    def tokenize_data(encoder_inputs, decoder_inputs, decoder_outputs):
        tokenized_encoder = tokenizer(encoder_inputs, return_tensors="pt", padding=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'])
        tokenized_decoder = tokenizer(decoder_inputs, return_tensors="pt", padding=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'])
        tokenized_labels = tokenizer(decoder_outputs, return_tensors="pt", padding=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'])
        return CustomDataset(tokenized_encoder, tokenized_decoder, tokenized_labels)

    train_dataset = tokenize_data(encoder_input_train, decoder_input_train, decoder_output_train)
    val_dataset = tokenize_data(encoder_input_val, decoder_input_val, decoder_output_val)

    print('-'*10, 'Make dataset complete', '-'*10,)
    return train_dataset, val_dataset

# 2. Trainer 및 Trainingargs 구축하기
def compute_metrics(config, tokenizer, pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids
    
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE 점수 계산
    results = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    result = {key: value["f"] for key, value in results.items()}
    
    # --- ✨ 조화 평균을 사용하여 대회의 최종 점수 계산 (가장 중요한 수정사항) ✨ ---
    r1 = result['rouge-1']
    r2 = result['rouge-2']
    rl = result['rouge-l']
    
    # 0으로 나누는 것을 방지하기 위해 작은 값(epsilon)을 더해줍니다.
    epsilon = 1e-8
    
    # 3개 점수의 조화 평균을 계산합니다.
    harmonic_mean = 3 / (1/(r1 + epsilon) + 1/(r2 + epsilon) + 1/(rl + epsilon))
    
    result['competition_score'] = harmonic_mean
    # --------------------------------------------------------------------

    return result

def load_tokenizer_and_model_for_train(config, device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    model_name = config['general']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    print(model.config)
    print('-'*10, 'Load tokenizer & model complete', '-'*10,)
    return model, tokenizer

def load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset):
    print('-'*10, 'Make training arguments', '-'*10,)
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim=config['training']['optim'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        seed=config['training']['seed'],
        logging_dir=config['training']['logging_dir'],
        logging_strategy=config['training']['logging_strategy'],
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval'],
        report_to=config['training']['report_to']
    )
    
    os.environ["WANDB_LOG_MODEL"] = "true"
    os.environ["WANDB_WATCH"] = "false"
    
    my_callback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )
    
    trainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks=[my_callback]
    )
    
    print('-'*10, 'Make trainer complete', '-'*10,)
    return trainer

# 3. 모델 학습하기
def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    
    # WandB 초기화 (Sweep을 위해 main 함수 시작으로 이동)
    wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'])
    
    # Sweep이 전달한 하이퍼파라미터로 기존 config를 업데이트
    for key, value in wandb.config.items():
        if key in config['training']:
            print(f"Sweep is updating training config: {key} = {value}")
            config['training'][key] = value

    # 사용할 모델과 tokenizer를 불러옵니다.
    generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)
    print('-'*10, "tokenizer special tokens : ", tokenizer.special_tokens_map, '-'*10)
    
    # 학습에 사용할 데이터셋을 불러옵니다.
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    data_path = config['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)
    
    # Trainer 클래스를 불러옵니다.
    trainer = load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset)
    
    trainer.train()  # 모델 학습을 시작합니다.
    
    wandb.finish() # 학습이 끝나면 wandb를 종료합니다.

if __name__ == "__main__":
    main(loaded_config)
