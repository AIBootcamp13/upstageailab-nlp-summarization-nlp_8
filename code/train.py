# Dialogue Summarization 경진대회에 오신 여러분 환영합니다! 🎉
# 본 대회에서는 최소 2명에서 최대 7명이 등장하여 나누는 대화를 요약하는 BART 기반 모델의 baseline code를 제공합니다.
# 주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 만들어봅시다!




# ⚙️ 데이터 및 환경설정
# 1) 필요한 라이브러리 설치
# - 필요한 라이브러리를 설치한 후 불러옵니다.
import pandas as pd
import os
import re
import json
import yaml
from glob import glob
from tqdm import tqdm
from pprint import pprint
import torch
import pytorch_lightning as pl
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.
from torch.utils.data import Dataset , DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import wandb # 모델 학습 과정을 손쉽게 Tracking하고, 시각화할 수 있는 라이브러리입니다.
from koeda import AEasierDataAugmentation # <<< NEW: koeda 라이브러리 임포트 >>>
from googletrans import Translator # <<< NEW: 구글 번역 라이브러리 임포트 >>>
from datetime import datetime, timedelta, timezone
import argparse # <<< NEW: 커맨드라인 인자 처리를 위한 라이브러리 추가
# <<< NEW: Back-Translation(BT) 증강 함수 정의 >>>
def augment_text_data_with_BT(text, repetition):
    """입력된 문장에 대해서 BT를 통해 데이터 증강"""
    translator = Translator()
    result = []
    for i in range(repetition):
        en_translated = translator.translate(text, src='ko', dest='en')
        ja_translated = translator.translate(en_translated.text, src='en', dest='ja')
        ko_translated = translator.translate(ja_translated.text, src='ja', dest='ko')
        result.append(ko_translated.text)
    print("원문: ", text)
    print("--"*100)
    for i in range(repetition):
        print(f"증강문{i+1}: ", result[i])
    return result

# 2) Config file 불러오기 (기존 코드와 동일)
config_path = "./config.yaml"
with open(config_path, "r") as file:
    loaded_config = yaml.safe_load(file)

pprint(loaded_config)
# ... (기존 config 설정 코드는 생략)

# 실험별 output_dir을 wandb name + 날짜_시간(분)으로 지정
experiment_name = loaded_config['wandb']['name']
# 1. 한국 표준시(KST)를 UTC+9로 정의합니다.
kst = timezone(timedelta(hours=9))
# 2. 현재 시간을 KST 기준으로 가져와서 포맷팅합니다.
timestamp = datetime.now(kst).strftime("%Y%m%d%H%M")
output_dir = os.path.abspath(os.path.join("output", f"{experiment_name}_{timestamp}"))
os.makedirs(output_dir, exist_ok=True)
loaded_config['general']['output_dir'] = output_dir


# 1. 데이터 가공 및 데이터셋 클래스 구축
class Preprocess:
    def __init__(self,
                 bos_token: str,
                 eos_token: str,
                ) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    def make_set_as_df(file_path, is_train = True):
        df = pd.read_csv(file_path)
        if is_train:
            return df[['fname','dialogue','summary']]
        else:
            df = pd.read_csv(file_path)
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

# <<< MODIFIED: Dataset 클래스 통합 >>>
# DatasetForTrain과 DatasetForVal은 완전히 동일하므로 하나로 합칩니다.
class SummarizationDataset(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, length):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.length = length

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
        item2['decoder_input_ids'] = item2.pop('input_ids')
        item2['decoder_attention_mask'] = item2.pop('attention_mask')
        item.update(item2)
        item['labels'] = self.labels['input_ids'][idx]
        return item

    def __len__(self):
        return self.length

# <<< NEW: AEDA 및 BT 증강을 위한 함수 정의 >>>
def augment_with_aeda(text_list, config):
    if not config.get('training', {}).get('use_aeda', False):
        return []
    repetition = config['training']['repetition']
    aeda = AEasierDataAugmentation(morpheme_analyzer="Okt", punctuations=[".", ",", "!", "?", ";", ":"])
    augmented_texts = []
    for text in tqdm(text_list, desc="Augmenting data with AEDA"):
        try:
            augmented_texts.extend(aeda(text, p=0.3, repetition=repetition))
        except Exception as e:
            print(f"Error processing text: {text[:50]}...")
            print(f"Exception: {e}")
    return augmented_texts

def augment_with_bt(text_list, config):
    if not config.get('training', {}).get('use_bt', False):
        return []
    repetition = config['training']['repetition']
    augmented_texts = []
    for text in tqdm(text_list, desc="Augmenting data with Back-Translation"):
        augmented_texts.extend(augment_text_data_with_BT(text, repetition))
    return augmented_texts

# <<< MODIFIED: 데이터 준비 함수 수정 >>>
# --full 옵션에 따라 다르게 데이터를 로드하도록 함수를 수정합니다.
def prepare_dataset(config, preprocessor, data_path, tokenizer, is_full_train):
    if is_full_train:
        print("--- Loading data for FULL training (train + dev) ---")
        train_file_path = os.path.join(data_path, 'train.csv')
        val_file_path = os.path.join(data_path, 'dev.csv')

        train_df = preprocessor.make_set_as_df(train_file_path)
        val_df = preprocessor.make_set_as_df(val_file_path)
        
        # train.csv와 dev.csv를 하나로 합칩니다.
        full_train_data = pd.concat([train_df, val_df], ignore_index=True)
        
        # <<< NEW: 데이터 증강 로직 추가 >>>
        if config.get('training', {}).get('use_aeda', False):
            print(f"Original training data size: {len(full_train_data)}")
            augmented_dialogues = augment_with_aeda(full_train_data['dialogue'].tolist(), config)
            repetition = config['training']['repetition']
            augmented_summaries = full_train_data['summary'].tolist() * repetition
            augmented_fnames = full_train_data['fname'].tolist() * repetition
            augmented_df = pd.DataFrame({'fname': augmented_fnames, 'dialogue': augmented_dialogues, 'summary': augmented_summaries})
            full_train_data = pd.concat([full_train_data, augmented_df], ignore_index=True)
            print(f"Total training data size after train: {len(full_train_data)}")
        # <<< NEW: Back-Translation 증강 로직 추가 >>>
        if config.get('training', {}).get('use_bt', False):
            print(f"Original training data size: {len(full_train_data)}")
            augmented_dialogues = augment_with_bt(full_train_data['dialogue'].tolist(), config)
            repetition = config['training']['repetition']
            augmented_summaries = full_train_data['summary'].tolist() * repetition
            augmented_fnames = full_train_data['fname'].tolist() * repetition
            augmented_df = pd.DataFrame({'fname': augmented_fnames, 'dialogue': augmented_dialogues, 'summary': augmented_summaries})
            full_train_data = pd.concat([full_train_data, augmented_df], ignore_index=True)
            print(f"Total training data size after BT train: {len(full_train_data)}")

        print(f"Total training data size: {len(full_train_data)}")

        encoder_input, decoder_input, decoder_output = preprocessor.make_input(full_train_data)
        
        tokenized_encoder_inputs = tokenizer(encoder_input, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'])
        tokenized_decoder_inputs = tokenizer(decoder_input, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'])
        tokenized_decoder_outputs = tokenizer(decoder_output, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'])
        
        train_dataset = SummarizationDataset(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_outputs, len(encoder_input))
        
        print('-'*10, 'Make FULL dataset complete', '-'*10,)
        return train_dataset, None # Full 학습 시에는 validation set이 필요 없습니다.

    else:
        print("--- Loading data for standard training (train / dev split) ---")
        train_file_path = os.path.join(data_path, 'train.csv')
        val_file_path = os.path.join(data_path, 'dev.csv')

        train_data = preprocessor.make_set_as_df(train_file_path)
        val_data = preprocessor.make_set_as_df(val_file_path)
        # <<< NEW: 데이터 증강 로직 추가 >>>
        if config.get('training', {}).get('use_aeda', False):
            print(f"Original training data size: {len(train_data)}")
            augmented_dialogues = augment_with_aeda(train_data['dialogue'].tolist(), config)
            repetition = config['training']['repetition']
            augmented_summaries = train_data['summary'].tolist() * repetition
            augmented_fnames = train_data['fname'].tolist() * repetition
            augmented_df = pd.DataFrame({'fname': augmented_fnames, 'dialogue': augmented_dialogues, 'summary': augmented_summaries})
            train_data = pd.concat([train_data, augmented_df], ignore_index=True)
            print(f"Total training data size after train: {len(train_data)}")
        # <<< NEW: Back-Translation 증강 로직 추가 >>>
        if config.get('training', {}).get('use_bt', False):
            print(f"Original training data size: {len(train_data)}")
            augmented_dialogues = augment_with_bt(train_data['dialogue'].tolist(), config)
            repetition = config['training']['repetition']
            augmented_summaries = train_data['summary'].tolist() * repetition
            augmented_fnames = train_data['fname'].tolist() * repetition
            augmented_df = pd.DataFrame({'fname': augmented_fnames, 'dialogue': augmented_dialogues, 'summary': augmented_summaries})
            train_data = pd.concat([train_data, augmented_df], ignore_index=True)
            print(f"Total training data size after BT train: {len(train_data)}")

        encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
        encoder_input_val, decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)

        # Train dataset
        tokenized_encoder_train = tokenizer(encoder_input_train, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'])
        tokenized_decoder_train = tokenizer(decoder_input_train, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'])
        tokenized_output_train = tokenizer(decoder_output_train, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'])
        train_dataset = SummarizationDataset(tokenized_encoder_train, tokenized_decoder_train, tokenized_output_train, len(encoder_input_train))

        # Validation dataset
        tokenized_encoder_val = tokenizer(encoder_input_val, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'])
        tokenized_decoder_val = tokenizer(decoder_input_val, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'])
        tokenized_output_val = tokenizer(decoder_output_val, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'])
        val_dataset = SummarizationDataset(tokenized_encoder_val, tokenized_decoder_val, tokenized_output_val, len(encoder_input_val))
        
        print('-'*10, 'Make train/val dataset complete', '-'*10,)
        return train_dataset, val_dataset

# 2. Trainer 및 Trainingargs 구축하기
def compute_metrics(config, tokenizer, pred):
    # ... (기존 코드와 동일)
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        decoded_preds = [sentence.replace(token," ") for sentence in decoded_preds]
        labels = [sentence.replace(token," ") for sentence in labels]
    results = rouge.get_scores(decoded_preds, labels, avg=True)
    result = {key: value["f"] for key, value in results.items()}
    return result

# <<< MODIFIED: Trainer 로딩 함수 수정 >>>
# --full 옵션에 따라 TrainingArguments를 다르게 설정합니다.
def load_trainer(config, generate_model, tokenizer, train_dataset, val_dataset, is_full_train):
    print('-'*10, 'Make training arguments', '-'*10,)
    
    # Full 학습 시에는 evaluation 관련 설정을 비활성화합니다.
    if is_full_train:
        evaluation_strategy = 'no'
        do_eval = False
        load_best_model_at_end = False
        wandb_name = config['wandb']['name'] + "_full"
    else:
        evaluation_strategy = config['training']['evaluation_strategy']
        do_eval = config['training']['do_eval']
        load_best_model_at_end = config['training']['load_best_model_at_end']
        wandb_name = config['wandb']['name']

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
        evaluation_strategy=evaluation_strategy, # <<< MODIFIED
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        load_best_model_at_end=load_best_model_at_end, # <<< MODIFIED
        seed=config['training']['seed'],
        logging_dir=config['training']['logging_dir'],
        logging_strategy=config['training']['logging_strategy'],
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        do_train=config['training']['do_train'],
        do_eval=do_eval, # <<< MODIFIED
        report_to=config['training']['report_to']
    )

    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        name=wandb_name, # <<< MODIFIED
    )

    os.environ["WANDB_LOG_MODEL"]="true"
    os.environ["WANDB_WATCH"]="false"

    callbacks = []
    # Full 학습 시에는 EarlyStopping이 의미가 없으므로 제외합니다.
    if not is_full_train:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=config['training']['early_stopping_patience'],
            early_stopping_threshold=config['training']['early_stopping_threshold']
        ))

    trainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, # val_dataset이 None일 수 있습니다.
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks=callbacks # <<< MODIFIED
    )
    print('-'*10, 'Make trainer complete', '-'*10,)
    return trainer

def load_tokenizer_and_model_for_train(config,device):
    # ... (기존 코드와 동일)
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    bart_config = BartConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
    special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    print('-'*10, 'Load tokenizer & model complete', '-'*10,)
    return generate_model, tokenizer

# 3. 모델 학습하기
# <<< MODIFIED: main 함수 수정 >>>
# is_full_train 인자를 받도록 수정합니다.
def main(config, is_full_train):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    
    generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)
    
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    data_path = config['general']['data_path']
    
    # is_full_train 값에 따라 데이터셋을 준비합니다.
    train_dataset, val_dataset = prepare_dataset(config, preprocessor, data_path, tokenizer, is_full_train)
    
    # Trainer를 불러옵니다.
    trainer = load_trainer(config, generate_model, tokenizer, train_dataset, val_dataset, is_full_train)
    
    trainer.train()
    
    wandb.finish()

# <<< NEW: __main__ 블록에서 커맨드라인 인자 파싱 >>>
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BART Dialogue Summarization Training")
    parser.add_argument(
        '--full',
        action='store_true', # 이 옵션이 주어지면 True가 저장됩니다.
        help='Train on the full dataset (train + dev) without evaluation.'
    )
    args = parser.parse_args()

    main(loaded_config, is_full_train=args.full)
