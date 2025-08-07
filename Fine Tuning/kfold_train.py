# 💁🏻🗨️💁🏻♂️대화 요약 Baseline code (K-Fold 통합 버전)
# 이 스크립트는 일반 학습과 K-Fold 교차 검증 학습을 모두 지원합니다.
# ✨ 최종 수정: 엔티티 치환(Entity Swapping) 데이터 증강 기능이 추가되었습니다.

# 실행 방법:
# - 일반 학습: python kfold_train.py
# - K-Fold 학습: python kfold_train.py --kfold --n_splits 5

# ⚙️ 데이터 및 환경설정
import pandas as pd
import os
import re
import json
import yaml
import numpy as np
import argparse # 모드 전환을 위한 라이브러리
from glob import glob
from tqdm import tqdm
from pprint import pprint
import torch
from sklearn.model_selection import KFold
from rouge import Rouge
from torch.utils.data import Dataset , DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import wandb
from datetime import datetime
import copy # ✨ K-Fold 설정을 위해 추가
import random # ✨ 엔티티 치환을 위해 추가

# --- ✨ 여기에 '엔티티 치환' 데이터 증강 함수 추가! ✨ ---
def augment_entity_swapping(dialogue, summary):
    """
    대화와 요약 쌍에서 #Person 토큰들을 일관되게 바꿔치기하여
    새로운 '가짜 연습문제' 데이터를 생성합니다.
    """
    person_tokens = set(re.findall(r"(#Person\d+#)", dialogue + summary))
    if len(person_tokens) < 2:
        return dialogue, summary
    
    original_tokens = list(person_tokens)
    shuffled_tokens = random.sample(original_tokens, len(original_tokens))
    
    while any(o == s for o, s in zip(original_tokens, shuffled_tokens)):
        random.shuffle(shuffled_tokens)
        
    swap_map = dict(zip(original_tokens, shuffled_tokens))
    
    new_dialogue = dialogue
    new_summary = summary
    for old, new in swap_map.items():
        new_dialogue = re.sub(r'\b' + re.escape(old) + r'\b', new, new_dialogue)
        new_summary = re.sub(r'\b' + re.escape(old) + r'\b', new, new_summary)
        
    return new_dialogue, new_summary
# ----------------------------------------------------------------

# 1. 데이터 가공 및 데이터셋 클래스 구축 (기존 코드와 동일)
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
    def make_input(self, dataset, is_test = False):
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x : str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()

class DatasetForTrain(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids'); item2.pop('attention_mask')
        item.update(item2)
        item['labels'] = self.labels['input_ids'][idx]
        return item
    def __len__(self):
        return self.len

class DatasetForVal(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids'); item2.pop('attention_mask')
        item.update(item2)
        item['labels'] = self.labels['input_ids'][idx]
        return item
    def __len__(self):
        return self.len

def prepare_train_dataset(config, preprocessor, tokenizer, train_data, val_data):
    print('-'*150)
    print(f'train_data:\n {train_data["dialogue"].iloc[0]}')
    print(f'train_label:\n {train_data["summary"].iloc[0]}')
    print('-'*150)
    print(f'val_data:\n {val_data["dialogue"].iloc[0]}')
    print(f'val_label:\n {val_data["summary"].iloc[0]}')
    encoder_input_train , decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val , decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    print('-'*10, 'Load data complete', '-'*10)
    tokenized_encoder_inputs = tokenizer(encoder_input_train, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_inputs = tokenizer(decoder_input_train, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_ouputs = tokenizer(decoder_output_train, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    train_inputs_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_ouputs, len(encoder_input_train))
    val_tokenized_encoder_inputs = tokenizer(encoder_input_val, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_inputs = tokenizer(decoder_input_val, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_ouputs = tokenizer(decoder_output_val, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    val_inputs_dataset = DatasetForVal(val_tokenized_encoder_inputs, val_tokenized_decoder_inputs, val_tokenized_decoder_ouputs, len(encoder_input_val))
    print('-'*10, 'Make dataset complete', '-'*10)
    return train_inputs_dataset, val_inputs_dataset

# 2. Trainer 및 기타 함수들 (기존 코드와 동일)
def compute_metrics(config,tokenizer,pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    results = rouge.get_scores(decoded_preds, labels, avg=True)
    result = {key: value["f"] for key, value in results.items()}
    return result

def load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset):
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
        optim =config['training']['optim'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy =config['training']['save_strategy'],
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
    if config['training']['report_to'] == 'wandb':
        wandb.init(
            entity=config['wandb']['entity'],
            project=config['wandb']['project'],
            name=config['wandb']['name'],
            reinit=True
        )
        os.environ["WANDB_LOG_MODEL"]="true"
        os.environ["WANDB_WATCH"]="false"
    
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )
    trainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics = lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks = [MyCallback]
    )
    return trainer

def load_tokenizer_and_model_for_train(config, device):
    model_name = config['general']['model_name']
    bart_config = BartConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
    if 'special_tokens' in config['tokenizer']:
        special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
        tokenizer.add_special_tokens(special_tokens_dict)
        generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    return generate_model, tokenizer

# 3. 메인 실행 로직 (기존 코드와 동일)
def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-'*10, f'Device: {device}', '-'*10)
    data_path = config['general']['data_path']
    train_df = pd.read_csv(os.path.join(data_path,'train.csv'))
    val_df = pd.read_csv(os.path.join(data_path,'dev.csv'))
    generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, tokenizer, train_df, val_df)
    trainer = load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset)
    trainer.train()
    if config['training']['report_to'] == 'wandb':
        wandb.finish()

def kfold_main(config, n_splits):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_path = config['general']['data_path']
    train_file_path = os.path.join(data_path, 'train.csv')
    train_df = pd.read_csv(train_file_path)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config['training']['seed'])
    all_eval_scores = []
    base_output_dir = config['general'].get('output_dir', './output/kfold_experiment')
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        
    for fold, (train_index, val_index) in enumerate(kf.split(train_df)):
        print(f"\n{'='*25} Fold {fold+1}/{n_splits} Training Start {'='*25}")
        
        generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)
        train_fold_df = train_df.iloc[train_index].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_index].reset_index(drop=True)

        # --- ✨ 여기에 데이터 증강(특별 과외) 코드 추가! ✨ ---
        print(f"Original fold training data size: {len(train_fold_df)}")
        augmented_data = []
        # 원본 데이터 1개당 2개의 '가짜 연습문제'를 만듭니다 (필요시 횟수 조절)
        for _ in range(2): 
            for _, row in train_fold_df.iterrows():
                new_dialogue, new_summary = augment_entity_swapping(row['dialogue'], row['summary'])
                augmented_data.append({'fname': row['fname'], 'dialogue': new_dialogue, 'summary': new_summary})
        
        augmented_df = pd.DataFrame(augmented_data)
        # 원본 데이터와 증강 데이터를 합치고 순서를 섞습니다.
        train_fold_df = pd.concat([train_fold_df, augmented_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
        print(f"Fold training data size after augmentation: {len(train_fold_df)}")
        # ----------------------------------------------------

        config_fold = copy.deepcopy(config)
        fold_output_dir = os.path.join(base_output_dir, f"fold_{fold+1}")
        config_fold['general']['output_dir'] = fold_output_dir
        config_fold['training']['logging_dir'] = os.path.join(fold_output_dir, 'logs')
        if config_fold.get('training', {}).get('report_to') == 'wandb':
            original_wandb_name = config_fold.get('wandb', {}).get('name', 'kfold_run')
            config_fold['wandb']['name'] = f"{original_wandb_name}_fold_{fold+1}"
            
        preprocessor = Preprocess(config_fold['tokenizer']['bos_token'], config_fold['tokenizer']['eos_token'])
        train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config_fold, preprocessor, tokenizer, train_fold_df, val_fold_df)
        trainer = load_trainer_for_train(config_fold, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset)
        trainer.train()
        eval_results = trainer.evaluate()
        print(f"\nFold {fold+1} Evaluation Results: {eval_results}")
        all_eval_scores.append(eval_results['eval_rouge-1'])
        if config_fold.get('training', {}).get('report_to') == 'wandb':
            wandb.finish()
            
    print(f"\n{'='*25} K-Fold Cross-Validation Final Results {'='*25}")
    print(f"All Fold Scores (rouge-1): {all_eval_scores}")
    print(f"Average Score (rouge-1): {np.mean(all_eval_scores):.4f}")
    print(f"Standard Deviation: {np.std(all_eval_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="대화 요약 모델 학습 스크립트")
    parser.add_argument('--kfold', action='store_true', help="K-Fold 교차 검증 모드로 학습을 실행합니다.")
    parser.add_argument('--n_splits', type=int, default=5, help="K-Fold에 사용할 폴드 수 (기본값: 5)")
    args = parser.parse_args()
    config_path = "./config.yaml"
    with open(config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    if args.kfold:
        print("K-Fold 교차 검증 모드로 학습을 시작합니다.")
        kfold_main(loaded_config, n_splits=args.n_splits)
    else:
        print("일반 학습 모드로 학습을 시작합니다.")
        main(loaded_config)
