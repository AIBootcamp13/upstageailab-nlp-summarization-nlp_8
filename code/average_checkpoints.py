import torch
import os
import shutil
from safetensors.torch import load_file, save_file

def average_checkpoints(checkpoint_dirs, output_dir):
    """
    K-Fold로 생성된 여러 체크포인트 폴더에 있는
    model.safetensors 파일의 가중치 평균을 계산합니다.
    (소스 파일 검사 및 메타데이터 추가 기능 포함)
    """
    if not checkpoint_dirs:
        print("오류: 평균을 계산할 체크포인트 폴더가 지정되지 않았습니다.")
        return

    # --- ✨ 개선: 모든 소스 파일 검사 (존재 여부 + 크기 확인) ✨ ---
    print("소스 체크포인트 파일 검사를 시작합니다...")
    valid_checkpoint_dirs = []
    for checkpoint_dir in checkpoint_dirs:
        model_path = os.path.abspath(os.path.join(checkpoint_dir, 'model.safetensors'))  # 절대 경로로 변환
        if not os.path.exists(model_path):
            print(f"오류: model.safetensors 파일이 없습니다 -> {model_path}")
            continue
        file_size = os.path.getsize(model_path)
        if file_size < 1024:  # 1KB 미만이면 빈 파일로 간주
            print(f"오류: model.safetensors 파일이 비어있습니다 (크기: {file_size} bytes) -> {model_path}")
            continue
        print(f"유효: {model_path} (크기: {file_size / (1024*1024):.2f} MB)")
        valid_checkpoint_dirs.append(checkpoint_dir)
    
    if not valid_checkpoint_dirs:
        print("오류: 유효한 체크포인트가 하나도 없습니다. K-Fold 학습 결과를 확인하세요.")
        return
    if len(valid_checkpoint_dirs) < 2:
        print("경고: 유효한 체크포인트가 1개뿐입니다. 평균 계산 없이 복사만 합니다.")
    # -----------------------------------------------------------------

    print("\n다음 체크포인트들의 가중치 평균을 계산합니다:")
    for d in valid_checkpoint_dirs:
        print(f"- {d}")

    # 첫 번째 체크포인트 로드
    first_checkpoint_path = os.path.join(valid_checkpoint_dirs[0], 'model.safetensors')
    avg_state_dict = load_file(first_checkpoint_path, device="cpu")

    # 나머지 체크포인트 더하기
    for i in range(1, len(valid_checkpoint_dirs)):
        checkpoint_path = os.path.join(valid_checkpoint_dirs[i], 'model.safetensors')
        state_dict = load_file(checkpoint_path, device="cpu")
        for key in avg_state_dict.keys():
            if key in state_dict:
                avg_state_dict[key] += state_dict[key]
            else:
                print(f"경고: {checkpoint_path}에 '{key}' 키가 없습니다. 건너뜁니다.")

    # 평균 계산
    num_checkpoints = len(valid_checkpoint_dirs)
    for key in avg_state_dict.keys():
        avg_state_dict[key] = avg_state_dict[key].float() / num_checkpoints

    # --- 평균 모델 저장 (메타데이터 추가) ---
    os.makedirs(output_dir, exist_ok=True)
    
    output_model_path = os.path.join(output_dir, 'model.safetensors')
    # ✨ 개선: 메타데이터를 명시적으로 추가하여 transformers 호환성 높임 ✨
    metadata = {"format": "pt"}  # PyTorch 형식 명시
    save_file(avg_state_dict, output_model_path, metadata=metadata)
    
    # 설정 파일 복사
    source_dir = valid_checkpoint_dirs[0]
    files_to_copy = [
        'config.json', 'generation_config.json', 'training_args.bin',
        'special_tokens_map.json', 'tokenizer_config.json', 'vocab.json'
    ]
    for filename in files_to_copy:
        src_file = os.path.join(source_dir, filename)
        if os.path.exists(src_file):
            shutil.copy(src_file, output_dir)
    
    print(f"\n성공! 앙상블 모델이 '{output_dir}' 폴더에 저장되었습니다.")
    print("최종 추론(inference) 시 이 폴더 경로를 사용하세요.")


if __name__ == '__main__':
    # --- ✨ 사용자 설정 영역 ✨ ---
    # 각 Fold 실험에서 가장 성능이 좋았던 체크포인트들의 전체 경로 리스트
    # 실제 폴더 경로와 이름이 정확한지 다시 한번 확인해주세요.
    kfold_champion_checkpoints = [
        './output/fold_1/checkpoint-1196',  # 1번 Fold
        './output/fold_2/checkpoint-1196',  # 2번 Fold
        './output/fold_3/checkpoint-1196',  # 3번 Fold
        './output/fold_4/checkpoint-1196',  # 4번 Fold
        './output/fold_5/checkpoint-1196'   # 5번 Fold
    ]

    # K-Fold 앙상블 모델을 저장할 새로운 폴더 이름
    output_kfold_ensemble_dir = './kfold_ensemble_checkpoint'

    # --- 실행 ---
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        print("`safetensors` 라이브러리가 필요합니다. `pip install safetensors` 명령어로 설치해주세요.")
        exit()

    print("K-Fold 앙상블을 시작합니다. 각 Fold의 최고 성능 체크포인트를 사용합니다.")
    average_checkpoints(kfold_champion_checkpoints, output_kfold_ensemble_dir)
