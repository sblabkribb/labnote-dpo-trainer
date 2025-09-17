import os
import logging
import redis
import argparse
from dotenv import load_dotenv
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

# --- 초기 설정 ---
# .env 파일이 스크립트와 동일한 디렉토리에 있으므로 경로를 수정합니다.
load_dotenv(dotenv_path='.env')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 환경 변수 및 상수 ---
REDIS_URL = os.getenv("REDIS_URL")
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH")
NEW_MODEL_NAME = os.getenv("NEW_MODEL_NAME", "biollama3-v2-dpo")
DPO_KEY_PREFIX = "dpo:preference:"

def fetch_dpo_data_from_redis() -> Dataset:
    """
    Redis에서 'dpo:preference:*' 패턴의 모든 키를 읽어와 Hugging Face Dataset으로 변환합니다.
    하나의 'chosen'과 여러 개의 'rejected' 항목이 있을 경우, 여러 개의 데이터 포인트로 확장합니다.
    """
    logger.info(f"Connecting to Redis at {REDIS_URL}...")
    if not REDIS_URL:
        raise ValueError("REDIS_URL environment variable is not set.")

    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    r.ping()
    
    logger.info("Fetching DPO preference data from Redis...")
    keys = r.keys(f"{DPO_KEY_PREFIX}*")
    
    if not keys:
        logger.warning("No DPO data found in Redis. Exiting.")
        return Dataset.from_dict({"prompt": [], "chosen": [], "rejected": []})

    data = {"prompt": [], "chosen": [], "rejected": []}
    
    for key in keys:
        try:
            pref_data = r.json().get(key)
            
            if isinstance(pref_data, dict) and "prompt" in pref_data and "chosen" in pref_data and "rejected" in pref_data:
                prompt = pref_data["prompt"]
                chosen = pref_data["chosen"]
                
                # 각 rejected 항목에 대해 별도의 데이터 포인트를 생성합니다.
                if isinstance(pref_data["rejected"], list):
                    for rejected_item in pref_data["rejected"]:
                        if rejected_item: # 비어있지 않은 경우에만 추가
                            data["prompt"].append(prompt)
                            data["chosen"].append(chosen)
                            data["rejected"].append(rejected_item)
            else:
                logger.warning(f"Skipping malformed data at key: {key}")

        except Exception as e:
            logger.error(f"Failed to process key {key}: {e}")

    logger.info(f"Successfully fetched and processed {len(keys)} records into {len(data['prompt'])} training examples.")
    return Dataset.from_dict(data)

def main(args):
    """
    DPO 학습 파이프라인 메인 함수
    """
    # 1. 데이터 로드
    dpo_dataset = fetch_dpo_data_from_redis()
    if len(dpo_dataset) == 0:
        return

    # 2. 모델 및 토크나이저 로드
    logger.info(f"Loading base model and tokenizer from: {BASE_MODEL_PATH}")
    if not BASE_MODEL_PATH or not os.path.exists(BASE_MODEL_PATH):
        logger.error(f"BASE_MODEL_PATH ('{BASE_MODEL_PATH}') is not set or does not exist. Please check your .env file and setup.sh script.")
        return
        
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, low_cpu_mem_usage=True, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. DPOTrainer 설정 (CLI 인자 사용)
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        output_dir=args.output_dir, # CLI 인자 사용
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_steps=50,
        report_to="none",
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
    )

    # 4. 학습 실행
    logger.info("Starting DPO training...")
    dpo_trainer.train()
    logger.info("DPO training completed.")

    # 5. 모델 저장
    logger.info(f"Saving trained model to {args.output_dir}")
    dpo_trainer.save_model(args.output_dir)
    
    logger.info("--- DPO Training Pipeline Finished ---")
    logger.info(f"To use the new model with Ollama, you might need to convert it to GGUF and create a new Modelfile.")
    logger.info(f"Example: ollama create {NEW_MODEL_NAME} -f {args.output_dir}/Modelfile")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO training pipeline.")
    
    # 하이퍼파라미터 인자 추가
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of training steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device for training.")
    parser.add_argument("--grad_acc_steps", type=int, default=4, help="Gradient accumulation steps.")
    
    # 경로 인자 추가
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=f"./{NEW_MODEL_NAME}", 
        help="Directory to save the trained model."
    )
    
    cli_args = parser.parse_args()
    main(cli_args)
