"""
작성자 : kp
작성일 : 2025-05-14
목적 : HR 추천 시스템의 주요 설정 값 관리
내용 : 시스템 운영에 필요한 모델 이름, 통합 데이터 파일 경로 (직원 및 채용 공고 포함), 
ChromaDB 관련 설정 및 기본 추천 결과 수 등의 핵심 설정 변수를 정의합니다. 
이 파일을 통해 시스템의 기본 동작 방식을 쉽게 변경하고 관리할 수 있습니다.
"""
# hr_recommender/config.py

# --- 데이터 파일 경로 ---
# 이제 hr_data.json에 직원과 채용 공고 정보가 모두 포함됩니다.
INTEGRATED_DATA_FILE = 'data/hr_data.json'

# --- Sentence Transformer 모델 이름 ---
MODEL_NAME = 'all-MiniLM-L6-v2'
# 다국어 모델 또는 한국어 특화 모델 사용시 아래 주석 해제 후 MODEL_NAME 변경
# MODEL_NAME = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
# MODEL_NAME = 'jhgan/ko-sroberta-multitask'

# --- ChromaDB 설정 ---
COLLECTION_NAME = "hr_job_embeddings_collection_v2" # 컬렉션 이름 변경 (데이터 구조 변경 반영)
CHROMA_DB_PATH = "./chroma_db_store"
CHROMA_UPSERT_BATCH_SIZE = 5000 # ChromaDB upsert 시 최대 아이템 수

# --- 추천 설정 ---
DEFAULT_NUM_RESULTS = 5 # 추천 결과 수를 늘려 직원과 공고가 섞여 나올 수 있도록 함
