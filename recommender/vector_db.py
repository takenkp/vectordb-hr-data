"""
작성자 : kp
작성일 : 2025-05-14
목적 : ChromaDB 벡터 데이터베이스 연동 및 관리 (직원 및 채용 공고 통합)
내용 : ChromaDB 클라이언트를 초기화하고 지정된 컬렉션을 사용합니다. 
직원(HR) 데이터와 채용 공고(Job Description) 데이터를 각각의 준비 함수를 통해 임베딩용 텍스트로 변환하고,
임베딩 모델을 사용해 벡터로 변환합니다. 'doc_type' 메타데이터를 추가하여 문서 종류를 구분하며,
ChromaDB에서 지원하는 타입으로 메타데이터를 가공하여 저장합니다. 데이터 변경 감지 시 컬렉션을 재생성하며, 
대량 데이터 임베딩 및 ChromaDB 업로드 시 설정된 배치 크기(CHROMA_UPSERT_BATCH_SIZE)를 적용합니다.
"""
# hr_recommender/recommender/vector_db.py

import chromadb
from .embedding_utils import prepare_text_for_employee_embedding, prepare_text_for_job_embedding
import math # 배치 처리를 위해 추가

# config는 main.py에서 로드되므로, 여기서 직접 임포트하지 않고 upsert_batch_size를 인자로 받도록 수정 가능
# 또는 main에서 config 객체를 넘겨받도록 할 수 있음. 여기서는 config에서 직접 값을 가져오는 것으로 가정.
try:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import CHROMA_UPSERT_BATCH_SIZE
except ImportError:
    print("경고: config 모듈에서 CHROMA_UPSERT_BATCH_SIZE를 가져올 수 없습니다. 기본값 1000을 사용합니다.")
    CHROMA_UPSERT_BATCH_SIZE = 1000


def _process_metadata_for_db(item_data):
    """ChromaDB 저장을 위해 메타데이터를 가공합니다 (리스트는 문자열로, 딕셔너리는 펼치기)."""
    processed_metadata = {}
    for key, value in item_data.items():
        if key == 'education' and isinstance(value, dict): # 직원의 education 필드
            for edu_key, edu_value in value.items():
                processed_metadata[f"education_{edu_key}"] = str(edu_value) if edu_value is not None else None
        elif isinstance(value, list):
            processed_metadata[key] = ", ".join(map(str, value)) if value else ""
        elif isinstance(value, (str, int, float, bool)) or value is None:
            processed_metadata[key] = value
        else:
            processed_metadata[key] = str(value)
    return processed_metadata

def setup_chromadb_collection(client, collection_name, employee_data, job_data, embedding_model):
    """
    ChromaDB 컬렉션을 설정하고 직원 및 채용 공고 데이터를 임베딩하여 저장합니다.
    Args:
        client (chromadb.Client): ChromaDB 클라이언트.
        collection_name (str): 컬렉션 이름.
        employee_data (list): HR 직원 데이터 리스트.
        job_data (list): 채용 공고 데이터 리스트.
        embedding_model (SentenceTransformer): 임베딩 모델.
    Returns:
        chromadb.Collection: ChromaDB 컬렉션 객체.
    """
    all_data_for_embedding = []
    if employee_data:
        all_data_for_embedding.extend([{'data': emp, 'type': 'employee'} for emp in employee_data])
    if job_data:
        all_data_for_embedding.extend([{'data': job, 'type': 'job'} for job in job_data])

    if not all_data_for_embedding:
        print("임베딩할 데이터가 없습니다 (직원 및 채용 공고 모두 비어 있음).")
        # 컬렉션이 존재하지 않으면 생성은 하되 비어있는 상태로 반환
        try:
            collection = client.get_or_create_collection(name=collection_name)
            print(f"'{collection_name}' 컬렉션이 비어있는 상태로 준비되었습니다.")
        except Exception as e:
            print(f"빈 컬렉션 '{collection_name}' 생성/가져오기 중 오류: {e}")
            # 심각한 오류 시 None 반환 또는 예외 재발생 고려
            return None
        return collection


    current_doc_ids_from_files = set()
    if employee_data:
        current_doc_ids_from_files.update(emp['id'] for emp in employee_data if isinstance(emp, dict) and 'id' in emp)
    if job_data:
        current_doc_ids_from_files.update(job['id'] for job in job_data if isinstance(job, dict) and 'id' in job)
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"'{collection_name}' 컬렉션에서 기존 데이터를 로드했습니다.")
        
        db_item_count = collection.count()
        
        should_recreate = False
        if db_item_count != len(current_doc_ids_from_files):
            print(f"데이터 개수 변경 감지 (DB: {db_item_count}개, 파일 통합: {len(current_doc_ids_from_files)}개).")
            should_recreate = True
        else: # 개수가 같으면 ID 목록 비교 (샘플링 또는 전체 비교 - 여기서는 개수만으로 단순화)
            # 더 정확하려면 DB의 모든 ID를 가져와 current_doc_ids_from_files와 비교해야 함.
            # collection.get(include=[]) 등으로 ID만 가져오는 기능이 제한적일 수 있어, 개수 기반으로 우선 처리.
            print("데이터 개수 일치. ID 상세 비교는 생략하고 기존 데이터 사용 가정 (정확한 변경 감지 로직 개선 가능).")


        if should_recreate:
            print(f"컬렉션 '{collection_name}'을(를) 재생성합니다.")
            client.delete_collection(name=collection_name)
            collection = client.create_collection(name=collection_name)
        elif not should_recreate and db_item_count == len(current_doc_ids_from_files) and db_item_count > 0 :
             print("데이터 변경 없음 (개수 기준). 기존 임베딩을 사용합니다.")
             return collection
            
    except Exception: 
        print(f"'{collection_name}' 컬렉션을 새로 생성합니다.")
        collection = client.create_collection(name=collection_name)

    print("통합 데이터(직원, 채용 공고) 임베딩 및 ChromaDB 저장 중...")
    
    all_documents = []
    all_metadatas_to_db = []
    all_ids = []

    for item_info in all_data_for_embedding:
        item = item_info['data']
        doc_type = item_info['type']

        if not isinstance(item, dict) or 'id' not in item:
            print(f"경고: 유효하지 않은 데이터 형식 또는 ID 없음 ({doc_type}). 데이터: '{str(item)[:100]}...'. 건너<0xEB><0x8B><0x88>다.")
            continue

        text_to_embed = ""
        if doc_type == 'employee':
            text_to_embed = prepare_text_for_employee_embedding(item)
        elif doc_type == 'job':
            text_to_embed = prepare_text_for_job_embedding(item)
        
        if not text_to_embed:
            print(f"경고: 임베딩 텍스트 생성 실패 ({doc_type}, ID: {item['id']}). 건너<0xEB><0x8B><0x88>다.")
            continue

        all_documents.append(text_to_embed)
        
        processed_metadata = _process_metadata_for_db(item)
        processed_metadata['doc_type'] = doc_type # 문서 타입 추가
        all_metadatas_to_db.append(processed_metadata)
        all_ids.append(item['id'])

    if not all_documents:
        print("임베딩할 유효한 문서(직원/채용공고)가 없습니다.")
        return collection

    # --- ChromaDB 배치 Upsert ---
    total_items = len(all_documents)
    num_batches = math.ceil(total_items / CHROMA_UPSERT_BATCH_SIZE)
    print(f"총 {total_items}개의 아이템을 {num_batches}개의 배치로 나누어 ChromaDB에 저장합니다 (배치 크기: {CHROMA_UPSERT_BATCH_SIZE}).")

    for i in range(num_batches):
        start_idx = i * CHROMA_UPSERT_BATCH_SIZE
        end_idx = min((i + 1) * CHROMA_UPSERT_BATCH_SIZE, total_items)
        
        batch_ids = all_ids[start_idx:end_idx]
        batch_documents = all_documents[start_idx:end_idx]
        batch_metadatas = all_metadatas_to_db[start_idx:end_idx]
        
        print(f"배치 {i+1}/{num_batches} (아이템 {start_idx+1}-{end_idx}) 임베딩 및 저장 중...")
        
        try:
            batch_embeddings = embedding_model.encode(batch_documents, convert_to_tensor=False).tolist()
            
            if batch_ids and batch_embeddings and batch_documents and batch_metadatas:
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                print(f"배치 {i+1}/{num_batches} ({len(batch_ids)}개 아이템) 저장 완료.")
            else:
                print(f"배치 {i+1}/{num_batches}에 저장할 유효 데이터가 없습니다 (임베딩 실패 또는 데이터 누락).")

        except Exception as e:
            print(f"배치 {i+1}/{num_batches} 처리 중 오류 발생: {e}")
            print(f"  오류 발생 데이터 샘플 (첫번째 ID): {batch_ids[0] if batch_ids else 'N/A'}")
            # 선택: 오류 발생 시 해당 배치 건너뛰고 계속 진행할지, 중단할지 결정
            # 여기서는 다음 배치로 계속 진행

    print(f"총 {total_items}개 중 성공적으로 처리된 아이템에 대한 ChromaDB 저장/업데이트 시도 완료.")
    return collection

if __name__ == '__main__':
    print("vector_db.py는 직접 실행용이 아닌 모듈입니다.")
