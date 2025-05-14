"""
작성자 : kp
작성일 : 2025-05-14
목적 : HR 인재 및 채용 공고 추천 시스템의 메인 실행 로직
내용 : 전체 추천 프로세스를 총괄합니다. 
설정 로드, 통합 데이터(직원, 채용 공고) 로드, 임베딩 모델 초기화, ChromaDB 설정 및 데이터 저장/로드를 수행합니다. 
사용자로부터 프로젝트 관련 정보를 입력받아 추천 모듈을 호출하고, 
최종 결과를 문서 타입(직원/채용공고)에 따라 구분하여 사용자에게 출력합니다.
"""
# hr_recommender/main.py

import chromadb
import os

try:
    import config
    from recommender.data_loader import load_employees_from_integrated_file, load_job_descriptions_from_integrated_file
    from recommender.embedding_utils import get_embedding_model
    from recommender.vector_db import setup_chromadb_collection
    from recommender.talent_recommender import recommend_talent_from_db
except ImportError as e:
    print(f"모듈 임포트 중 오류 발생: {e}")
    print("PYTHONPATH 환경 변수를 확인하거나, hr_recommender 폴더의 상위 디렉토리에서 "
          "python -m hr_recommender.main 방식으로 실행해보세요.")
    exit()


def run_recommender():
    """메인 실행 함수"""
    print("HR 인재 및 채용 공고 추천 시스템 시작")

    if not os.path.exists(config.CHROMA_DB_PATH):
        try:
            os.makedirs(config.CHROMA_DB_PATH)
            print(f"ChromaDB 저장 디렉토리 생성: {config.CHROMA_DB_PATH}")
        except OSError as e:
            print(f"ChromaDB 저장 디렉토리 생성 실패: {e}. 권한을 확인하세요.")
            return
        
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    print(f"ChromaDB 클라이언트 초기화 완료. 저장 경로: {config.CHROMA_DB_PATH}")

    # --- 데이터 로드 ---
    employee_data = load_employees_from_integrated_file(config.INTEGRATED_DATA_FILE)
    job_data = load_job_descriptions_from_integrated_file(config.INTEGRATED_DATA_FILE)

    if not employee_data and not job_data:
        print(f"{config.INTEGRATED_DATA_FILE} 에서 직원 및 채용 공고 데이터를 모두 로드할 수 없습니다. 시스템을 종료합니다.")
        return

    embedding_model = get_embedding_model(config.MODEL_NAME)
    if not embedding_model:
        print("임베딩 모델을 로드할 수 없습니다. 시스템을 종료합니다.")
        return

    try:
        hr_job_collection = setup_chromadb_collection(
            client=client,
            collection_name=config.COLLECTION_NAME,
            employee_data=employee_data,
            job_data=job_data,
            embedding_model=embedding_model
        )
        if not hr_job_collection:
            print("ChromaDB 컬렉션 준비에 실패했습니다. 시스템을 종료합니다.")
            return

        print(f"ChromaDB 컬렉션 '{hr_job_collection.name}' 준비 완료. 현재 아이템 수: {hr_job_collection.count()}")
        if hr_job_collection.count() == 0 and (employee_data or job_data):
            print("경고: 데이터 파일에 내용은 있으나, ChromaDB 컬렉션이 비어있거나 새로 생성되었습니다.")
        
    except Exception as e:
        print(f"ChromaDB 설정 중 심각한 오류 발생: {e}")
        print(f"ChromaDB 저장소({config.CHROMA_DB_PATH})에 문제가 있을 수 있습니다. 확인 후 다시 시도해 보세요.")
        return

    # --- 사용자 입력 ---
    print("\n--- 추천 검색 정보 입력 ---")
    project_description = input("검색할 프로젝트 또는 직무에 대해 설명해주세요: ")
    if not project_description.strip():
        print("설명이 입력되지 않았습니다. 기본값으로 진행합니다.")
        project_description = "소프트웨어 개발 프로젝트"

    target_department = input("특정 부서의 인력/공고를 찾으시나요? (없으면 Enter): ").strip()
    
    required_lang_input = input("필요한 언어가 있나요 (직원 검색 시)? (쉼표로 구분, 없으면 Enter): ").strip()
    required_languages_list = [lang.strip() for lang in required_lang_input.split(',') if lang.strip()] \
        if required_lang_input else None
    
    search_type_input = input("검색 대상을 선택하세요 (1: 직원, 2: 채용공고, Enter: 모두): ").strip()
    target_doc_type_param = None
    if search_type_input == '1':
        target_doc_type_param = 'employee'
    elif search_type_input == '2':
        target_doc_type_param = 'job'


    # --- 추천 실행 ---
    print("\n--- 추천 아이템(직원/채용공고) 검색 중 ---")
    if hr_job_collection.count() == 0:
        print("데이터베이스에 검색할 아이템이 없습니다. 데이터를 먼저 추가해주세요.")
        recommendations = []
    else:
        recommendations = recommend_talent_from_db(
            collection=hr_job_collection,
            embedding_model=embedding_model,
            project_description=project_description,
            num_results=config.DEFAULT_NUM_RESULTS,
            department_filter=target_department if target_department else None,
            required_languages=required_languages_list,
            target_doc_type=target_doc_type_param
        )

    # --- 결과 출력 ---
    print("\n--- 최종 추천 결과 ---")
    if recommendations:
        for i, item in enumerate(recommendations):
            similarity_score = 1 - item['distance'] if item['distance'] is not None else 'N/A'
            score_display = f"{similarity_score:.4f}" if isinstance(similarity_score, float) else similarity_score
            
            doc_type_display = "직원" if item['doc_type'] == 'employee' else "채용공고" if item['doc_type'] == 'job' else "알 수 없음"

            print(f"\n추천 {i+1}: [{doc_type_display}] {item['name_or_title']} (부서: {item['department']}) - (유사도 점수: {score_display})")
            
            if item['doc_type'] == 'employee':
                print(f"  직무: {item.get('position', 'N/A')}")
                print(f"  프로필 요약: {item.get('profile_or_description', 'N/A')}")
                print(f"  보유 기술: {item.get('skills_info', 'N/A')}")
                if item.get('languages'): print(f"  사용 언어: {item.get('languages')}")
            elif item['doc_type'] == 'job':
                print(f"  근무지: {item.get('location', 'N/A')}, 고용형태: {item.get('employment_type', 'N/A')}")
                print(f"  경력: {item.get('experience_years', 'N/A')}")
                print(f"  필수 기술: {item.get('skills_info', 'N/A')}")
                print(f"  상세 설명: {item.get('profile_or_description', 'N/A')}")

            if item['reasoning']:
                print("  추천 이유:")
                for reason in item['reasoning']:
                    print(f"    - {reason}")
            # print(f"  (내부 ID: {item['id']}, 거리: {item['distance']:.4f})") 
    else:
        print("입력하신 조건에 맞는 아이템(직원/채용공고)을 찾지 못했습니다.")
        if hr_job_collection.count() > 0 :
            print("팁: 검색 설명을 더 자세히 작성하거나 필터 조건을 완화해보세요.")

if __name__ == '__main__':
    run_recommender()
