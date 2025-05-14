"""
작성자 : kp
작성일 : 2025-05-14
목적 : 통합 데이터 파일(직원 및 채용 공고) 로딩
내용 : 단일 JSON 파일에서 직원(HR) 데이터와 채용 공고(Job Description) 데이터를 각각 추출하여 
시스템에서 사용할 수 있도록 불러옵니다. 파일 경로 유효성, JSON 형식 유효성을 검사하고, 
지정된 키(예: 'employees', 'job_descriptions')를 통해 데이터 리스트를 추출합니다. 
오류 발생 시 적절한 메시지를 출력하며 빈 리스트를 반환하여 안정적인 데이터 처리를 지원합니다.
"""
# hr_recommender/recommender/data_loader.py

import json
import os

def _load_specific_data_from_integrated_file(file_path, data_key):
    """
    통합 JSON 파일에서 특정 키에 해당하는 데이터 리스트를 로드하는 내부 함수.
    Args:
        file_path (str): JSON 파일 경로.
        data_key (str): 데이터 리스트를 포함하는 JSON 내의 키.
    Returns:
        list: 데이터 리스트. 오류 발생 시 빈 리스트 반환.
    """
    if not os.path.exists(file_path):
        print(f"오류: 통합 데이터 파일 {file_path}을(를) 찾을 수 없습니다.")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_content = json.load(f)
        
        if isinstance(data_content, dict) and data_key in data_content:
            item_list = data_content[data_key]
            if isinstance(item_list, list):
                print(f"성공: {file_path}에서 '{data_key}' 키를 통해 {len(item_list)}개의 아이템을 로드했습니다.")
                return item_list
            else:
                print(f"오류: {file_path} 파일의 '{data_key}' 키의 값이 리스트가 아닙니다.")
                return []
        else:
            # data_key가 파일에 없는 경우
            print(f"오류: {file_path} 파일에 '{data_key}' 키가 없거나 파일 구조가 예상과 다릅니다.")
            return []

    except json.JSONDecodeError:
        print(f"오류: {file_path} 파일의 JSON 형식이 올바르지 않습니다.")
        return []
    except Exception as e:
        print(f"데이터 로드 중 오류 발생 ({file_path}, 키: {data_key}): {e}")
        return []

def load_employees_from_integrated_file(file_path):
    """통합 파일에서 HR 직원 데이터를 로드합니다."""
    return _load_specific_data_from_integrated_file(file_path, 'employees')

def load_job_descriptions_from_integrated_file(file_path):
    """통합 파일에서 채용 공고(Job Description) 데이터를 로드합니다."""
    return _load_specific_data_from_integrated_file(file_path, 'job_descriptions')


if __name__ == '__main__':
    # --- 테스트용 샘플 데이터 및 파일 생성 ---
    sample_integrated_data = {
        "employees": [
            {"id": "EMP_TEST_001", "name": "김직원", "position": "개발자", "department": "개발팀"},
            {"id": "EMP_TEST_002", "name": "이직원", "position": "디자이너", "department": "디자인팀"}
        ],
        "job_descriptions": [
            {"id": "JOB_TEST_001", "title": "백엔드 개발자", "department": "개발팀", "required_skills": ["Java", "Spring"]},
            {"id": "JOB_TEST_002", "title": "UX 디자이너", "department": "디자인팀", "required_skills": ["Figma", "UX Research"]}
        ]
    }
    integrated_test_file = 'integrated_data_temp_test.json'

    with open(integrated_test_file, 'w', encoding='utf-8') as f:
        json.dump(sample_integrated_data, f, ensure_ascii=False, indent=2)

    print("--- 직원 데이터 로드 테스트 (통합 파일) ---")
    hr_records = load_employees_from_integrated_file(integrated_test_file)
    if hr_records:
        print(f"로드된 직원 수: {len(hr_records)}")
        if hr_records: print(f"첫 번째 직원: {hr_records[0].get('name')}")
    
    print("\n--- 채용 공고 데이터 로드 테스트 (통합 파일) ---")
    job_records = load_job_descriptions_from_integrated_file(integrated_test_file)
    if job_records:
        print(f"로드된 채용 공고 수: {len(job_records)}")
        if job_records: print(f"첫 번째 채용 공고: {job_records[0].get('title')}")

    # --- 테스트 파일 삭제 ---
    os.remove(integrated_test_file)
