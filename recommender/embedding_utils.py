"""
작성자 : kp
작성일 : 2025-05-14
목적 : 텍스트 임베딩 관련 유틸리티 제공
내용 : Hugging Face의 Sentence Transformer 모델을 로드하고, 
직원 정보 및 채용 공고 정보를 임베딩 생성을 위한 단일 텍스트 문자열로 가공하는 기능을 수행합니다. 
모델 로드 실패 시 오류 처리를 포함합니다.
"""
# hr_recommender/recommender/embedding_utils.py

from sentence_transformers import SentenceTransformer

def get_embedding_model(model_name):
    """
    지정된 이름의 Sentence Transformer 모델을 로드합니다.
    Args:
        model_name (str): Hugging Face 모델 이름 또는 로컬 경로
    Returns:
        SentenceTransformer: 로드된 모델 객체. 오류 시 None 반환.
    """
    print(f"Sentence Transformer 모델 ({model_name}) 로드 중...")
    try:
        model = SentenceTransformer(model_name)
        print("모델 로드 완료.")
        return model
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        print("Hugging Face 모델 다운로드에 실패했을 수 있습니다. 인터넷 연결을 확인하거나 모델 이름을 확인하세요.")
        return None

def prepare_text_for_employee_embedding(employee):
    """
    임베딩을 위해 직원(employee) 정보를 하나의 문자열로 결합합니다.
    Args:
        employee (dict): 직원 정보 딕셔너리
    Returns:
        str: 임베딩을 위해 결합된 텍스트 문자열
    """
    education_info = employee.get('education', {})
    skills_str = ", ".join(employee.get('skills', []))
    projects_str = ". ".join(employee.get('projects', []))
    languages_str = ", ".join(employee.get('languages', []))
    education_details = f"{education_info.get('degree', '')} {education_info.get('school', '')} ({education_info.get('graduation_year', 'N/A')})"

    text_for_embedding = f"직원 유형. 프로필: {employee.get('profile_summary', '')}. " \
                         f"직무: {employee.get('position', '')}. " \
                         f"부서: {employee.get('department', '')}. " \
                         f"보유 기술: {skills_str}. " \
                         f"수행 프로젝트: {projects_str}. " \
                         f"학력: {education_details}. " \
                         f"사용 언어: {languages_str}."
    return text_for_embedding.strip()

def prepare_text_for_job_embedding(job):
    """
    임베딩을 위해 채용 공고(job_description) 정보를 하나의 문자열로 결합합니다.
    Args:
        job (dict): 채용 공고 정보 딕셔너리
    Returns:
        str: 임베딩을 위해 결합된 텍스트 문자열
    """
    req_skills_str = ", ".join(job.get('required_skills', []))
    pref_skills_str = ", ".join(job.get('preferred_skills', []))
    responsibilities_str = ". ".join(job.get('responsibilities', []))

    text_for_embedding = f"채용 공고 유형. 공고명: {job.get('title', '')}. " \
                         f"부서: {job.get('department', '')}. " \
                         f"근무지: {job.get('location', '')}. " \
                         f"고용 형태: {job.get('employment_type', '')}. " \
                         f"필수 기술: {req_skills_str}. " \
                         f"우대 기술: {pref_skills_str}. " \
                         f"경력: {job.get('experience_years', '')}. " \
                         f"학력 조건: {job.get('education', '')}. " \
                         f"주요 업무: {responsibilities_str}. " \
                         f"상세 설명: {job.get('description', '')}."
    return text_for_embedding.strip()


if __name__ == '__main__':
    # --- 테스트용 코드 ---
    try:
        # config 모듈이 상위 디렉토리에 있다고 가정
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from config import MODEL_NAME
    except ImportError:
        print("경고: config 모듈을 찾을 수 없습니다. 테스트를 위해 기본 MODEL_NAME을 사용합니다.")
        MODEL_NAME = 'all-MiniLM-L6-v2'
    
    test_model = get_embedding_model(MODEL_NAME)
    if test_model:
        print(f"{MODEL_NAME} 모델이 성공적으로 로드되었습니다.")

    sample_employee = {
        "id": "EMP00001", "name": "홍길동", "position": "시니어 개발자", "department": "R&D팀",
        "skills": ["Python", "Django", "AWS"], "projects": ["신규 서비스 개발", "레거시 시스템 개선"],
        "education": {"degree": "컴퓨터공학 석사", "school": "한국대학교", "graduation_year": 2018},
        "languages": ["한국어(원어민)", "영어(업무 가능)"],
        "profile_summary": "다양한 웹 서비스 개발 경험을 가진 개발자입니다."
    }
    emp_text = prepare_text_for_employee_embedding(sample_employee)
    print(f"\n직원 임베딩용 텍스트 예시:\n{emp_text}")

    sample_job = {
        "id": "JOB001", "title": "풀스택 개발자", "department": "IT 개발팀", "location": "부산",
        "employment_type": "프리랜서", "required_skills": ["Firebase", "TypeScript", "Flask"],
        "preferred_skills": ["C#", "Flutter"], "experience_years": "5년 이상", "education": "학사 이상",
        "responsibilities": ["API 개발 아키텍처 분석", "인증 시스템 환경 표준화"],
        "description": "IT 개발팀에서 함께할 5년 이상 풀스택 개발자을(를) 찾고 있습니다."
    }
    job_text = prepare_text_for_job_embedding(sample_job)
    print(f"\n채용 공고 임베딩용 텍스트 예시:\n{job_text}")
