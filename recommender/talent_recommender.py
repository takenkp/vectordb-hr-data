"""
작성자 : kp
작성일 : 2025-05-14
목적 : 핵심 인재 및 채용 공고 추천 로직 수행
내용 : 프로젝트 설명을 임베딩하여 ChromaDB에서 유사한 직원 프로필 및 채용 공고를 검색합니다.
 검색 결과의 'doc_type' 메타데이터를 통해 문서 종류를 식별합니다. 
 이후, 부서, 프로젝트 경험, 언어 능력 등의 추가 필터링 기준(검색 증강)을 적용하여 최종 후보를 선정하고, 
 추천 이유와 함께 정렬된 결과를 반환합니다.
"""
# hr_recommender/recommender/talent_recommender.py

def recommend_talent_from_db(collection, embedding_model, project_description, num_results=3, department_filter=None, required_languages=None, target_doc_type=None):
    """
    프로젝트 설명을 기반으로 ChromaDB에서 유사한 직원 및/또는 채용 공고를 검색하고 필터링하여 추천합니다.
    Args:
        collection (chromadb.Collection): 검색할 ChromaDB 컬렉션.
        embedding_model (SentenceTransformer): 프로젝트 설명 임베딩용 모델.
        project_description (str): 프로젝트 설명.
        num_results (int): 반환할 추천 아이템 수.
        department_filter (str, optional): 필터링할 부서 이름.
        required_languages (list, optional): (직원 대상) 필요한 언어 목록.
        target_doc_type (str, optional): 'employee', 'job', 또는 None (모두). 특정 타입의 문서만 검색.
    Returns:
        list: 추천된 아이템 정보 딕셔너리의 리스트. 각 아이템은 'doc_type' 포함.
    """
    project_embedding = embedding_model.encode([project_description], convert_to_tensor=False).tolist()

    initial_query_count = num_results * 5 if num_results * 5 > 10 else 20 # 더 많은 결과 요청

    # ChromaDB의 where 필터를 사용하여 doc_type 필터링 (지원하는 경우)
    where_filter = {}
    if target_doc_type and target_doc_type in ['employee', 'job']:
        where_filter = {"doc_type": target_doc_type}
        print(f"'{target_doc_type}' 타입의 문서만 검색합니다.")

    try:
        query_results = collection.query(
            query_embeddings=project_embedding,
            n_results=initial_query_count,
            where=where_filter if where_filter else None, # where 필터 적용
            include=['metadatas', 'documents', 'distances']
        )
    except Exception as e:
        print(f"ChromaDB 쿼리 중 오류: {e}")
        # where 필터가 지원되지 않거나 다른 문제일 수 있음. 필터 없이 재시도.
        if where_filter:
            print("Warning: Where 필터 적용 실패. 필터 없이 모든 타입 문서 검색 시도.")
            query_results = collection.query(
                query_embeddings=project_embedding,
                n_results=initial_query_count,
                include=['metadatas', 'documents', 'distances']
            )
        else:
            return []


    if not query_results['ids'] or not query_results['ids'][0]:
        print("유사한 아이템(직원/채용공고)을 찾지 못했습니다.")
        return []

    candidates = []
    for i in range(len(query_results['ids'][0])):
        metadata = query_results['metadatas'][0][i]
        doc_type = metadata.get('doc_type', 'unknown') # doc_type 가져오기

        # target_doc_type이 설정되었고, where 필터가 작동 안했을 경우 여기서 한번 더 필터링
        if target_doc_type and target_doc_type != doc_type and not where_filter: # where_filter가 없었다는 것은 위에서 필터 없이 재시도했다는 의미
            continue

        candidate = {
            'id': query_results['ids'][0][i],
            'doc_type': doc_type, # 문서 타입 추가
            'distance': query_results['distances'][0][i],
            'reasoning': [],
            # 공통 필드 및 타입별 필드 추가
            'name_or_title': metadata.get('name') if doc_type == 'employee' else metadata.get('title', 'N/A'),
            'department': metadata.get('department', 'N/A'),
            'profile_or_description': metadata.get('profile_summary') if doc_type == 'employee' else metadata.get('description', 'N/A'),
            'skills_info': metadata.get('skills') if doc_type == 'employee' else metadata.get('required_skills', 'N/A') # 문자열로 변환된 상태
        }
        
        # 직원 특화 정보 (메타데이터에서 원본 필드명으로 접근)
        if doc_type == 'employee':
            candidate['position'] = metadata.get('position', 'N/A')
            candidate['projects'] = metadata.get('projects', "") # 문자열로 변환된 상태
            candidate['languages'] = metadata.get('languages', "") # 문자열로 변환된 상태
        # 채용 공고 특화 정보
        elif doc_type == 'job':
            candidate['location'] = metadata.get('location', 'N/A')
            candidate['employment_type'] = metadata.get('employment_type', 'N/A')
            candidate['experience_years'] = metadata.get('experience_years', 'N/A')
            candidate['responsibilities'] = metadata.get('responsibilities', "") # 문자열로 변환된 상태

        candidates.append(candidate)

    # --- 필터링 (검색 증강) ---
    # 부서 필터 (공통)
    if department_filter:
        filtered_candidates = []
        for c in candidates:
            if c['department'].lower() == department_filter.lower():
                c['reasoning'].append(f"부서 일치: {c['department']}")
                filtered_candidates.append(c)
        candidates = filtered_candidates
        print(f"부서 필터링 후 후보 수: {len(candidates)}")

    # 프로젝트 키워드 매칭 (직원: projects, 채용공고: responsibilities 또는 description)
    project_keywords = [kw.strip().lower() for kw in project_description.split() if len(kw.strip()) > 2]
    if project_keywords:
        for candidate in candidates:
            text_to_search_in = ""
            if candidate['doc_type'] == 'employee':
                text_to_search_in = candidate.get('projects', '').lower()
            elif candidate['doc_type'] == 'job':
                text_to_search_in = (candidate.get('responsibilities', '') + " " + candidate.get('profile_or_description', '')).lower()
            
            matched_keywords_count = sum(1 for keyword in project_keywords if keyword in text_to_search_in)
            if matched_keywords_count > 0:
                candidate['reasoning'].append(f"프로젝트/업무 관련 키워드 {matched_keywords_count}개 매칭")


    # 언어 필터 (직원 대상)
    if required_languages:
        filtered_by_lang = []
        for candidate in candidates:
            if candidate['doc_type'] == 'employee':
                # 메타데이터의 languages는 이미 문자열로 변환되어 있음 ("언어1, 언어2")
                candidate_langs_str_lower = candidate.get('languages', "").lower()
                all_langs_met = True
                matched_langs_display = []
                for req_lang in required_languages:
                    req_lang_lower = req_lang.lower()
                    if req_lang_lower not in candidate_langs_str_lower:
                        all_langs_met = False
                        break
                    # 표시용 언어 찾기 (정확한 매칭은 어려우므로, 요구 언어 자체를 사용)
                    matched_langs_display.append(req_lang) 
                
                if all_langs_met:
                    candidate['reasoning'].append(f"요구 언어 충족: {', '.join(matched_langs_display)}")
                    filtered_by_lang.append(candidate)
            else: # 채용 공고는 언어 필터 적용 안 함 (필요시 추가 가능)
                filtered_by_lang.append(candidate)
        candidates = filtered_by_lang
        print(f"언어 필터링 후 후보 수: {len(candidates)}")


    candidates.sort(key=lambda c: (-len(c['reasoning']), c['distance']))

    return candidates[:num_results]

if __name__ == '__main__':
    print("talent_recommender.py는 직접 실행용이 아닌 모듈입니다.")
