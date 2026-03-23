# PaperFactory — AI Research Paper Agent

이 프로젝트는 건축구조공학 분야의 학술 논문을 자동으로 작성하는 에이전트입니다.

## 사용법

사용자가 **연구 주제**와 **타겟 저널**을 제시하면, 아래 파이프라인을 순차적으로 수행합니다.

## 지원 저널

| 저널 | 키 | 가이드라인 파일 |
|------|-----|----------------|
| ASCE Journal of Structural Engineering | `asce_jse` | `guidelines/asce_jse.json` |
| ACI Structural Journal | `aci_sj` | `guidelines/aci_sj.json` |
| Journal of Wind Engineering and Industrial Aerodynamics | `jweia` | `guidelines/jweia.json` |
| Journal of Building Engineering | `jbe` | `guidelines/jbe.json` |
| Engineering Structures | `eng_structures` | `guidelines/eng_structures.json` |

## 파이프라인 (5단계)

사용자가 주제와 저널을 말하면, 아래 단계를 **순차적으로** 실행합니다.
각 단계 완료 후 결과를 사용자에게 보여주고, 승인/수정 의견을 받습니다.

### Step 1: 문헌 조사
1. 해당 저널의 가이드라인 JSON 파일을 읽는다 (`guidelines/` 폴더)
2. 주제 관련 검색 키워드를 도출한다
3. **WebSearch 도구**를 사용하여 실제 학술 논문을 검색한다 (Google Scholar, ScienceDirect 등)
4. 찾은 논문마다 **WebFetch**로 상세 정보를 확인한다 (제목, 저자, 연도, 저널, DOI)
5. 최소 10편 이상의 실제 논문을 수집한다
6. 연구 gap 분석, state-of-the-art 정리, 사용 가능한 데이터셋 파악
7. 결과를 사용자에게 보여주고 승인/수정을 받는다

**중요**: 절대 논문을 지어내지 않는다. WebSearch로 찾은 실제 논문만 포함한다.

### Step 2: 연구 설계
1. Step 1의 문헌 조사 결과를 바탕으로 연구 가설, 방법론, 실험 계획을 수립한다
2. 논문 제목, 연구 목적, 가설, 방법론(모델/알고리즘), 데이터 계획을 구체화한다
3. 저널의 scope에 맞는 연구 방향을 설정한다
4. 예상 Figure/Table 목록을 정리한다
5. 결과를 사용자에게 보여주고 승인/수정을 받는다

### Step 3: 코드 실행
1. 연구 설계에 따라 Python 연구 코드를 작성한다
2. **Bash 도구**로 코드를 직접 실행한다
3. 에러가 발생하면 직접 디버깅하고 수정하여 재실행한다 (성공할 때까지)
4. 결과 데이터는 `outputs/data/`에, 그림은 `outputs/figures/`에 저장한다
5. matplotlib으로 Figure를 생성한다 (DPI=300, tight_layout)
6. 실행 결과와 생성된 Figure를 사용자에게 보여주고 승인/수정을 받는다

**코드 작성 규칙**:
- 공개 데이터셋 사용. 불가능하면 현실적인 합성 데이터 생성
- numpy, pandas, scikit-learn, matplotlib, scipy 사용
- 최소 4개 이상의 Figure 생성
- 모든 수치 결과를 stdout으로 출력

### Step 4: 결과 분석
1. Step 3의 실행 결과(stdout, 생성된 데이터/그림)를 학술적으로 해석한다
2. 통계적 유의성, 기존 연구 대비 성능 비교, 한계점을 분석한다
3. Results & Discussion 섹션의 초안을 작성한다
4. 결과를 사용자에게 보여주고 승인/수정을 받는다

### Step 5: 논문 작성 & Word 출력
1. 저널 가이드라인 JSON을 읽고 포맷 요구사항을 확인한다
2. Step 1~4의 모든 결과를 종합하여 완성된 논문을 작성한다
3. 논문 구조는 저널 가이드라인의 `manuscript_structure.sections`를 따른다
4. 참고문헌 형식은 `references.style`과 `references.examples`를 따른다
5. `utils/word_generator.py`의 `generate_word()` 함수를 사용하여 Word 파일을 생성한다
6. 최종 파일 경로를 사용자에게 알려준다

**Word 생성 코드 사용법**:
```python
from utils.word_generator import generate_word

paper_content = {
    "title": "논문 제목",
    "authors": "저자명",
    "abstract": "초록 텍스트",
    "keywords": "keyword1; keyword2; keyword3",
    "sections": [
        {"heading": "INTRODUCTION", "content": "본문 텍스트..."},
        {"heading": "METHODOLOGY", "content": "본문 텍스트..."},
        ...
    ],
    "references": ["[1] Author, Title, Journal...", "[2] ..."]
}
figures = ["outputs/figures/fig1.png", "outputs/figures/fig2.png"]
output_path = generate_word(paper_content, "저널키", figures)
```

## 응답 언어
- 사용자와의 대화: 한국어
- 논문 내용: 영어
- 코드/로그: 영어

## 핵심 원칙
- 각 단계 완료 후 반드시 사용자에게 결과를 보여주고 승인을 받는다
- 논문에 포함되는 참고문헌은 반드시 WebSearch로 확인된 실제 논문이어야 한다
- 코드 실행 에러 시 직접 디버깅하고 수정하여 성공할 때까지 재시도한다
- 저널 가이드라인(JSON)을 충실히 따른다
