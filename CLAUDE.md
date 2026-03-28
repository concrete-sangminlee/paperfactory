# PaperFactory -- AI Research Paper Agent

AI agent that automatically writes civil engineering research papers: user provides a **topic** + **target journal**, and the agent executes a 5-step pipeline from literature review to final manuscript export.

## Supported Fields

구조공학을 중심으로, 아래 토목공학 세부 분야를 지원합니다:
- **구조공학**: RC 구조, 강구조, 합성구조, 내진설계, 풍공학
- **건설관리/자동화**: AI 기반 건설 자동화, 스마트 건설
- **건축재료**: 콘크리트, 강재, 복합재, 지속가능 건축자재
- **구조 신뢰성**: 확률론적 안전성 평가, 리스크 분석
- **건물공학**: 에너지 효율, 실내환경, 건물 성능

## Language Support

- **English papers**: 기본 모드 — 모든 저널 지원
- **한국어 논문**: KCI 저널 (KSCE 등) 대상, 한국어 본문 + 영문 초록 형태 지원
- 사용자와의 대화는 항상 **한국어**, 논문 내용은 저널 언어에 맞춤

## Usage

사용자가 **연구 주제**와 **타겟 저널**을 제시하면, 아래 5단계 파이프라인을 순차적으로 수행합니다.

## Supported Journals

| Journal | Key | Guideline File |
|---------|-----|----------------|
| ASCE Journal of Structural Engineering | `asce_jse` | `guidelines/asce_jse.json` |
| ACI Structural Journal | `aci_sj` | `guidelines/aci_sj.json` |
| Journal of Wind Engineering and Industrial Aerodynamics | `jweia` | `guidelines/jweia.json` |
| Journal of Building Engineering | `jbe` | `guidelines/jbe.json` |
| Engineering Structures | `eng_structures` | `guidelines/eng_structures.json` |
| Earthquake Engineering & Structural Dynamics | `eesd` | `guidelines/eesd.json` |
| Thin-Walled Structures | `thin_walled` | `guidelines/thin_walled.json` |
| Cement and Concrete Composites | `cem_con_comp` | `guidelines/cem_con_comp.json` |
| Computers & Structures | `comput_struct` | `guidelines/comput_struct.json` |
| Automation in Construction | `autom_constr` | `guidelines/autom_constr.json` |
| Structural Safety | `struct_safety` | `guidelines/struct_safety.json` |
| Construction and Building Materials | `const_build_mat` | `guidelines/const_build_mat.json` |
| KSCE Journal of Civil Engineering | `ksce_jce` | `guidelines/ksce_jce.json` |
| Buildings (MDPI) | `buildings_mdpi` | `guidelines/buildings_mdpi.json` |
| J. Constructional Steel Research | `steel_comp_struct` | `guidelines/steel_comp_struct.json` |

---

## Pipeline (5 Steps)

사용자가 주제와 저널을 말하면, 아래 단계를 **순차적으로** 실행합니다.
각 단계 완료 후 결과를 사용자에게 보여주고 **승인/수정 의견**을 받은 뒤 다음 단계로 진행합니다.

---

### Step 1: Literature Review (문헌 조사)

**Procedure**
1. 해당 저널의 가이드라인 JSON 파일을 읽는다 (`guidelines/` 폴더).
2. 주제 관련 검색 키워드를 여러 조합으로 도출한다.
3. **WebSearch** 도구를 사용하여 실제 학술 논문을 검색한다 (Google Scholar, ScienceDirect, Scopus 등).
4. 찾은 논문마다 **WebFetch**로 상세 정보를 확인한다 (제목, 저자, 연도, 저널, DOI).
5. 최소 **15편** 이상의 실제 논문을 수집한다.
6. 연구 gap 분석, state-of-the-art 정리, 사용 가능한 데이터셋을 파악한다.
7. 결과를 사용자에게 보여주고 승인/수정을 받는다.

**Quality Criteria**
- 최소 15편 이상의 논문 수집
- 수집 논문의 50% 이상이 최근 5년 이내 출판
- 모든 논문에 DOI 포함 (가능한 경우)
- **절대 논문을 지어내지 않는다** -- WebSearch로 찾은 실제 논문만 포함

**Failure Handling**
- 검색 결과가 15편 미만이면 키워드를 수정/확장하여 재검색한다.
- 3회 재검색 후에도 부족하면 관련 분야로 범위를 넓힌다.

---

### Step 2: Research Design (연구 설계)

**Procedure**
1. Step 1의 문헌 조사 결과를 바탕으로 연구 가설, 방법론, 실험 계획을 수립한다.
2. 논문 제목, 연구 목적, 가설, 방법론(모델/알고리즘), 데이터 계획을 구체화한다.
3. 저널의 scope에 맞는 연구 방향을 설정한다.
4. 예상 Figure(최소 6개) / Table(최소 3개) 목록을 정리한다.
5. 결과를 사용자에게 보여주고 승인/수정을 받는다.

**Quality Criteria**
- 명확한 novelty statement 포함 (기존 연구 대비 차별점)
- 최소 6개의 Figure 계획
- 최소 3개의 Table 계획
- 저널 scope와의 적합성 확인

**Failure Handling**
- Novelty가 약하다고 판단되면 대안적 연구 각도를 제시한다.
- 사용자와 협의하여 연구 방향을 조정한다.

---

### Step 3: Code Execution (코드 실행)

**Procedure**
1. 연구 설계에 따라 Python 연구 코드를 작성한다.
2. `utils/figure_utils.py`의 `setup_style()`을 스크립트 상단에서 호출한다.
3. **Bash** 도구로 코드를 직접 실행한다.
4. 에러가 발생하면 직접 디버깅하고 수정하여 재실행한다.
5. 결과 데이터는 `outputs/data/`에, 그림은 `outputs/figures/`에 저장한다.
6. `figure_utils.save_figure()`로 Figure를 저장한다 (DPI=300, tight_layout 자동 적용).
7. 실행 결과와 생성된 Figure를 사용자에게 보여주고 승인/수정을 받는다.

**Allowed Packages**
- 기본: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`
- ML/DL: `xgboost`, `lightgbm`, `tensorflow`, `keras`, `pytorch`
- 시각화: `plotly`
- 구조해석: `openseespy`
- 신호처리: `pywt`

**Code Rules**
- 공개 데이터셋 사용. 불가능하면 현실적인 합성 데이터를 생성한다.
- 모든 Figure는 `utils/figure_utils.py`를 통해 생성한다 (일관된 스타일 보장).
- 최소 **6개** 이상의 Figure를 생성한다.
- 재현성을 위해 모든 난수에 `random_seed`를 고정한다 (예: `np.random.seed(42)`).
- 모든 수치 결과를 stdout으로 출력한다.

**Failure Handling**
- 에러 시 최대 **5회** 디버깅/재실행을 시도한다.
- 5회 시도 후에도 실패하면 에러 내용과 시도한 해결 방법을 사용자에게 보고한다.

---

### Step 4: Result Analysis (결과 분석)

**Procedure**
1. Step 3의 실행 결과(stdout, 생성된 데이터/그림)를 학술적으로 해석한다.
2. 통계적 유의성을 검증한다 (p-value, confidence interval 등).
3. 기존 연구 대비 성능/결과를 비교 분석한다 (최소 3건).
4. 연구의 한계점을 식별한다 (최소 3개).
5. Results & Discussion 섹션의 초안을 작성한다.
6. 결과를 사용자에게 보여주고 승인/수정을 받는다.

**Quality Criteria**
- 모든 주장에 수치적 근거 포함 (numbers backing claims)
- 기존 연구와 최소 **3건** 이상 비교
- 최소 **3개** 이상의 한계점 명시

**Failure Handling**
- 결과가 가설과 크게 다를 경우 원인 분석을 수행한다.
- 필요시 Step 3로 돌아가 코드/파라미터를 수정한다.

---

### Step 5: Paper Writing & Export (논문 작성 및 출력)

**Procedure**
1. 저널 가이드라인 JSON을 읽고 포맷 요구사항을 확인한다.
2. Step 1~4의 모든 결과를 종합하여 완성된 논문을 작성한다.
3. 논문 구조는 저널 가이드라인의 `manuscript_structure.sections`를 따른다.
4. 참고문헌 형식은 `references.style`과 `references.examples`를 따른다.
5. `utils/reference_utils.py`로 참고문헌 형식을 검증한다.
6. `utils/word_generator.py`의 `generate_word()`로 Word 파일을 생성한다 (기본 출력).
7. 사용자가 LaTeX를 요청하면 `utils/latex_generator.py`의 `generate_latex()`로 출력한다.
8. 최종 파일 경로를 사용자에게 알려준다.

**Quality Criteria**
- Abstract: 저널별 단어 수 제한 준수
- 모든 필수 섹션 포함 (저널 가이드라인 기준)
- 모든 참고문헌이 WebSearch로 검증된 실제 논문
- 본문 최소 **6,000 단어** 이상

**Output Format**
- 기본: Word (.docx) -- `utils/word_generator.py`
- 선택: LaTeX (.tex + .bib) -- `utils/latex_generator.py`
- 선택: PDF (.pdf) -- `utils/pdf_generator.py`

**Post-Generation**
- `utils/quality_checker.py`의 `check_paper()`로 품질 검증을 실행한다.
- `utils/submission_utils.py`의 `submission_checklist()`로 투고 준비 상태를 확인한다.
- 저널이 graphical abstract를 요구하면, PaperBanana `generate_diagram`으로 생성한다.
- `generate_cover_letter()`로 에디터 커버레터를 자동 생성한다.

**Failure Handling**
- Word/LaTeX/PDF 생성 에러 시 에러 로그를 확인하고 paper_content 구조를 수정하여 재시도한다.
- 참고문헌 검증 실패 시 해당 항목을 수정하거나 제외한다.
- Reject 후 다른 저널 재투고 시, `reformat_paper()`로 포맷을 변환한다.

---

## Response Language Rules (응답 언어)

- 사용자와의 대화: **한국어**
- 논문 내용: **English**
- 코드/로그: **English**

---

## Core Principles (핵심 원칙)

1. **단계별 승인**: 각 단계 완료 후 반드시 결과를 사용자에게 보여주고 승인을 받은 뒤 다음 단계로 진행한다.
2. **실제 참고문헌만**: 논문에 포함되는 참고문헌은 반드시 WebSearch로 확인된 실제 논문이어야 한다 -- 절대 지어내지 않는다.
3. **디버깅 후 재시도**: 코드 실행 에러 시 직접 디버깅하고 수정하여 성공할 때까지 재시도한다 (최대 5회). 5회 초과 시 사용자에게 보고한다.
4. **저널 가이드라인 준수**: 저널 가이드라인 JSON 파일의 포맷, 구조, 참고문헌 스타일을 충실히 따른다.
5. **일관된 Figure 품질**: 모든 Figure는 `utils/figure_utils.py`를 통해 생성하여 일관된 학술 품질을 보장한다.
6. **PaperBanana 다이어그램**: 논문의 methodology overview diagram이 필요할 때는 PaperBanana MCP 도구(`generate_diagram`)를 사용한다.

---

## PaperBanana Integration (학술 다이어그램 생성)

PaperBanana는 MCP 서버로 연결되어 있으며, 논문에 필요한 고품질 학술 다이어그램을 AI로 자동 생성한다.

### 사용 시점
- **Step 2 (연구 설계)**: methodology overview diagram을 계획한다.
- **Step 3 (코드 실행)**: 데이터 기반 statistical plot이 필요할 때 `generate_plot`을 사용한다.
- **Step 5 (논문 작성)**: methodology 섹션에 들어갈 framework diagram을 `generate_diagram`으로 생성한다.

### MCP 도구

1. **`generate_diagram`** -- Methodology / framework diagram 생성
   - `source_context`: 방법론 설명 텍스트
   - `communicative_intent`: 그림 캡션 / 목적
   - `optimize`: true (입력 최적화)
   - 결과: 고품질 PNG 이미지

2. **`generate_plot`** -- Statistical plot 생성
   - `data`: CSV/JSON 데이터 경로
   - `intent`: 플롯 목적 설명
   - 결과: 학술 품질 통계 플롯

3. **`evaluate_diagram`** -- 생성된 다이어그램 품질 평가
   - `generated`: 생성된 이미지
   - `reference`: 참고 이미지
   - 4개 기준 평가: Faithfulness, Readability, Conciseness, Aesthetics

### 사용 예시

Step 5에서 methodology diagram 생성:
```
generate_diagram 도구를 사용하여:
- source_context: "The proposed framework consists of three modules: (1) data preprocessing with wavelet decomposition, (2) feature extraction using CNN encoder, (3) prediction via gradient boosting ensemble..."
- communicative_intent: "Overview of the proposed hybrid CNN-GBM framework for wind pressure prediction"
- optimize: true
```

생성된 다이어그램은 `outputs/figures/`에 저장하고 논문 Figure로 포함한다.

---

## Utility Code Usage Examples

### Word Generator (`utils/word_generator.py`)

```python
from utils.word_generator import generate_word

paper_content = {
    "title": "Paper Title",
    "authors": "Author Names",
    "abstract": "Abstract text",
    "keywords": "keyword1; keyword2; keyword3",
    "sections": [
        {"heading": "INTRODUCTION", "content": "Body text...", "subsections": [
            {"heading": "Background", "content": "Sub-section text..."},
        ]},
        {"heading": "METHODOLOGY", "content": "Body text..."},
        {"heading": "RESULTS AND DISCUSSION", "content": "Body text..."},
        {"heading": "CONCLUSIONS", "content": "Body text..."},
    ],
    "tables": [
        {
            "caption": "Table 1. Description.",
            "headers": ["Col A", "Col B"],
            "rows": [["1", "2"], ["3", "4"]],
        }
    ],
    "references": ["[1] Author, Title, Journal...", "[2] ..."],
    "figure_captions": ["Fig. 1. Description.", "Fig. 2. Description."],
    "practical_applications": "For ASCE journals only...",
    "highlights": ["Highlight 1", "Highlight 2"],
    "data_availability": "Data available on request.",
    "acknowledgments": "Acknowledgment text.",
    "notation": [{"symbol": "f_c", "definition": "compressive strength of concrete"}],
}
figures = ["outputs/figures/fig1.png", "outputs/figures/fig2.png"]
output_path = generate_word(paper_content, "journal_key", figures)
```

### LaTeX Generator (`utils/latex_generator.py`)

```python
from utils.latex_generator import generate_latex

tex_path, bib_path = generate_latex(paper_content, "journal_key", figures)
```

### Figure Utils (`utils/figure_utils.py`)

```python
from utils.figure_utils import setup_style, save_figure, get_colors, get_figsize

setup_style()                  # Call once at the top of the research script
colors = get_colors()          # 8 colors, B&W print safe
colors = get_colors("muted")   # Alternative palette
w, h = get_figsize("single")   # 3.5 x 2.8 inches (single column)
w, h = get_figsize("double")   # 7.0 x 4.5 inches (double column)

fig, ax = plt.subplots(figsize=get_figsize("single"))
ax.plot(x, y, color=colors[0])
save_figure(fig, "fig_1_model_comparison")   # saves to outputs/figures/
```

### Reference Utils (`utils/reference_utils.py`)

```python
from utils.reference_utils import validate_references, check_duplicates

issues = validate_references(references_list, guideline)  # format issues per ref
duplicates = check_duplicates(references_list)             # duplicate detection
```

### Quality Checker (`utils/quality_checker.py`)

```python
from utils.quality_checker import check_paper

result = check_paper(paper_content, "jweia", figures=figure_paths)
print(result["summary"])    # Quality Score: 100/100, Status: PASS
print(result["passed"])     # True/False (all critical checks pass)
print(result["score"])      # 0-100
print(result["checks"])     # list of individual check results
```

Validates: abstract word count (journal limit), body word count (min 6,000), required sections, reference count (min 15), recent reference ratio (min 50%), figure count (min 6), table count (min 3), keywords, highlights (if required), title, data availability.
