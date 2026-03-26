# PaperFactory

PaperFactory is an AI agent that automates research paper writing for structural engineering. It runs inside Claude Code CLI and executes a 5-step pipeline — literature review, research design, code execution, result analysis, and paper writing — entirely through natural conversation. The final output is a journal-formatted Word document (or LaTeX) ready for author review and submission.

---

## Requirements

- **Claude Code CLI**
  ```bash
  npm install -g @anthropic-ai/claude-code
  ```
- **Claude subscription** — Max Plan recommended (required for extended tool use and web search)
- **Python 3.10+** with packages listed in `requirements.txt`

---

## Installation

```bash
git clone https://github.com/concrete-sangminlee/paperfactory.git
cd paperfactory
pip install -r requirements.txt
claude  # first time: will prompt for login
```

---

## Quick Start

```
$ cd paperfactory
$ claude

> "Machine learning-based prediction of wind pressure coefficients on high-rise buildings"
  주제로 JWEIA 저널에 낼 논문 써줘
```

Claude will then:

- **Step 1: Literature Review** — Searches Google Scholar and ScienceDirect for real papers on the topic. Collects at least 10 verified references with DOIs.
- **Step 2: Research Design** — Proposes a research hypothesis, methodology, experiment plan, and expected figures/tables. Waits for your approval.
- **Step 3: Code Execution** — Writes Python research code, runs it via Bash, debugs any errors automatically, and saves figures to `outputs/figures/` and data to `outputs/data/`.
- **Step 4: Result Analysis** — Interprets the numerical results academically, compares against prior work, and drafts the Results and Discussion section.
- **Step 5: Paper Writing** — Assembles the full manuscript following the journal's guidelines and exports a formatted `.docx` file to `outputs/papers/`.

At each step, Claude shows you the output and asks for approval or revision requests before moving on.

---

## Supported Journals

| Journal | Key | Field |
|---------|-----|-------|
| ASCE Journal of Structural Engineering | `asce_jse` | General Structural |
| ACI Structural Journal | `aci_sj` | Concrete |
| J. Wind Engineering and Industrial Aerodynamics | `jweia` | Wind |
| Journal of Building Engineering | `jbe` | Building |
| Engineering Structures | `eng_structures` | General Structural |
| Earthquake Engineering and Structural Dynamics | `eesd` | Seismic |
| Thin-Walled Structures | `thin_walled` | Structural |
| Cement and Concrete Composites | `cem_con_comp` | Concrete |
| Computers and Structures | `comput_struct` | Computational |
| Automation in Construction | `autom_constr` | AI + Construction |

---

## Pipeline Overview

**Step 1 — Literature Review**
Reads the journal's guideline file, derives search keywords, and uses WebSearch to find real published papers. Verifies each paper's title, authors, year, journal, and DOI. Identifies research gaps and available datasets.

**Step 2 — Research Design**
Based on the literature, defines the paper title, research objective, hypothesis, methodology, and data plan. Produces a preliminary list of figures and tables aligned with the journal's scope.

**Step 3 — Code Execution**
Writes Python code using numpy, pandas, scikit-learn, matplotlib, and scipy. Runs the code directly via Bash. Fixes errors and reruns until successful. Generates at least 4 publication-quality figures (DPI 300).

**Step 4 — Result Analysis**
Interprets computational results in an academic context: statistical significance, comparison with baselines from literature, and honest discussion of limitations. Produces a draft Results and Discussion section.

**Step 5 — Paper Writing and Word Export**
Reads the journal's formatting guidelines from `guidelines/<key>.json`. Compiles all prior outputs into a complete manuscript. Calls `utils/word_generator.py` to produce the final `.docx` file.

---

## Usage Tips

**Writing a good topic description**

| Less effective | More effective |
|----------------|----------------|
| "ML for structures" | "Gradient boosting prediction of lateral drift in RC shear walls under cyclic loading" |
| "Wind on buildings" | "CFD-validated ML model for across-wind response of super-tall buildings above 300 m" |
| "Concrete strength" | "Ensemble ML prediction of compressive strength of recycled aggregate concrete with supplementary cementitious materials" |

**Requesting revisions at any step**
After Claude presents a step's output, you can say things like:
- "Add two more references on LSTM-based structural monitoring."
- "Change the methodology to use Random Forest instead of neural networks."
- "The figures look fine. Proceed."

**Output file locations**
- Figures: `outputs/figures/`
- Data and result tables: `outputs/data/`
- Final paper: `outputs/papers/`

**LaTeX output**
By default, PaperFactory generates a Word (`.docx`) file. To get a LaTeX file instead, say: "Please generate the LaTeX version as well." Claude will use `utils/latex_generator.py`.

---

## Example Topics by Field

**Wind Engineering**
"CFD-validated machine learning model for across-wind response prediction of super-tall buildings"

**Seismic Engineering**
"Deep learning-based rapid seismic damage assessment of RC frame structures using sensor data"

**Concrete**
"Ensemble ML prediction of compressive strength of recycled aggregate concrete with fly ash and slag"

**AI and Structural Engineering**
"Physics-informed neural network for real-time structural health monitoring of cable-stayed bridges"

---

## Project Structure

```
paperfactory/
├── CLAUDE.md               # Agent instructions (pipeline definition)
├── README.md
├── requirements.txt
├── guidelines/             # Journal guidelines (10 JSON files)
│   ├── asce_jse.json
│   ├── aci_sj.json
│   ├── jweia.json
│   ├── jbe.json
│   ├── eng_structures.json
│   ├── eesd.json
│   ├── thin_walled.json
│   ├── cem_con_comp.json
│   ├── comput_struct.json
│   └── autom_constr.json
├── utils/
│   ├── word_generator.py
│   ├── latex_generator.py
│   ├── figure_utils.py
│   └── reference_utils.py
├── tests/
└── outputs/
    ├── figures/
    ├── data/
    └── papers/
```

---

## Troubleshooting

**"Claude Code not found" or `claude: command not found`**
Run `npm install -g @anthropic-ai/claude-code` and ensure your Node.js `bin` directory is in your PATH.

**"Not logged in" or authentication errors**
Run `claude login` and follow the browser prompt to authenticate with your Anthropic account.

**"Python package missing" or `ModuleNotFoundError`**
Run `pip install -r requirements.txt` from the project root. Using a virtual environment is recommended.

**WebSearch returns no results or fails**
Claude Code requires permission to use network tools. When prompted, allow web access. If the issue persists, check your Claude Code subscription plan — web search requires Max Plan or equivalent.

---

## Disclaimer

Papers generated by PaperFactory are AI-assisted drafts. They require thorough human review, verification, and refinement before submission to any journal. Always validate research results, citations, numerical claims, and conclusions independently. The authors and contributors of this project are not responsible for the accuracy or fitness of any generated content.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
