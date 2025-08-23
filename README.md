# data-science-ideas

Repository of exploratory **Data Science proof-of-concept notebooks**.  
Each notebook is designed to run in Google Colab and demonstrates a compact but practical workflow around text processing, large language models (LLMs) and dataset preparation.

---

## Table of Contents
1. [Notebooks](#notebooks)
2. [Environment & Dependencies](#environment--dependencies)
3. [Usage](#usage)
4. [Notebook Summaries](#notebook-summaries)
   - [Create legal Dataset](#create-legal-datasetipynb)
   - [Criminal Provisions – Mistral v2](#criminal_proviosions_mistralv2ipynb)
   - [Criminal Provisions – HF Hub](#criminal_proviosions-mistral-v2-huggingfaceipynb)
   - [Customer Complaint Classification](#project_customer_complaintipynb)
5. [Contributing](#contributing)
6. [License](#license)

---

## Notebooks
| Notebook | Description |
| --- | --- |
| `Create legal Dataset.ipynb` | OCR pipeline converting PDF acts into structured chapter/section text. |
| `Criminal_Proviosions_mistralv2.ipynb` | Classifies legal provisions as *Criminal* vs *Not Criminal* using a quantized Mistral‑7B. |
| `Criminal_Proviosions-mistral-v2-huggingface.ipynb` | Hugging Face–centric variant of the above classifier. |
| `project_Customer_complaint.ipynb` (`project-Customer_complaint.ipynb`) | LLM‑based complaint triage and summarization for banking products. |

---

## Environment & Dependencies
Each notebook includes installation cells, but the core runtime requirements are:

- **Python libraries**:  
  `transformers>=4.41.0`, `accelerate==0.26.1`, `bitsandbytes`, `huggingface_hub>=0.24.0`, `pandas`, `numpy`, `tqdm`, `orjson`.
- **OCR utilities** (used in `Create legal Dataset.ipynb`):  
  `tesseract-ocr`, `poppler-utils`, `pypdf`, `pdf2image`, `pytesseract`, `pyarrow`.
- **External accounts**:  
  - Google Drive (for data storage/mounting in Colab)  
  - Hugging Face account & token for accessing gated models or pushing to the Hub.

> **Note:** All notebooks are configured for Google Colab. Running locally may require additional setup for GPU support and system packages.

---

## Usage
1. Open a notebook in Google Colab using the provided badge at the top of each `.ipynb`.
2. Execute the setup cell to install dependencies.
3. Authenticate where prompted (Hugging Face token, Google Drive mounting, etc.).
4. Follow the step‑by‑step sections within each notebook.

---

## Notebook Summaries

### `Create legal Dataset.ipynb`
End‑to‑end OCR and parsing pipeline for legal PDF documents.

- **Goal**: Turn a directory of PDF acts into a structured dataset with chapters and sections.
- **Key Steps**  
  1. Install Tesseract & PDF utilities.  
  2. Mount Google Drive and specify input folder / output path.  
  3. For each PDF page:  
     - Try extracting embedded text via `pypdf`.  
     - If text is too short, fall back to Tesseract OCR via `pdf2image`.  
  4. Normalize text and assemble a page-level DataFrame.  
  5. Parse chapters/sections into columns:

    | act_title | chapter_no | chapter_title | section_no | section_heading | section_text |
    |-----------|------------|---------------|------------|-----------------|--------------|

  6. Save as `.csv` or `.parquet`.

- **Customization**: `OCR_LANG`, `OCR_DPI`, and output path variables at the top of the notebook.

---

### `Criminal_Proviosions_mistralv2.ipynb`
LLM-driven classifier identifying whether a legal provision creates a criminal offence.

- **Model**: Quantized `mistralai/Mistral-7B-Instruct-v0.2` loaded with `BitsAndBytesConfig` (4‑bit).
- **Workflow**  
  1. Install and import Hugging Face Transformers, `bitsandbytes`, etc.  
  2. Mount Google Drive for access to `acts_dataset.csv`.  
  3. Define a detailed **system prompt** outlining legal analysis steps and expected JSON output:

    ```json
    {
      "classification": "Criminal / Not Criminal / Likely Criminal / Likely Not Criminal",
      "reasoning": "...",
      "indicators": {
        "offence_creating_language": true/false,
        "criminal_punishments": true/false,
        "criminal_procedure_references": true/false,
        "statutory_context": "...",
        "state_prosecution_likely": true/false
      }
    }
    ```

  4. `generate_completion()` builds a chat prompt and uses deterministic generation (`do_sample=False`).
  5. Output is parsed JSON per provision.

- **Typical Use Cases**: Penal code analysis, compliance checks, legal-data labeling.

---

### `Criminal_Proviosions-mistral-v2-huggingface.ipynb`
Alternative version of the criminal provisions classifier emphasizing Hugging Face Hub integration (model loading, token login, etc.). The classification logic and prompt template mirror the previous notebook.

---

### `project_Customer_complaint.ipynb`
Financial product complaint classification and summarization.

- **Dataset**: 500 complaints (`train_df` 400 rows, `eval_df` 100 rows) with label `product`.  
  Derived labels:

  ```python
  LABELS = ['credit_reporting', 'credit_card',
            'debt_collection', 'mortgages_and_loans', 'retail_banking']
  ```

- **Objective**: For each complaint narrative, output structured JSON:

  ```json
  {"category": "<one_label>"}
  ```

- **Prompting**:  
  - **Zero-shot** and **few-shot** prompts supported.  
  - `few_shot_prompt_from_df()` builds labeled examples.  
  - `generate_completion()` handles chat template creation and inference.

- **Extras**:  
  - Text cleaning, sample selection utilities.  
  - Result visualizations and data summaries (displayed as HTML tables in the notebook).

---

## Contributing
1. Fork the repository.
2. Create or modify notebooks within your fork.
3. Submit a pull request with a short description of changes.

For large or incompatible changes, please open an issue first.

---

## License
No explicit license is provided. If you intend to reuse the code or datasets, confirm usage rights with the repository owner.

