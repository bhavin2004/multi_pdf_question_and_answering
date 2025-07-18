# Multi PDF/ TEXT Q&A

A simple tool to ask questions across multiple PDF/ text files using Google Gemini AI.

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [How to Run Q&A](#how-to-run-qa)
- [Libraries and Tools Used](#libraries-and-tools-used)
- [License](#license)

## Setup Instructions

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Multi_pdf_reader.git
    cd Multi_pdf_reader
    ```
2. **(Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Set up Google Gemini API:**
    - Sign up for access to [Google AI Studio](https://aistudio.google.com/app/apikey) and get your API key.
    - Set your API key as an environment variable:
        ```bash
        set GOOGLE_API_KEY=your_api_key_here  # On Windows
        export GOOGLE_API_KEY=your_api_key_here  # On macOS/Linux
        ```

## How to Run Q&A

- Run the application:
    ```bash
    streamlit run app.py
    ```
- Upload your PDF/txt files in the document uploader.
- Use the interface to ask questions. The app will use Gemini AI to answer based on the content of all loaded PDFs.

## Libraries and Tools Used

- [PyPDF2](https://pypi.org/project/PyPDF2/) – PDF file reading
- [Tkinter](https://docs.python.org/3/library/tkinter.html) – GUI interface
- [google-generativeai](https://pypi.org/project/google-generativeai/) – Gemini API and embeddings

## License

This project is licensed under the MIT License.