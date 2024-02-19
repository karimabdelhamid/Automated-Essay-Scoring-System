# Automated Essay Grading System

Welcome to the Automated Essay Grading System, an advanced tool designed to revolutionize the way essays are evaluated. Leveraging cutting-edge Natural Language Processing (NLP) and Machine Learning (ML) technologies, this system provides an automated, efficient, and accurate method of grading essays.

## Features

- **NLP Analysis**: Utilizes Spacy, a leading NLP library, to deeply understand and analyze the structure and complexity of essays.
- **Feature Extraction**: Identifies key characteristics of the essay, such as word count, sentence count, syllable count, and more, to assess the essay's quality.
- **TF-IDF Vectorization**: Employs TF-IDF (Term Frequency-Inverse Document Frequency) to evaluate the importance of words within the essay, highlighting the essay's relevance and depth.
- **Machine Learning Grading**: Uses a RandomForestRegressor, a robust ML model, to predict the essay score based on extracted features, simulating a nuanced grading process.
- **PDF Integration**: Allows for easy essay submission by reading directly from PDF files, ensuring convenience and accessibility for users.

## How to Use

1. **Preparation**: Ensure you have Python installed on your system along with the required libraries (`spacy`, `pandas`, `sklearn`, `numpy`, `PyMuPDF`).

2. **Installation**: Clone this repository to your local machine and navigate to the project directory.

3. **Setting Up**: Run `pip install -r requirements.txt` to install all necessary dependencies.

4. **Running the System**:
   - Launch the system by running `python automated_essay_grading_system.py` in your terminal.
   - When prompted, enter the full path to the PDF file containing your essay.
   - The system will analyze the essay and display the predicted score.

## Advanced Technologies

This system is built on the foundation of advanced technologies and methodologies in the field of NLP and ML, making it a state-of-the-art solution for educational institutions and professionals seeking an automated approach to essay grading.

- **Spacy NLP**: For comprehensive linguistic analysis.
- **TF-IDF Vectorization**: To quantify word significance.
- **RandomForestRegressor**: For accurate prediction modeling.
- **PyMuPDF**: For seamless PDF text extraction.

## Contributing

We welcome contributions from the community. If you have suggestions for improvements or new features, please feel free to fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Thank you for exploring the Automated Essay Grading System. We believe in harnessing the power of technology to enhance educational processes and outcomes.
