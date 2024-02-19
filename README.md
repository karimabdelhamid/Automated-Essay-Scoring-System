# Automated Essay Grading System

Welcome to the Automated Essay Grading System, an advanced tool designed to revolutionize the way essays are evaluated and graded. Leveraging Natural Language Processing (NLP) and Machine Learning (ML) technologies, this system provides an automated, efficient, and accurate method of grading essays.

## Features

- **NLP Analysis**: Utilizes Spacy, a leading NLP library, to deeply understand and analyze the structure and complexity of essays.
- **Feature Extraction**: Identifies key characteristics of the essay, such as word count, sentence count, syllable count, and more to assess the essay's quality.
- **TF-IDF Vectorization**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to evaluate the importance of words within the essay, highlighting the essay's relevance and depth.
- **Machine Learning Grading**: Uses a RandomForestRegressor, an ML model, to predict the essay score based on extracted features, simulating a complex grading process.
- **PDF Integration**: Allows for easy essay submission by reading directly from PDF files, ensuring convenience and accessibility for users.

## How to Use

Follow these steps to set up and run the Automated Essay Grading System on your machine:

1. **Preparation**:
   - Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
   - Make sure Python and pip (Python's package installer) are added to your system's PATH.

2. **Installation**:
   - Clone this repository to your local machine or download the project files.
   - Open a terminal or command prompt and navigate to the project directory.

3. **Setting Up**:
   - Install the necessary Python libraries by running the following commands one by one in your terminal:

     ```
     pip install spacy
     pip install pandas
     pip install scikit-learn
     pip install numpy
     pip install PyMuPDF
     ```

   - After installing the libraries, you need to download the Spacy English language model. Run:

     ```
     py -m spacy download en_core_web_sm
     ```

4. **Running the System**:
   - Launch the system by running the following command in your terminal:

     ```
     python automated_essay_grading_system.py
     ```

   - When prompted, enter the full path to the PDF file containing your essay. The system will then analyze the essay and display the predicted score.

Enjoy using the Automated Essay Grading System! Feel free to contribute to its development or suggest improvements.


## Advanced Technologies

This system is built on the foundation of advanced technologies and methodologies in the field of NLP and ML, making it a state-of-the-art solution for educational institutions and professionals seeking an automated approach to essay grading.

- **Spacy NLP**: For comprehensive linguistic analysis.
- **TF-IDF Vectorization**: To quantify word significance.
- **RandomForestRegressor**: For accurate prediction modeling.
- **PyMuPDF**: For seamless PDF text extraction.

## Contributing

I welcome contributions from the community. If you have suggestions for improvements or new features, please feel free to fork the repository, make changes, and submit a pull request.


Thank you for exploring the Automated Essay Grading System. 
