# Vector Space Model Information Retrieval System

## Overview
This Streamlit application implements a Vector Space Model (VSM) for information retrieval. The system preprocesses text data, builds an inverted index, calculates TF-IDF weights, and ranks documents based on their relevance to user queries using cosine similarity. Users can input search queries, retrieve relevant documents, and explore the results interactively through a web interface.

## Features

1. **User Interface**
   - Interactive web interface powered by Streamlit.
   - Simple and intuitive design for easy querying and result visualization.

2. **Text Preprocessing**
   - Tokenization: Splits text into individual words.
   - Lowercasing: Converts text to lowercase for uniformity.
   - Stopword Removal: Removes common words using a custom stopwords file and NLTK's stopwords.
   - URL and Punctuation Filtering: Removes URLs and punctuation.
   - Stemming: Reduces words to their root forms using the Porter Stemmer.
   - Long Word Splitting: Splits long concatenated words using WordNinja.

3. **Inverted Index Creation**
   - Builds an inverted index from a collection of text documents, mapping terms to documents containing them.

4. **TF-IDF Calculation**
   - Computes Term Frequency-Inverse Document Frequency (TF-IDF) weights for each term in the document corpus.

5. **Query Processing and Ranking**
   - Processes user queries, calculates their TF-IDF weights, and ranks documents based on cosine similarity to the query.

6. **Results Display**
   - Displays retrieved documents with their relevance scores.
   - Highlights search terms within the document text for easy identification.

## Applications

1. **Information Retrieval**
   - Ideal for applications requiring efficient search capabilities over large datasets, such as document management systems, e-commerce platforms, and content management systems.

2. **Data Exploration**
   - Enables users to explore and analyze structured and unstructured data stored in text files.

3. **Educational Purposes**
   - Useful for teaching and learning about search algorithms, indexing techniques, and information retrieval systems.

## How to Use

### Libraries and Imports Used
- `os`: Interacting with the operating system (file and directory operations).
- `re`: Regular expressions for string matching and manipulation.
- `streamlit`: Creating interactive web applications.
- `nltk`: Natural language processing (tokenization, stopwords, stemming).
- `wordninja`: Splitting concatenated words.
- `collections.defaultdict` and `collections.Counter`: Specialized data structures.
- `numpy`: Numerical computations with arrays and matrices.

### Installation
1. Ensure Python and the required dependencies are installed.
2. Clone the repository from GitHub:
   ```sh
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
3. Install the required Python libraries:
   ```sh
   pip install -r requirements.txt
   ```
4. Download NLTK data:
   ```sh
   python -m nltk.downloader punkt stopwords
   ```

### Configuration
- Update the paths to your text documents and custom stopwords file in the `main` function of the script.

### Running the App
1. Open a terminal or command prompt.
2. Navigate to the directory containing the Streamlit app script (`vector-space.py`).
3. Run the app using the command:
   ```sh
   streamlit run vector-space.py
   ```
4. Access the app through the provided local URL (usually `http://localhost:8501`).

### Querying
- Enter search queries into the provided input field.
- Explore search results displayed on the web interface.
- Interact with filters and sorting options to refine results as needed.

### Visualization and Analysis
- Utilize interactive features to drill down into specific documents or categories of results.
- Monitor query performance metrics to assess system efficiency.

## Conclusion
The Vector Space Model Information Retrieval System leverages Streamlit's capabilities to create a user-friendly interface for interacting with text data. It empowers users to perform advanced searches, visualize results, and gain insights from indexed datasets efficiently. Ideal for developers, researchers, and businesses needing robust search functionalities.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries or issues, please contact saniahasan167@gmail.com.

## Acknowledgements
- [Streamlit](https://streamlit.io/)
- [NLTK](https://www.nltk.org/)
- [WordNinja](https://github.com/keredson/wordninja)

---
