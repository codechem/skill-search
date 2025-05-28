## Skill Search CV Recommendation System
This project is a Skill Search CV Recommendation System that utilizes a combination of machine learning and natural language processing techniques to recommend candidates based on their skills and job requirements.
### Features
- **Skill Extraction**: Extracts skills from job descriptions and resumes using NLP techniques.
- **Skill Matching**: Matches candidates to job requirements based on extracted skills.
- **Ranking**: Ranks candidates based on their skill match score.

### Setup
1. **Environment Setup**: Ensure you have Python 3.8 or higher installed on your machine.
2. **Enviroment variables**: Set the following environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key for using the GPT model.

### How to use
1. **Install Dependencies**: Make sure to install dependencies with poetry.
   ```bash
   poetry install
   ```
2. **Run the Application**: Start the application using the following command.
   ```bash
    poetry run streamlit run skill_search/web_ui.py
    ```
3. **Access the Web UI**: Open your web browser and navigate to `http://localhost:8501` to access the Skill Search CV Recommendation System.
