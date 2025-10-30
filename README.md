Movie Recommendation System

An AI-powered Movie Recommendation System built using Python, Pandas, NumPy, and Streamlit.
The app suggests movies based on user similarity and preferences using collaborative filtering techniques.

🚀 Features

User-based and Item-based Collaborative Filtering

SVD-based Matrix Factorization model

Interactive Streamlit web interface

Clean and responsive UI

Ready for deployment on Streamlit Cloud or PythonAnywhere

🧠 Tech Stack

Python 3.10+

Pandas, NumPy, Scikit-learn

Streamlit

📂 Project Structure
MovieRecommendation/
- app.py # Streamlit application file
- ml-100k/ # Dataset folder
   - u.data
   - u.item

- requirements.txt # Python dependencies
- README.md # Project documentation

⚙️ Installation & Setup

Clone this repository

git clone https://github.com/Shahnazaqsa/MovieRecommendation.git
cd MovieRecommendation

Create & activate virtual environment

python -m venv venv
source venv/bin/activate # For Mac/Linux
venv\Scripts\activate # For Windows

Install dependencies

pip install -r requirements.txt

Run the app

streamlit run app.py

Open your browser at http://localhost:8501

🧑‍💻 Developer Info

Author: Shahnaz Aqsa
Role: Machine Learning Engineer / AI Developer
Built this project as part of an ML portfolio showcasing end-to-end recommendation systems.

