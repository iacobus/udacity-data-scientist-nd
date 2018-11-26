# Udacity's Data Scientist Nanodegree

## Project: Disaster Response Pipelines

- Use NLTK to classify disaster text messages in several categories
- Use Scikit Pipeline to structure prep & learning tasks
- Build Flask web UI to enter new message and classify it

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Important Files:
- data/process_data.py: Script that accepts two CSV files (one for categories, one for messages) and loads their data, cleans it, and dumps it into a local database.
- models/train_classifier.py: Script that builds an NLP classifier to predict the category of a given disaster message.
- app/templates/*.html: HTML templates for the web app on top of the model. You can use this app to classify new messages.
- run.py: Boot the web app server.
