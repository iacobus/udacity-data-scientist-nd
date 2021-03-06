import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Reads and combines into a DataFrame the provided CSV files

    Args:
        messages_filepath (String): Path to messages CSV file
        categories_filepath (String): Path to categories CSV file

    Returns:
        df (pandas DataFrame): DataFrame concatenating
        the provided messages and categories data
    """
    messages_df = pd.DataFrame.from_csv(messages_filepath)
    categories_df = pd.DataFrame.from_csv(categories_filepath)
    return pd.concat([messages_df, categories_df], axis=1)


def clean_data(df):
    """
    Fixes values, removes duplicates,
    and extracts categories out of DataFrame

    Args:
        df (pandas DataFrame): DataFrame to clean

    Returns:
        df (pandas dataframe): Cleaned DataFrame
    """
    # Extract column names from values, and rename
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = list(map(lambda item: item.split("-")[0], row))
    categories.columns = category_colnames

    # Fix values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))

    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Saves provided DataFrame into provided SQLite database file

    Args:
        df (pandas DataFrame): DataFrame to save to DB
        database_filename (String): Path representing DB file

    Returns:
        None
    """
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql('messages', engine, index=False)


def main():
    """
    Runs this file to load, clean, and persist to a DB
    the data provided in filenames

    Args:
        None

    Returns:
        None
    """
    if len(sys.argv) == 4:

        messages_path, categories_path, database_path = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_path, categories_path))
        df = load_data(messages_path, categories_path)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_path))
        save_data(df, database_path)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
