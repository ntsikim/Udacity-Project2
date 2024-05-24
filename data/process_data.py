#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them.

    Args:
    messages_filepath: str. Filepath for the messages dataset.
    categories_filepath: str. Filepath for the categories dataset.

    Returns:
    df: dataframe. Merged dataset.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    Clean the merged dataset.

    Args:
    df: dataframe. Merged dataset.

    Returns:
    df: dataframe. Cleaned dataset.
    """
    # Split the categories
    categories = df['categories'].str.split(';', expand=True)
    
    # Use the first row to extract new column names
    row = categories.iloc[0]
    category_colnames = [x.split('-')[0] for x in row]
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1].astype(int)
    
    # Drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop rows with related column values of 2
    df = df[df['related'] != 2]
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Assert statements to check data integrity
    assert df.duplicated().sum() == 0, "There are duplicate rows in the dataframe."
    assert df['related'].isin([0, 1]).all(), "There are values other than 0 and 1 in the 'related' column."
    
    return df

def save_data(df, database_filepath):
    """
    Save the clean dataset into an sqlite database.

    Args:
    df: dataframe. Cleaned dataset.
    database_filepath: str. Filepath for the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')  

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as well '\
              'as the filepath of the database to save the cleaned data to as the '\
              'third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
