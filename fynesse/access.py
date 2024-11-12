from .config import *
import requests
import pymysql
import csv
import time
import zipfile
import io
import pandas as pd
from sqlalchemy import create_engine

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

def hello_world():
  print("Hello from the data science library!")

def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)

def upload_zip_file(url, csv_file_name):
    response = requests.get(url)
    if response.status_code == 200:
        print("Download successful, extracting file...")
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extract(csv_file_name, ".") 
            
        print(f"File '{csv_file_name}' extracted successfully.")
    else:
        print("Failed to download the file. Status code:", response.status_code)

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def housing_upload_join_data(conn, year):
  start_date = str(year) + "-01-01"
  end_date = str(year) + "-12-31"

  cur = conn.cursor()
  print('Selecting data for year: ' + str(year))
  cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
  rows = cur.fetchall()

  csv_file_path = 'output_file.csv'

  # Write the rows to the CSV file
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)
  print('Storing data for year: ' + str(year))
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  print('Data stored for year: ' + str(year))
  conn.commit()


  def count_rows_in(conn, table_name):
    """
    Count the number of rows in a database table.
    """
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cur.fetchone()[0]
    return count


def find_all_unique_years_in(conn, table_name):
    """
    Find all unique years in a database table.
    """
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT YEAR(date_of_transfer) FROM {table_name}")
    years = [row[0] for row in cur.fetchall()]
    return years


def get_top_5_rows(conn, table_name):
    """
    Get the top 5 rows of a database table.
    """
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name} LIMIT 5")
    rows = cur.fetchall()

    column_names = [description[0] for description in cur.description]  # Get column names from the cursor

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=column_names)

    return df

def get_df(conn, table_name):
    """
    Get a DataFrame from a database table.
    """
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name}")
    rows = cur.fetchall()

    column_names = [description[0] for description in cur.description]  # Get column names from the cursor

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=column_names)

    return df


def create_indices(conn, table_name, indices):
    cur = conn.cursor()
    for index in indices:
        index_name = f"idx_{index.replace(', ', '_')}"
        cur.execute(f"CREATE INDEX {index_name} ON {table_name} ({index}) USING BTREE;")
    conn.commit()
    print("All indexes created successfully.")


def load_pp_data_to_dataframe(file_paths):
    """
    Load multiple CSV files into a single pandas DataFrame with specified column names.

    Parameters:
    - file_paths: list of file paths for the CSV files

    Returns:
    - A concatenated DataFrame containing data from all the CSV files with specified column names
    """

    # Define the column names to match the SQL table `pp_data`
    columns = [
        "transaction_unique_identifier",
        "price",
        "date_of_transfer",
        "postcode",
        "property_type",
        "new_build_flag",
        "tenure_type",
        "primary_addressable_object_name",
        "secondary_addressable_object_name",
        "street",
        "locality",
        "town_city",
        "district",
        "county",
        "ppd_category_type",
        "record_status",
    ]
    # List to store DataFrames for each file
    data_frames = []

    for file_path in file_paths:
        # Load each CSV into a DataFrame with specified columns
        df = pd.read_csv(file_path, header=None, names=columns)
        data_frames.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(data_frames, ignore_index=True)

    return combined_df

def load_postcode_data_to_dataframe(file_path):
    postcode_columns = [
        "postcode",
        "status",
        "usertype",
        "easting",
        "northing",
        "positional_quality_indicator",
        "country",
        "latitude",
        "longitude",
        "postcode_no_space",
        "postcode_fixed_width_seven",
        "postcode_fixed_width_eight",
        "postcode_area",
        "postcode_district",
        "postcode_sector",
        "outcode",
        "incode",
    ]
    postcode_df = pd.read_csv(file_path, header=None, names=postcode_columns)

    return postcode_df

def merge_df(pp_data_df, postcode_data_df):
    """
    Join pp_data and postcode_data DataFrames in memory, filter by the specified year,
    and upload the result to the prices_coordinates_data table in the database.

    Parameters:
    - conn: Database connection object
    - year: Year to filter data by (int)
    - pp_data_df: DataFrame containing pp_data
    - postcode_data_df: DataFrame containing postcode_data
    """
    # Perform the join with postcode_data on the 'postcode' column
    merged_df = pp_data_df.merge(
        postcode_data_df,
        on='postcode',
        how='inner',
        suffixes=('_pp', '_po')
    )

    # Select only the required columns to match the target table structure
    final_df = merged_df[[
        'price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag',
        'tenure_type', 'locality', 'town_city', 'district', 'county', 'country',
        'primary_addressable_object_name', 'secondary_addressable_object_name', 'street',
        'latitude', 'longitude'
    ]]

    return final_df



def upload_dataframe_to_sql(df, table_name, username, password, url, port, database):
    """
    Delete existing rows in the target SQL table and upload a DataFrame to the table.

    Parameters:
    - df: DataFrame to upload.
    - conn: Database connection object.
    - table_name: Name of the target SQL table.
    """
    # Using SQLAlchemy's engine to facilitate upload
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{url}:{port}/{database}')
    
    # Upload the DataFrame to the specified table
    df.to_sql(table_name, engine, if_exists='append', index=False)
    print(f"Data uploaded to {table_name}.")

def get_pp_df(start_year, end_year):
  file_paths = [f"./pp-{year}-part{part}.csv" for year in range(start_year, end_year+1) for part in range(1, 3)]
  pp_df = load_pp_data_to_dataframe(file_paths)
  return pp_df

def get_postcode_df():
  postcode_file_path = "./open_postcode_geo.csv"
  postcode_df = load_postcode_data_to_dataframe(postcode_file_path)
  return postcode_df

def get_merged_df(pp_df, postcode_df):
  return merge_df(pp_df, postcode_df)

def upload_years(start_year, end_year, username, password, url, port, database):
  postcode_df = get_postcode_df()
  pp_df = get_pp_df(start_year, end_year)
  prices_coordinates_df = get_merged_df(pp_df, postcode_df)
  upload_dataframe_to_sql(prices_coordinates_df, "prices_coordinates_data_fast_full", username, password, url, port, database)
  print(f"Data uploaded successfully.")