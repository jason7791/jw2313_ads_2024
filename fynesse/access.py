from .config import *
import requests
import pymysql
import csv
import time
import zipfile
import io
import os
import pandas as pd
from sqlalchemy import create_engine
import osmium 
import geopandas as gpd

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def process_osm_file(url, file_path):
    """
    Download and process an OSM file to extract node data.

    Parameters:
    - url (str): URL of the OSM PBF file to download.
    - file_path (str): Path to save the downloaded file.

    Returns:
    - pd.DataFrame: DataFrame containing node data (id, latitude, longitude, tags).
    """
    print("Downloading the file...")
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)
    print(f"File downloaded and saved as: {file_path}")

    class OSMHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.data = []

        def node(self, n):
            tags = dict(n.tags)
            if len(tags) > 1:  # Include nodes with more than 1 tag
                self.data.append([n.id, n.location.lat, n.location.lon, dict(n.tags)])

    print("Processing the file...")
    handler = OSMHandler()
    handler.apply_file(file_path)

    map_df = pd.DataFrame(handler.data, columns=["id", "latitude", "longitude", "tags"])
    print("Processing complete!")

    return map_df

def process_tags_in_chunks(df, chunk_size=100000):
    """
    Transform the tags column in a DataFrame into key-value pairs.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'tags' column containing dictionaries.
        chunk_size (int): Number of rows to process per chunk.

    Returns:
        pd.DataFrame: A transformed DataFrame with keys and values in separwate rows.
    """
    results = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        expanded_rows = []
        for _, row in chunk.iterrows():
            tags = row['tags']
            if isinstance(tags, dict):
                for key, value in tags.items():
                    expanded_rows.append({
                        'id': row['id'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'key': key,
                        'value': value
                    })
        results.append(pd.DataFrame(expanded_rows))
        print("current rows", len(results))
    return pd.concat(results, ignore_index=True)

def download_and_process_shapefile(url, extract_to="shapefile_data"):
    """
    Download a shapefile from a URL, extract it, and load it into a GeoDataFrame.

    Args:
        url (str): URL of the shapefile to download.
        extract_to (str): Directory where the shapefile contents will be extracted.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the shapefile data.

    Raises:
        Exception: If the download fails or if the file cannot be processed.
    """
    try:
        print("Downloading the shapefile...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for failed requests
        print("Download successful!")

        print("Extracting the shapefile...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(extract_to)
        print(f"Files extracted to: {extract_to}")

        print("Loading the shapefile into a GeoDataFrame...")
        shapefile_gdf = gpd.read_file(extract_to)
        print("Shapefile loaded successfully!")

        return shapefile_gdf

    except requests.exceptions.RequestException as e:
        print(f"Failed to download file: {e}")
        raise
    except zipfile.BadZipFile as e:
        print(f"Failed to unzip file: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def download_census_data(code, base_dir=''):
  url = f'https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip'
  extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])

  if os.path.exists(extract_dir) and os.listdir(extract_dir):
    print(f"Files already exist at: {extract_dir}.")
    return

  os.makedirs(extract_dir, exist_ok=True)
  response = requests.get(url)
  response.raise_for_status()

  with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(extract_dir)

  print(f"Files extracted to: {extract_dir}")


def load_census_data(code, year=2021, level='msoa'):
  return pd.read_csv(f'census{year}-{code.lower()}/census{year}-{code.lower()}-{level}.csv')

def create_table(conn, table_name, column_definitions):
    """
    Create a table in the database.

    Args:
        conn: The database connection object.
        table_name (str): Name of the table to create.
        column_definitions (str): Column definitions in SQL format.
    """
    cur = conn.cursor()

    # Drop table if it already exists
    cur.execute(f"DROP TABLE IF EXISTS `{table_name}`;")

    # Ensure proper use of AUTO_INCREMENT with a primary key column
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
        {column_definitions}
    ) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """)
    conn.commit()
    print(f"Table `{table_name}` created successfully.")

def upload_dataframe_to_sql(df, table_name, username, password, url, port, database):
    # Using SQLAlchemy's engine to facilitate upload
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{url}:{port}/{database}')

    # Upload the DataFrame to the specified table
    df.to_sql(table_name, engine, if_exists='append', index=False)
    print(f"Data uploaded to {table_name}.")

def upload_dataframe_in_chunks(df, table_name, username, password, host, port, database, chunk_size=200000):
    """
    Upload the DataFrame to a MySQL table in chunks of a given size.

    Args:
        df (pd.DataFrame): The DataFrame to upload.
        table_name (str): The name of the table to upload data into.
        username (str): Database username.
        password (str): Database password.
        host (str): Database host.
        port (int): Database port.
        database (str): Database name.
        chunk_size (int): The number of rows to upload in each chunk (default is 200,000).
    """
    # Create a connection to the MySQL database using SQLAlchemy
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')

    # Rename columns if needed (e.g., latitude to lat, longitude to long)
    # df.rename(columns={"latitude": "lat", "longitude": "long"}, inplace=True)

    # Break the DataFrame into smaller chunks and upload each one
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]

        # Upload the chunk
        chunk.to_sql(table_name, engine, if_exists='append', index=False)
        print(f"Uploaded rows {start} to {start + len(chunk)} to {table_name}.")


def parse_df(df, year, feature):
    cols_needed = ['date', 'geography', 'geography_code', feature]
    if('geographyCode' in df.columns):
        df.rename(columns={'geographyCode': 'geography_code'}, inplace= True)
    if('GeographyCode' in df.columns):
        df.rename(columns={'GeographyCode': 'geography_code'}, inplace= True)
    if('geography code' in df.columns):
        df.rename(columns={'geography code': 'geography_code'}, inplace = True)
    if('geography' not in df.columns):
        df['geography'] = df['geography_code']
    df['date'] = year
    df = df[cols_needed]
    return df


def stack_matching_rows(df_a, df_b, column_name='geography_code'):
    """
    Stacks rows of DataFrame B below DataFrame A,
    but only includes rows of B where the values in the specified column
    appear in DataFrame A.

    Parameters:
        df_a (pd.DataFrame): The first DataFrame.
        df_b (pd.DataFrame): The second DataFrame.
        column_name (str): The column name to match.

    Returns:
        pd.DataFrame: A new DataFrame with the stacked rows.
    """
    # Filter rows in df_b where column_name matches values in df_a[column_name]
    matching_rows = df_b[df_b[column_name].isin(df_a[column_name])]

    # Concatenate the two DataFrames
    stacked_df = pd.concat([df_a, matching_rows], ignore_index=True)

    return stacked_df

def download_csv_to_dataframe(url):
    """
    Downloads a CSV file from the given URL and loads it into a Pandas DataFrame.

    Parameters:
        url (str): The URL to download the CSV file from.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        response.raise_for_status()

        # Load the content into a Pandas DataFrame
        from io import StringIO
        csv_data = StringIO(response.text)  # Convert bytes to a string-like object
        df = pd.read_csv(csv_data)

        print("CSV file successfully downloaded and loaded into a DataFrame.")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Failed to download file: {e}")
        return None
    
def calculate_yearly_metrics(df, value_column, id_column="geography_code", date_column="date", year1=2011, year2=2021):
    """
    Calculates metrics for a given value column, such as values for specific years and percentage change.

    Args:
        df (pd.DataFrame): The input dataframe.
        value_column (str): The column name containing the values to analyze.
        id_column (str): The column name used to identify regions (default is "geography_code").
        date_column (str): The column name containing the year/date information.
        year1 (int): The first year (e.g., 2011).
        year2 (int): The second year (e.g., 2021).

    Returns:
        pd.DataFrame: Dataframe with additional columns for year1, year2 values and percentage change.
    """
    # Extract values for each year
    df[f"{value_column}_{year1}"] = df[value_column].where(df[date_column] == year1)
    df[f"{value_column}_{year2}"] = df[value_column].where(df[date_column] == year2)

    # Fill missing values by grouping by the region (id_column)
    df[f"{value_column}_{year1}"] = df.groupby(id_column)[f"{value_column}_{year1}"].transform('first')
    df[f"{value_column}_{year2}"] = df.groupby(id_column)[f"{value_column}_{year2}"].transform('first')

    # Calculate percentage change
    df[f"{value_column}_change"] = (
        (df[f"{value_column}_{year2}"] - df[f"{value_column}_{year1}"])
        /df[f"{value_column}_{year1}"]
    )

    # Keep only unique rows for each region
    return df[[id_column, f"{value_column}_{year1}", f"{value_column}_{year2}", f"{value_column}_change"]].drop_duplicates()

def calculate_car_to_public_transport_ratio(dataframe, car_columns, public_transport_columns, column_name='car_usage'):
    """
    Calculate the proportion of car users to public transport users in a given dataset.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame with mode of transport data.
        year (int): The year of the data for labeling or parsing purposes.
        car_columns (list): List of column names related to car usage.
        public_transport_columns (list): List of column names related to public transport usage.
        column_name (str): The name of the resulting ratio column. Default is 'car_usage'.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column for the car-to-public-transport ratio.
    """
    # Step 1: Calculate totals for car users and public transport users
    dataframe['Car Users'] = dataframe[car_columns].sum(axis=1)
    dataframe['Public Transport Users'] = dataframe[public_transport_columns].sum(axis=1)

    # Step 2: Calculate the car-to-public-transport ratio
    dataframe[column_name] = dataframe['Car Users'] / (dataframe['Public Transport Users'] + dataframe['Car Users'])

    return dataframe

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