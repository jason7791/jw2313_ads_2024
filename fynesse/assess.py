from .config import *
import pandas as pd
from . import access
import osmnx as ox
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from fuzzywuzzy import fuzz, process
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer



"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def read_table(conn, table_name, columns="*", conditions=None, limit=None):
    """
    Read data from a table in the database.

    Args:
        conn: The database connection object.
        table_name (str): Name of the table to read data from.
        columns (str): Columns to select (default is "*").
        conditions (str): WHERE clause conditions (optional).
        limit (int): Limit the number of rows returned (optional).

    Returns:
        pd.DataFrame: A DataFrame containing the queried data.
    """

    cur = conn.cursor()

    # Build the query
    query = f"SELECT {columns} FROM `{table_name}`"
    if conditions:
        query += f" WHERE {conditions}"
    if limit:
        query += f" LIMIT {limit}"

    print(query)
    # Execute the query
    cur.execute(query)

    # Fetch data and convert to DataFrame
    rows = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]  # Column names from cursor description
    df = pd.DataFrame(rows, columns=col_names)

    return df

def get_geometry_df(df, output_areas_geometry_df):
    df = pd.merge(df, output_areas_geometry_df, left_on='geography_code', right_on='OA21CD', how='inner')
    return  gpd.GeoDataFrame(df, geometry=df['geometry'])

def check_null_values(df):
    """
    Check for null values in a DataFrame and return detailed information.

    Parameters:
        df (pd.DataFrame or gpd.GeoDataFrame): The DataFrame to check for null values.

    Returns:
        pd.DataFrame: Summary of columns with null counts and percentages.
    """
    null_summary = df.isnull().sum()
    total_rows = len(df)

    null_details = null_summary[null_summary > 0]

    if not null_details.empty:
        null_df = pd.DataFrame({
            'Column': null_details.index,
            'Null Count': null_details.values,
            'Null Percentage': (null_details.values / total_rows) * 100
        }).sort_values(by='Null Percentage', ascending=False)

        print("Columns with Null Values:")
        print(null_df)
        return null_df
    else:
        print("No null values found in the DataFrame.")
        return pd.DataFrame()
    
def plot_frequency_distribution(data, column, bins=100, color="blue"):
    """
    Plot the frequency distribution of a given column.

    Parameters:
        data (pd.DataFrame or gpd.GeoDataFrame): The DataFrame containing the data.
        column (str): The column to plot.
        bins (int, optional): Number of bins for the histogram. Default is 30.
        color (str, optional): Color of the histogram. Default is "blue".
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=bins, kde=True, color=color, alpha=0.7)
    plt.title(f"Frequency Distribution of '{column}'", fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_map(df, col):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    df.plot(
        column=col,          # Column to plot
        cmap='coolwarm',              # Colormap
        legend=True,                  # Add a legend
        legend_kwds={'label': f"{col}", 'shrink': 0.8},
        ax=ax                         # Axes to plot on
    )

    # Add a title and remove axes for better visualization
    ax.set_title(f'{col} Distribution Across the UK', fontsize=16)
    ax.axis('off')  # Turn off axis

    plt.show()

def analyze_osm_features_high_low_proportion(general_gdf, osm_pbf_gdf, metric_col, high_quantile=0.8, low_quantile=0.2):
    """
    Analyze OSM features in areas with high and low percentages of a given metric.

    Parameters:
        ns_sec_gdf (GeoDataFrame): GeoDataFrame containing geographic data and the metric to analyze.
        osm_pbf_df (GeoDataFrame): GeoDataFrame containing OSM data with geographic coordinates.
        metric_col (str): Column name of the metric to analyze (e.g., 'ns_sec_l15').
        high_quantile (float): Quantile threshold for high values (default is 0.8).
        low_quantile (float): Quantile threshold for low values (default is 0.2).

    Returns:
        high_features_summary (DataFrame): Summary of common OSM features in high metric areas.
        low_features_summary (DataFrame): Summary of common OSM features in low metric areas.
    """
    if osm_pbf_gdf.crs != general_gdf.crs:
            osm_pbf_gdf = osm_pbf_gdf.to_crs(general_gdf.crs)

    high_threshold = general_gdf[metric_col].quantile(high_quantile)
    low_threshold = general_gdf[metric_col].quantile(low_quantile)

    high_areas = general_gdf[general_gdf[metric_col] >= high_threshold]
    low_areas = general_gdf[general_gdf[metric_col] <= low_threshold]

    high_features = gpd.sjoin(osm_pbf_gdf, high_areas, how="inner", predicate="within")
    low_features = gpd.sjoin(osm_pbf_gdf, low_areas, how="inner", predicate="within")

    high_features_summary = high_features.groupby(['key', 'value']).size().reset_index(name='count').sort_values(by='count', ascending=False)
    low_features_summary = low_features.groupby(['key', 'value']).size().reset_index(name='count').sort_values(by='count', ascending=False)

    return high_features_summary, low_features_summary

def compare_feature_counts(high_summary, low_summary):
    """
    Compare feature counts between two groups and compute metrics to identify distinct features.

    Args:
        high_summary (pd.DataFrame): DataFrame with 'key', 'value', and 'count' columns for the high group.
        low_summary (pd.DataFrame): DataFrame with 'key', 'value', and 'count' columns for the low group.

    Returns:
        pd.DataFrame: A DataFrame with merged counts, difference metric, and total counts, sorted by total count and difference.
    """
    # Merge the two tables on key-value pairs
    merged_features = pd.merge(
        high_summary,
        low_summary,
        on=['key', 'value'],
        how='outer',
        suffixes=('_high', '_low')
    )

    # Fill NaN counts with 0 for features that appear in only one of the groups
    merged_features['count_high'] = merged_features['count_high'].fillna(0)
    merged_features['count_low'] = merged_features['count_low'].fillna(0)

    # Calculate a difference metric (e.g., normalized difference)
    merged_features['difference'] = (merged_features['count_high'] - merged_features['count_low']) / (
        merged_features['count_high'] + merged_features['count_low']
    )

    # Add total count for sorting
    merged_features['total_count'] = merged_features['count_high'] + merged_features['count_low']

    # Sort by the absolute value of the difference to find the most distinct features
    merged_features = merged_features.sort_values(by=['difference'], key=abs, ascending=False)

    return merged_features

def filter_features(merged_features, total_count_threshold , difference_threshold):# Set thresholds for filtering
    # Filter rows based on total_count and difference
    return merged_features[
        (merged_features['total_count'] > total_count_threshold) &  # High total_count
        (merged_features['difference'].abs() >= difference_threshold)  # Difference close to 1 or -1
    ].sort_values(by=['total_count'], key=abs, ascending=False)

def count_exact_features_within_polygons(ns_sec_gdf, student_osm_gdf, feature_columns, crs_epsg=27700, area_threshold=1, cap_value=None):
    """
    Counts and normalizes the number of specific features (from student_osm_gdf) within each polygon (in ns_sec_gdf).
    """
    # Ensure the CRS of both GeoDataFrames match
    ns_sec_gdf = ns_sec_gdf.copy()
    student_osm_gdf = student_osm_gdf.copy()

    ns_sec_gdf = ns_sec_gdf.to_crs(epsg=crs_epsg)
    student_osm_gdf = student_osm_gdf.to_crs(epsg=crs_epsg)

    # Create combined feature column
    if 'key' in student_osm_gdf.columns and 'value' in student_osm_gdf.columns:
        student_osm_gdf['feature'] = student_osm_gdf['key'] + "_" + student_osm_gdf['value']
    else:
        raise ValueError("student_osm_gdf must have 'key' and 'value' columns.")

    # Spatial join
    joined_gdf = gpd.sjoin(student_osm_gdf, ns_sec_gdf, how='left', predicate='within')


    # Count features
    if not joined_gdf.empty:
        feature_counts = joined_gdf.groupby(['OA21CD', 'feature']).size().reset_index(name='count')
        pivot_table = feature_counts.pivot(index='OA21CD', columns='feature', values='count').fillna(0).reset_index()
        print("Pivot table created.")
    else:
        print("No features found within polygons.")
        pivot_table = pd.DataFrame()

    # Merge pivot table back to ns_sec_gdf
    result = ns_sec_gdf.merge(pivot_table, on='OA21CD', how='left').fillna(0)

    # Normalize by area
    result['area_sqm'] = result['geometry'].area
    for col in feature_columns:
        if col in result.columns:
            result[col] = (result[col] / result['area_sqm']) * 1e6
            if cap_value:
                result[col] = result[col].clip(upper=cap_value)
        else:
            print(f"Feature column '{col}' not found in result.")

    return result

def count_features_within_polygons(
    census_gdf, osm_gdf, feature_columns, feature_buffer_distances, crs_epsg=27700, cap_value=None
):
    """
    Counts and normalizes the number of specific features (from student_osm_gdf) within each polygon (in ns_sec_gdf),
    with feature-specific buffering and normalization improvements.

    Parameters:
        ns_sec_gdf (GeoDataFrame): GeoDataFrame containing polygons to analyze.
        student_osm_gdf (GeoDataFrame): GeoDataFrame containing features to buffer and count.
        feature_columns (list): List of features to consider for counting.
        crs_epsg (int): EPSG code for the CRS to use for both GeoDataFrames.
        area_threshold (float): Minimum polygon area to include in the analysis.
        cap_value (float): Optional maximum cap for normalized feature values.

    Returns:
        GeoDataFrame: Updated GeoDataFrame with feature counts normalized by area.
    """


    # Ensure the CRS of both GeoDataFrames match
    census_gdf = census_gdf.copy()
    osm_gdf = osm_gdf.copy()

    census_gdf = census_gdf.to_crs(epsg=crs_epsg)
    osm_gdf = osm_gdf.to_crs(epsg=crs_epsg)

    # Create a combined feature column in student_osm_gdf
    osm_gdf['feature'] = osm_gdf['key'] + "_" + osm_gdf['value']

    # Apply feature-specific buffering
    osm_gdf['buffer_distance'] = osm_gdf['feature'].map(feature_buffer_distances).fillna(200)
    osm_gdf['geometry'] = osm_gdf.apply(
        lambda row: row.geometry.buffer(row.buffer_distance), axis=1
    )

    # Perform spatial join
    joined_gdf = gpd.sjoin(census_gdf, osm_gdf, how='left', predicate='intersects')

    # Count the number of features for each OA21CD
    feature_counts = joined_gdf.groupby(['OA21CD', 'feature']).size().reset_index(name='count')

    # Pivot the table to have one column per feature
    pivot_table = feature_counts.pivot(index='OA21CD', columns='feature', values='count').fillna(0).reset_index()

    # Merge the counts back to the original ns_sec_gdf
    result = census_gdf.merge(pivot_table, on='OA21CD', how='left').fillna(0)

    # Calculate the area of each region in square meters
    result['area_sqm'] = result['geometry'].area

    # Normalize feature counts by region area (counts per square kilometer)
    for col in feature_columns:
        if col not in result.columns:
            result[col] = 0
        else:
            result[col] = (result[col] / result['area_sqm']) * 1e6  # Normalize to counts per km²
            if cap_value:
                result[col] = result[col].clip(upper=cap_value)  # Cap extreme values if specified

    # Drop any remaining NaN values
    result = result.dropna()

    return result



def plot_correlation_matrix(dataframe, cols_to_include, title="Correlation Matrix", figsize=(12,8)):
    """
    Plot a correlation matrix for the selected columns in the DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        cols_to_include (list): List of column names to include in the correlation matrix.
        title (str): Title for the correlation plot.
    """
    # Compute the correlation matrix
    correlation_matrix = dataframe[cols_to_include].corr()

    # Plot the correlation matrix
    plt.figure(figsize=figsize)
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True
    )
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def perform_pca_and_plot_cumulative_variance(data, features, n_components=None, figsize=(10, 6)):
    """
    Perform PCA on selected numerical features, plot explained variance, and return the PCA results.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        features (list): List of numerical feature columns to include in PCA.
        n_components (int, optional): Number of principal components to retain. If None, all components are used.
        figsize (tuple, optional): The figure size for the explained variance plot.

    Returns:
        tuple: A tuple containing:
            - pca_df (pd.DataFrame): DataFrame with the principal components.
            - explained_variance_ratio (list): List of explained variance ratios for each principal component.
    """
    # Select numerical features and handle missing values
    imputer = SimpleImputer(strategy="mean")  # Replace missing values with the mean
    imputed_features = imputer.fit_transform(data[features])

    # Standardize the features (mean=0, variance=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(imputed_features)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_features)

    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
    )

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Compute the cumulative explained variance
    cumulative_variance = explained_variance_ratio.cumsum()
    n_components_needed = (cumulative_variance >= 0.95).argmax() + 1

    # Plot the explained variance ratio
    plt.figure(figsize=figsize)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center')
    plt.step(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio.cumsum(), where='mid', label='Cumulative Explained Variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Explained')
    plt.title('Explained Variance by Principal Components')
    plt.legend()
    plt.show()

    # Print the explained variance ratio for each principal component
    print("Explained Variance Ratios:")
    for i, ratio in enumerate(explained_variance_ratio, 1):
        print(f"PC{i}: {ratio:.4f}")

    print(f"Number of components needed to explain at least 95% of variance: {n_components_needed}")

    return pca_df, explained_variance_ratio

def plot_high_student_proportion_map(dataframe, column, threshold=0.4):
    """
    Plots a map highlighting areas where the proportion of students exceeds a given threshold.

    Parameters:
        dataframe (GeoDataFrame): GeoDataFrame containing the data and geometries.
        column (str): Column name representing the student proportion.
        threshold (float): Threshold for highlighting areas (default is 0.4).
    """
    # Filter areas with high and low student proportions
    high_proportion_areas = dataframe[dataframe[column] > threshold]
    low_proportion_areas = dataframe[dataframe[column] <= threshold]

    # Plot the map
    fig, ax = plt.subplots(figsize=(12, 12))
    low_proportion_areas.plot(ax=ax, color="lightblue", alpha=0.5, label=f"{column} <= {threshold}")
    high_proportion_areas.plot(ax=ax, color="red", alpha=0.7, label=f"{column} > {threshold}")

    # Add map details
    plt.title(f"Map Highlighting Areas with High {column} Proportion (> {threshold})", fontsize=16)
    plt.legend()
    plt.show()

def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """

    distance_m = distance_km * 1000

    point = (latitude, longitude)

    pois = ox.geometries_from_point(point, tags=tags, dist=distance_m)

    poi_counts = {}

    for tag in tags:
        poi_counts[tag] = pois[tag].notnull().sum() if tag in pois.columns else 0

    return poi_counts

def get_osm_feature_counts(locations_dict, tags, distance_km):
  poi_data = []
  for location, coords in locations_dict.items():
      latitude, longitude = coords
      poi_counts = count_pois_near_coordinates(latitude, longitude, tags, distance_km)
      poi_counts['Location'] = location
      poi_data.append(poi_counts)

  poi_counts_df = pd.DataFrame(poi_data)

  # Reorder columns so 'Location' is first
  poi_counts_df = poi_counts_df[['Location'] + [col for col in poi_counts_df.columns if col != 'Location']]

  return poi_counts_df


def get_kmeans_clusters(poi_counts_df, n_clusters=3):
  poi_counts_df = poi_counts_df.copy()
  poi_features = poi_counts_df.drop(columns=['Location'])

  #normalise
  scaler = StandardScaler()
  poi_features_scaled = scaler.fit_transform(poi_features)

  #kmeans
  kmeans = KMeans(n_clusters=n_clusters, random_state=0)
  poi_counts_df['Cluster'] = kmeans.fit_predict(poi_features_scaled)

  return poi_counts_df

def plot_two_clusters(kmeans_poi_counts_df, manual_poi_counts_df):
  # Prepare color palettes to ensure consistency between plots
  kmeans_unique_clusters = kmeans_poi_counts_df['Cluster'].nunique()
  manual_unique_clusters = manual_poi_counts_df['Manual Category'].nunique()
  kmeans_palette = sns.color_palette("viridis", kmeans_unique_clusters)
  manual_palette = sns.color_palette("plasma", manual_unique_clusters)

  # Create a figure with side-by-side subplots
  fig, axes = plt.subplots(1, 2, figsize=(15, 8))

  # Plot 1: KMeans Clustering
  sns.scatterplot(
      data=kmeans_poi_counts_df,
      x='amenity', y='historic',
      hue='Cluster', palette= kmeans_palette,
      ax=axes[0], s=100, edgecolor='k'
  )
  axes[0].set_title('Map of KMeans Clustering Results')
  axes[0].legend(title="KMeans Cluster")

  # Plot 2: Manual Clustering
  sns.scatterplot(
      data=manual_poi_counts_df,
      x='amenity', y='historic',
      hue='Manual Category', palette=manual_palette,
      ax=axes[1], s=100, edgecolor='k'
  )
  axes[1].set_title('Map of Manual Clustering Results')
  axes[1].legend(title="Manual Category")

  # Set common labels
  for ax in axes:
      ax.set_xlabel("amenity")
      ax.set_ylabel("historic")

  plt.tight_layout()
  plt.show()

def get_distance_df(poi_counts_df):
  # Select relevant POI features and normalize them
  poi_features = poi_counts_df.drop(columns=['Location'])
  scaler = StandardScaler()
  poi_features_scaled = scaler.fit_transform(poi_features)

  # Compute the distance matrix (Euclidean distance)
  distance_matrix = squareform(pdist(poi_features_scaled, metric='euclidean'))

  # Convert the distance matrix into a DataFrame for better readability in heatmap
  distance_df = pd.DataFrame(distance_matrix, index=poi_counts_df['Location'], columns=poi_counts_df['Location'])

  return distance_df


def plot_distance_df(distance_df):
  plt.figure(figsize=(10, 8))
  sns.heatmap(distance_df, cmap="viridis", annot=False, linewidths=0.5)
  plt.title('Distance Matrix of Locations Based on Normalized POI Features')
  plt.xlabel('Location')
  plt.ylabel('Location')
  plt.show()


def plot_feature_correlation_matrix(poi_counts_df):
  poi_features = poi_counts_df.drop(columns=['Location'])

  # Compute the correlation matrix
  correlation_matrix = poi_features.corr()

  # Plot the correlation matrix as a heatmap
  plt.figure(figsize=(10, 8))
  sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
  plt.title('Feature Correlation Matrix for POI Features')
  plt.show()

def select_houses(conn, latitude, longitude, distance_km=1):
    lat_delta = distance_km  / 111  # Approximate degree latitude for given distance
    lon_delta = distance_km  / 111  # Approximate degree longitude for given distance

    lat_min = latitude - lat_delta /2
    lat_max = latitude + lat_delta /2
    lon_min = longitude - lon_delta /2
    lon_max = longitude + lon_delta /2


    # Define the start date for transactions since 2020
    start_date = "2020-01-01"

    # Create a cursor
    cur = conn.cursor()
    print("Selecting data within 1km x 1km of Cambridge center since 2020...")

    # Execute the query
    cur.execute(f'''SELECT price, date_of_transfer, postcode, property_type, new_build_flag,
                           tenure_type, locality, town_city, district, county, country, latitude, longitude, primary_addressable_object_name, secondary_addressable_object_name, street
                    FROM prices_coordinates_data_fast_full
                    WHERE date_of_transfer >= "{start_date}"
                    AND latitude BETWEEN {lat_min} AND {lat_max}
                    AND longitude BETWEEN {lon_min} AND {lon_max};''')
    rows = cur.fetchall()
    column_names = [desc[0] for desc in cur.description]  # Extract column names from cursor description
    houses_df = pd.DataFrame(rows, columns=column_names)  # Create DataFrame

    return houses_df

def fetch_buildings_data(lat, lon, distance_km=1):
    lat_delta = distance_km / 111  
    lon_delta = distance_km / 111  

    lat_min = lat - lat_delta /2
    lat_max = lat + lat_delta /2
    lon_min = lon - lon_delta /2
    lon_max = lon + lon_delta /2

    tags = {'building': True}

    buildings = ox.geometries_from_bbox(lat_max, lat_min, lon_max, lon_min, tags)

    address_fields = ['addr:housenumber', 'addr:street', 'addr:postcode']
    osm_df_with_address = buildings.dropna(subset=address_fields).copy()
    osm_df_with_address['area_sqm'] = osm_df_with_address.geometry.area 

    osm_df_without_address = buildings[~buildings.index.isin(osm_df_with_address.index)]

    return osm_df_with_address, osm_df_without_address

def plot_buildings(df_with_address, df_without_address):
    fig, ax = plt.subplots(figsize=(10, 10))
    df_with_address.plot(ax=ax, color='blue', alpha=0.6, edgecolor='k', label='With Address')
    df_without_address.plot(ax=ax, color='red', alpha=0.3, edgecolor='k', label='Without Address')
    plt.title("Buildings in Specified Area with Address Information")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show()


def exact_address_match(pp_df, osm_df):
    # Standardize column names for easier merging
    pp_df = pp_df.rename(columns={
        'primary_addressable_object_name': 'housenumber',
    })
    osm_df = osm_df.rename(columns={
        'addr:housenumber': 'housenumber',
        'addr:street': 'street',
        'addr:postcode': 'postcode'
    })

    # Perform exact matches on address columns
    matched_df = pd.merge(pp_df, osm_df, on=['postcode', 'street', 'housenumber'], how='inner', suffixes=('_pp', '_osm'))

    # Identify unmatched rows in pp_df
    pp_unmatched = pp_df.merge(matched_df[['postcode', 'street', 'housenumber']], on=['postcode', 'street', 'housenumber'], how='left', indicator=True)
    pp_unmatched = pp_unmatched[pp_unmatched['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Identify unmatched rows in osm_df
    osm_unmatched = osm_df.merge(matched_df[['postcode', 'street', 'housenumber']], on=['postcode', 'street', 'housenumber'], how='left', indicator=True)
    osm_unmatched = osm_unmatched[osm_unmatched['_merge'] == 'left_only'].drop(columns=['_merge'])


    return matched_df, pp_unmatched, osm_unmatched

def fuzzy_address_match(pp_unmatched, osm_unmatched):
    """
    Perform fuzzy matching on unmatched records to find potential non-exact matches.

    Args:
        pp_unmatched (DataFrame): Unmatched price paid data.
        osm_unmatched (GeoDataFrame): Unmatched OSM data.

    Returns:
        fuzzy_matched_df (DataFrame): DataFrame with fuzzy matched records.
    """
    fuzzy_matches = []

    # Iterate over unmatched PP records to find closest matches in OSM
    for _, pp_row in pp_unmatched.iterrows():
        # Generate a full address string for fuzzy matching
        pp_address = f"{pp_row['housenumber']} {pp_row['street']} {pp_row['postcode']}"

        # Apply fuzzy matching to find best match from unmatched OSM addresses
        osm_unmatched['osm_address'] = osm_unmatched['housenumber'].astype(str) + ' ' + osm_unmatched['street'] + ' ' + osm_unmatched['postcode']
        matched = process.extractOne(pp_address, osm_unmatched['osm_address'], scorer=fuzz.token_sort_ratio)
        if matched:
            matched_osm_address, score, _ = matched
            if score >= 85:
                matched_osm_row = osm_unmatched[osm_unmatched['osm_address'] == matched_osm_address].iloc[0]
                matched_row = pd.concat([pp_row, matched_osm_row], axis=0)
                fuzzy_matches.append(matched_row.to_dict())
    print(fuzzy_matches)
    # Create DataFrame from fuzzy matched records
    fuzzy_matched_df = pd.DataFrame(fuzzy_matches)

    return fuzzy_matched_df

def get_matches(houses_df, osm_df_with_address):
  pp_df = houses_df.copy()
  osm_df = osm_df_with_address.copy()

  # Perform exact matching
  matched_df, pp_unmatched, osm_unmatched = exact_address_match(pp_df, osm_df)

  # Perform fuzzy matching on unmatched records
  fuzzy_matched_df = fuzzy_address_match(pp_unmatched, osm_unmatched)

  # Combine exact and fuzzy matches
  final_matched_df = pd.concat([matched_df, fuzzy_matched_df], ignore_index=True)

  return final_matched_df


def analyze_price_area_relationship(df):
    df = df.dropna(subset=['price', 'area_sqm'])

    # Calculate correlation between price and area
    correlation = df['price'].corr(df['area_sqm'])
    print(f"Correlation between Price and Area (sqm): {correlation:.2f}")

    # Scatter plot of Price vs. Area
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='area_sqm', y='price', hue='property_type', palette='viridis', alpha=0.6)
    plt.title("Price vs. Area by Property Type")
    plt.xlabel("Area (sqm)")
    plt.ylabel("Price")
    plt.legend(title='Property Type')
    plt.show()