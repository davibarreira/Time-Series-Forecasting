#%%
import polars as pl 
import numpy as np
import os
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import seaborn as sns
from lets_plot import *
LetsPlot.setup_html()

pl.Config.set_tbl_rows(100)

# # Load environment variables from .env file
load_dotenv()
# # Retrieve the dataset path from the environment variable
# DATASET = os.getenv('DATASET')
DATASET = '../../ds-spatiotemporal-mosqlimate/data/03_primary/dataset_complete_dengue_municipality.parquet'
df = pl.read_parquet(DATASET)
df.head()
#%%

es = df.filter(pl.col('uf') == 'ES')
es = es.with_columns(
    np.log1p(pl.col('casos')).alias('log_casos')
)

# Add week number column for seasonal interpolation
es = es.with_columns(
    pl.col('date').dt.week().alias('week')
)

# Create a copy of the dataframe for interpolation
es_interpolated = es.clone()

# Get the years with missing data
missing_years = [2021, 2022]

# For each missing year
for year in missing_years:
    # For each week
    for week in range(1, 53):
        # Get the average value for this week from previous years
        historical_avg = es.filter(
            (pl.col('week') == week) & 
            (~pl.col('date').dt.year().is_in(missing_years))
        ).select(pl.col('log_casos').mean()).item()
        
        # Update the value in the interpolated dataframe
        es_interpolated = es_interpolated.with_columns(
            pl.when(
                (pl.col('date').dt.year() == year) & 
                (pl.col('week') == week)
            ).then(historical_avg).otherwise(pl.col('log_casos')).alias('log_casos')
        )

# Plot original and interpolated data
plt.figure(figsize=(15, 6))
sns.lineplot(data=es, x='date', y='log_casos', label='Original', alpha=0.5)
sns.lineplot(data=es_interpolated, x='date', y='log_casos', label='Interpolated', alpha=0.5)
plt.title('Time Series with Seasonal Average Interpolation')
plt.legend()
plt.show()

# %%


