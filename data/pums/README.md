This folder contains ACS Public Use Microdata Sample (PUMS) data used for creating simulated tourists.
Considering copyright restrictions and their large file sizes, these files are not included in this repository.

To obtain the required PUMS data files and process them:

1. Obtain the following PUMS data

- [csv_hus.zip](https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year/csv_hus.zip)
- [csv_pus.zip](https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year/csv_pus.zip)

1. Unzip the downloaded files and place `.csv` files into this directory.

1. Run the `pums-convert.py` script to convert the CSV files into Parquet format for easier processing.