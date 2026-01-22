import pandas as pd

#config vars
IN_PATH = "../../data/twitter/raw/"
DATE_COLUMN = "created_at"
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

#function takes a path to a csv and changes the datetime column to standard ISO 8601 in UTC (no output)
def standardise(path: str):
    #read csv into dataframe
    df = pd.read_csv(path)

    #collect time and standardise 
    stan_time = pd.to_datetime(
        df[DATE_COLUMN],
        utc=True,
        dayfirst=True,
        errors="coerce"
    )

    #write standardised time back to column
    df[DATE_COLUMN] = stan_time.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    #write dataframe back to csv
    df.to_csv(path, index=False)

for month in months:
    standardise(f"{IN_PATH}FT_{month}_2020.csv")