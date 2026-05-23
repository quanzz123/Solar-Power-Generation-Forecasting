import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"

def merge_plant_data(plant_no: int):
    print(f"--- Merging Data for Plant {plant_no} ---")
    
    # 1. File paths
    gen_path = DATA_DIR / f"Plant_{plant_no}_Generation_Data.csv"
    sensor_weather_path = DATA_DIR / f"Plant_{plant_no}_Weather_Sensor_Data.csv"
    api_weather_path = DATA_DIR / f"open_meteo_weather_plant_{plant_no}.csv"
    
    # Check existences
    if not gen_path.exists() or not sensor_weather_path.exists():
        raise FileNotFoundError(f"Missing generation or weather sensor CSV for Plant {plant_no}")
    if not api_weather_path.exists():
        raise FileNotFoundError(f"Missing Open-Meteo crawled CSV for Plant {plant_no}")

    # 2. Read files
    gen_df = pd.read_csv(gen_path)
    sensor_df = pd.read_csv(sensor_weather_path)
    api_df = pd.read_csv(api_weather_path)

    # 3. Synchronize Datetime columns
    # Plant 1 generation date format is dd-mm-yyyy HH:MM, others are yyyy-mm-dd HH:MM:SS
    if plant_no == 1:
        gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"], format="%d-%m-%Y %H:%M", errors="coerce")
    else:
        gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"], errors="coerce")
        
    sensor_df["DATE_TIME"] = pd.to_datetime(sensor_df["DATE_TIME"], errors="coerce")
    api_df["DATE_TIME"] = pd.to_datetime(api_df["DATE_TIME"], errors="coerce")

    # Drop any row with invalid datetime in gen
    gen_df = gen_df.dropna(subset=["DATE_TIME"])

    # 4. Merge step 1: Generation Data + Weather Sensor Data
    # Drop SOURCE_KEY and PLANT_ID from sensor weather data to avoid column duplication
    sensor_subset = sensor_df[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
    
    merged_df = pd.merge(gen_df, sensor_subset, on="DATE_TIME", how="left")
    print(f"Merged Generation + Sensor Weather shape: {merged_df.shape}")

    # 5. Merge step 2: Result + Open-Meteo weather data
    # Drop redundant columns from api weather
    api_drop_cols = ["PLANT_NO", "LATITUDE", "LONGITUDE", "OPEN_METEO_TIMEZONE", "OPEN_METEO_UTC_OFFSET_SECONDS"]
    api_subset = api_df.drop(columns=[col for col in api_drop_cols if col in api_df.columns])

    final_merged = pd.merge(merged_df, api_subset, on="DATE_TIME", how="left")
    print(f"Merged with Open-Meteo API features shape: {final_merged.shape}")

    # 6. Linear interpolation on weather columns to ensure no NaN values remain
    weather_cols = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    om_cols = [col for col in final_merged.columns if col.startswith("OM_")]
    all_weather_cols = weather_cols + om_cols
    
    # Interpolate, back-fill, and forward-fill any edge cases
    final_merged[all_weather_cols] = final_merged[all_weather_cols].interpolate(method="linear").bfill().ffill()

    # 7. Save final dataset
    output_path = DATA_DIR / f"plant_{plant_no}_merged.csv"
    final_merged.to_csv(output_path, index=False)
    print(f"Successfully saved merged dataset to: {output_path}")
    print(f"Final columns ({len(final_merged.columns)}): {final_merged.columns.tolist()}\n")

def main():
    for p in [1, 2]:
        merge_plant_data(p)

if __name__ == "__main__":
    main()
