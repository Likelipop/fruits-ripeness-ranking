from utils import upload_file, upload_to_data_lake


if __name__ == "__main__":
    # Simple upload to a specific folder ID
    result = upload_file(
        "data/report.csv", folder_id="1TAYvnFJndClmvYc8819ulF0hgRdlr2en"
    )
    print(f"Uploaded: {result['name']} → {result['id']}")

    # Upload into the DataLake hierarchy  (DataLake/raw/2024-06/report.csv)
    result = upload_to_data_lake(
        local_path="data/report.csv",
        lake_root="DataLake",
        sub_folder="raw/2024-06",
    )
    print(f"Uploaded to lake: {result['name']} in folder {result['folder_id']}")
