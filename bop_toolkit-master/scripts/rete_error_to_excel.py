import json
import pandas as pd

# Load the JSON file
file_path= r"../result/YOLO-ICP/YOLO-ICP-method3-multi8_lmo-test/error=rete_ntop=-1/errors_000002.json"
save_path= r"../result/YOLO-ICP/YOLO-ICP-method3-multi8_lmo-test/error=rete_ntop=-1/errors_000002.csv"

with open(file_path, 'r') as f:
    data = json.load(f)

# Initialize lists to store extracted data
excel_data = []

# Extract data from each entry in the JSON file
for entry in data:
    est_id = entry['est_id']
    im_id = entry['im_id']
    obj_id = entry['obj_id']
    score = entry['score']

    errors = entry['errors']
    error_keys = list(errors.keys())  # Get all keys within 'errors'
    first_error_key = error_keys[0]
    error_re, error_te = errors[first_error_key]

    # Append extracted data to the list
    excel_data.append([error_re, error_te, im_id, obj_id, score])

# Create a DataFrame from the extracted data
df = pd.DataFrame(excel_data, columns=['error_re', 'error_te', 'im_id', 'obj_id', 'score'])


# Save the DataFrame to a CSV file
df.to_csv(save_path, index=False)
print("saved at ",save_path)