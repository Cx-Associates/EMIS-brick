import os
import requests
import yaml

# Load API configuration
env_filename = 'api_keys.yml'
f_drive_path = 'F:/PROJECTS/1715 Main Street Landing EMIS Pilot/code/API keys'
env_filepath = os.path.join(f_drive_path, env_filename)

with open(env_filepath, 'r') as file:
    config = yaml.safe_load(file)
    url = config['DATABASE_URL']
    auth_token = config['API_KEY']

# Set the endpoint and headers
start_time = '2024-03-01'
end_time = '2024-09-31'
endpoint = f'cxa_main_st_landing/2404:10-240410/binaryValue/5/timeseries?start_time={start_time}&end_time={end_time}'
full_url = url + endpoint
headers = {
    'Authorization': f'Bearer {auth_token}',
    'Content-Type': 'application/json',
}

# Debug outputs
print(f'Base URL: {url}')
print(f'Endpoint: {endpoint}')
print(f'Full URL: {full_url}')
print(f'Headers: {headers}')

# Send the request
response = requests.get(full_url, headers=headers)

# Print the response details
print(f'Status code: {response.status_code}')
print(f'Reason: {response.reason}')
print(f'Content: {response.content}')

# Detailed error handling
if response.status_code == 404:
    print("Error: The requested endpoint was not found. Please check the endpoint URL.")
elif response.status_code == 401:
    print("Error: Unauthorized. Please check the API key and permissions.")
elif response.status_code != 200:
    print(f"Error: Unexpected response. Status code: {response.status_code}, Reason: {response.reason}")

# Print sample data if available
if response.status_code == 200 and response.content:
    data = response.json()
    if 'point_samples' in data and data['point_samples']:
        print(f"Data received: {data['point_samples']}")
    else:
        print("No data available for the specified time range.")
