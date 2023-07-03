import openai

# The Firebase Admin SDK to access Cloud Firestore.
from firebase_admin import initialize_app, firestore
import google.cloud.firestore

openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future

def transform_input(input_data):
    """
    Transforms the input data from Google Sheets to the desired JSON format.

    Args:
        input_data (list): List of rows from the Google Sheets table.

    Returns:
        list: List of dictionaries in the desired JSON format.
    """
    json_data = []
    for row in input_data:
        prompt = " ".join(row[:-1])
        completion = row[-1]
        json_data.append({
            "prompt": prompt,
            "completion": completion
        })
    return json_data

def prepare_data(json_data, train_file_path, valid_file_path):
    """
    Prepares the training data by saving it locally in JSONL format.

    Args:
        json_data (list): List of JSON objects containing training data.
        train_file_path (str): Path to save the training data file.
        valid_file_path (str): Path to save the validation data file.

    Returns:
        None
    """
    with open(train_file_path, 'w') as train_file, open(valid_file_path, 'w') as valid_file:
        for i, item in enumerate(json_data):
            print("item:", item)
            json.dump(item, train_file)
            json.dump(item, valid_file)
            train_file.write('\n')
            valid_file.write('\n')

    # Step 2: Prepare data using the data preparation tool
    output = subprocess.run(['openai', 'tools', 'fine_tunes.prepare_data', '-f', train_file_path, '-f', valid_file_path, '-q'])
    print("output:", output)
    stdout = output.stdout
    print("stdout:", stdout)

def extract_fine_tune_id(fine_tune_output):
    """
    Extracts the fine-tune ID from the fine-tune output.

    Args:
        fine_tune_output (str): Output of the fine-tuning process.

    Returns:
        str: Fine-tune ID if found, None otherwise.
    """
    start_index = fine_tune_output.find("ft-")
    if start_index != -1:
        end_index = fine_tune_output.find(" ", start_index)
        if end_index != -1:
            fine_tune_id = fine_tune_output[start_index:end_index]
            return fine_tune_id
    logger.error("Error extracting fine-tune ID")
    return None

def write_to_firestore(client, uuid_key, column, value, collection):
    """
    Writes the value to Firestore with the given UUID key.

    Args:
        client: Firestore client.
        uuid_key (str): UUID key.
        value: Value to write.
        collection (str): Firestore collection name.

    Returns:
        None
    """
    doc_ref = client.collection(collection).document(uuid_key)
    doc_ref.set({column: value})

def fine_tune_model(db_collection, uuid_value, engine, json_data):
    """
    Fine-tunes a model using the OpenAI API.

    Args:
        json_data (list): List of JSON objects containing training data.
        api_key (str): OpenAI API key.

    Returns:
        str: Fine-tune ID if successful, None otherwise.
    """
    # Set the OPENAI_API_KEY environment variable
    #os.environ['OPENAI_API_KEY'] = api_key

    print('start')

    print("Initializing db inside fine-tune script...")
    # Initialize Firestore client
    db: google.cloud.firestore.Client = firestore.client()

    train_file_path = 'data_prepared_train.jsonl'
    valid_file_path = 'data_prepared_valid.jsonl'

    # Step 1: Prepare data
    time_start = time.time()
    prepare_data(json_data, train_file_path, valid_file_path)
    logger.info(f"Time preprocessing: {round(time.time() - time_start, 2)} seconds")

    print('fine-tune start')

    # Step 2: Fine-tune the model
    time_start = time.time()
    fine_tune_output = subprocess.run(['openai', 'api', 'fine_tunes.create', '-t', train_file_path,
                                       '-v', valid_file_path, '-m', 'ada'],
                                      capture_output=True, text=True)
    output = fine_tune_output.stdout
    print("fine-tune finished")
    print("stdout:", output)
    fine_tune_id = extract_fine_tune_id(output)

    write_to_firestore(client=db,
                        uuid_key=uuid_value,
                        column='status',
                        value='Fine-tuned',
                        collection=db_collection)

    write_to_firestore(client=db,
                        uuid_key=uuid_value,
                        column='token',
                        value=fine_tune_id,
                        collection=db_collection)

    return fine_tune_id