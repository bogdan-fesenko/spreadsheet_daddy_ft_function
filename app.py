import functions_framework
from custom_models import *
import openai

# The Firebase Admin SDK to access Cloud Firestore.
from firebase_admin import initialize_app, firestore
import google.cloud.firestore

@functions_framework.http
def run_train(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)
    request_args = request.args
    print(f"\n\nrequest_args: {request_args}")
    print(f"\n\nrequest_json: {request_json}")

    print("Initializing db inside main lambda handler...")

    # Initialize Firestore client
    db: google.cloud.firestore.Client = firestore.client()

    # Write fine_tune_id to Firestore with the generated UUID
    write_to_firestore(client=db,
                        uuid_key=uuid_value,
                        column='status',
                        value='Initialized',
                        collection=db_collection)

    # PARAMETERS
    # OPENAI_API_KEY = "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"
    engine = request_args['engine']  #'ada'
    db_collection = 'api_ft_queue'

    openai.api_key = os.getenv("AZURE_OPENAI_KEY")
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15' # this may change in the future

    # Set the OPENAI_API_KEY environment variable
    # os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    # Generate a UUID
    uuid_value = str(uuid.uuid4())

    print(f"\nuuid_value: {uuid_value}")

    # Extract input data from the event
    input_data = event['input_data']

    # Transform the input data to the desired JSON format
    json_data = transform_input(input_data)

    # Run fine_tune_model asynchronously
    fine_tune_process = subprocess.Popen(
        ['python', 'custom_models.py', 'fine_tune_model', db_collection, uuid_value, engine, json.dump(json_data)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Return UUID immediately
    response = {
        'uuid': uuid_value
    }

    return response