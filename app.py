import os
from dotenv import load_dotenv
import opik
from opik.integrations.openai import track_openai
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth

# Load environment variables from .env file
load_dotenv()

# Configure Opik with new settings
opik.configure(
    use_local=False,
    api_key=os.getenv('OPIK_API_KEY'),
)

# Configure OpenAI
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("Please set OPENAI_API_KEY environment variable")

# Configure Weaviate
WEAVIATE_CLUSTER_URL = os.getenv('WEAVIATE_CLUSTER_URL')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')

if not WEAVIATE_CLUSTER_URL or not WEAVIATE_API_KEY:
    raise EnvironmentError("Please set WEAVIATE_CLUSTER_URL and WEAVIATE_API_KEY environment variables")

# Initialize Weaviate client
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_CLUSTER_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
)

# Initialize OpenAI client with Opik tracking
client = track_openai(OpenAI())

@opik.track
def retrieve_context(user_query):
    # Get the DisasterInfo collection
    disaster_collection = weaviate_client.collections.get("DisasterInfo")

    # Semantic Search
    response = disaster_collection.query.near_text(
        query=user_query,
        limit=3
    )

    disaster_info = []
    for info in response.objects:
        disaster_info.append({
            'title': info.properties['title'],
            'content': info.properties['content'],
            'source': info.properties['source'],
            'url': info.properties['url']
        })
    return disaster_info

@opik.track
def generate_response(user_query, disaster_info):
    prompt = f"""
    You are a compassionate disaster relief assistant providing accurate, actionable information to people who may be in urgent situations.

    USER QUERY: "{user_query}"

    RETRIEVED INFORMATION:
    ```
    {disaster_info}
    ```

    Please provide a clear, concise, and empathetic response that:
    1. Directly addresses the user's immediate needs first
    2. Provides specific, actionable steps they can take right now
    3. Includes relevant contact information (phone numbers, websites) when available
    4. Organizes information in a structured, easy-to-scan format with bullet points where appropriate
    5. Cites all sources using their URLs in markdown format
    6. Acknowledges the emotional stress of disaster situations with a compassionate tone
    7. Prioritizes official government sources and recognized relief organizations

    If the information is incomplete or you're uncertain, acknowledge this and suggest reliable alternative resources.
    """

    response = client.chat.completions.create(
        model="gpt-4",  # Using GPT-4 for better response quality
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

@opik.track(name="disaster-relief-rag")
def llm_chain(user_query):
    context = retrieve_context(user_query)
    response = generate_response(user_query, context)
    return response

if __name__ == "__main__":
    print("Disaster Relief Information Assistant")
    print("Ask any question about disaster relief, emergency resources, or current situations.")
    print("Type 'quit' to exit.")

    while True:
        user_query = input("\nWhat would you like to know about disaster relief? ")
        if user_query.lower() == 'quit':
            break

        try:
            result = llm_chain(user_query)
            print("\nResponse:", result)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different query.")
