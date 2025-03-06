import os
from dotenv import load_dotenv
import opik
from opik.integrations.openai import track_openai
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
import geocoder
import json

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
def get_user_location():
    """Get the user's current location using IP-based geolocation."""
    try:
        g = geocoder.ip('me')
        if g.ok:
            location = {
                'city': g.city,
                'state': g.state,
                'country': g.country,
                'lat': g.lat,
                'lng': g.lng
            }
            return location
        else:
            return None
    except Exception as e:
        print(f"Error getting location: {str(e)}")
        return None

@opik.track
def retrieve_context(user_query, user_location=None):
    """Retrieve relevant disaster information based on user query and location."""
    # Get the DisasterInfo collection
    disaster_collection = weaviate_client.collections.get("DisasterInfo")

    # Enhance query with location if available
    enhanced_query = user_query
    if user_location and user_location.get('state'):
        enhanced_query = f"{user_query} in {user_location.get('city', '')} {user_location.get('state', '')} {user_location.get('country', '')}"
        print(f"Enhanced query with location: {enhanced_query}")

    # Semantic Search
    response = disaster_collection.query.near_text(
        query=enhanced_query,
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
def generate_response(user_query, disaster_info, user_location=None):
    """Generate a response based on the user query, retrieved information, and location."""
    # Add location context to the prompt if available
    location_context = ""
    if user_location:
        location_context = f"""
    USER LOCATION:
    City: {user_location.get('city', 'Unknown')}
    State/Region: {user_location.get('state', 'Unknown')}
    Country: {user_location.get('country', 'Unknown')}
    """

    prompt = f"""
    You are a compassionate disaster relief assistant providing accurate, actionable information to people who may be in urgent situations.

    USER QUERY: "{user_query}"
    {location_context}
    RETRIEVED INFORMATION:
    ```
    {disaster_info}
    ```

    Please provide a clear, concise, and empathetic response that:
    1. DIRECTLY PROVIDES the specific information the user needs - do NOT just tell them to visit a website
    2. Extracts and presents the most relevant details from the retrieved information
    3. Includes specific locations, phone numbers, and concrete steps when available
    4. Organizes information in a structured, easy-to-scan format with bullet points
    5. Only mentions source URLs at the end of your response in markdown format
    6. Uses a compassionate tone acknowledging the emotional stress of disaster situations
    7. Prioritizes actionable information over general advice
    8. If user location is provided, tailor the information to that specific location

    IMPORTANT: Users may be in emergency situations with limited connectivity. They need direct answers, not website referrals. Extract and provide the actual information from the sources rather than telling them to visit websites.
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
    """Main RAG pipeline that gets location, retrieves context, and generates a response."""
    # Get user's location
    user_location = get_user_location()
    if user_location:
        print(f"Detected location: {user_location.get('city', 'Unknown')}, {user_location.get('state', 'Unknown')}, {user_location.get('country', 'Unknown')}")

    # Retrieve context with location awareness
    context = retrieve_context(user_query, user_location)

    # Generate response with location awareness
    response = generate_response(user_query, context, user_location)

    return response

if __name__ == "__main__":
    print("Disaster Relief Information Assistant")
    print("Ask any question about disaster relief, emergency resources, or current situations.")
    print("Type 'quit' to exit.")

    # Check if we can detect location
    location = get_user_location()
    if location:
        print(f"Detected your location: {location.get('city', 'Unknown')}, {location.get('state', 'Unknown')}, {location.get('country', 'Unknown')}")
        print("Your queries will be enhanced with location-specific information.")
    else:
        print("Could not detect your location. Responses will be general.")

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
