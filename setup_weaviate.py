import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

# Load environment variables
load_dotenv()

# Configure Weaviate
WEAVIATE_CLUSTER_URL = os.getenv('WEAVIATE_CLUSTER_URL')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')

if not WEAVIATE_CLUSTER_URL or not WEAVIATE_API_KEY:
    raise EnvironmentError("Please set WEAVIATE_CLUSTER_URL and WEAVIATE_API_KEY environment variables")

# Initialize Weaviate client
client = None
try:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_CLUSTER_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )

    def scrape_disaster_info(url):
        """Scrape disaster relief information from a given URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract main content
            content = []

            # Get all paragraphs
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                if p.text.strip():
                    content.append(p.text.strip())

            # Get all headers
            headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for h in headers:
                if h.text.strip():
                    content.append(h.text.strip())

            # Get all list items
            list_items = soup.find_all('li')
            for li in list_items:
                if li.text.strip():
                    content.append(li.text.strip())

            return {
                'url': url,
                'title': soup.title.string if soup.title else url,
                'content': ' '.join(content),
                'source': url.split('/')[2],  # Extract domain as source
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    # List of disaster relief URLs to scrape - using only one URL now
    disaster_relief_urls = [
        "https://www.ca.gov/lafires/",
    ]

    # Create the schema and populate with data
    try:
        # Create the DisasterInfo collection with the correct format
        client.collections.create(
            name="DisasterInfo",
            description="Information about disaster relief and emergency resources",
            vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            properties=[
                Property(
                    name="url",
                    data_type=DataType.TEXT,
                    description="The source URL of the information"
                ),
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="The title or heading of the information"
                ),
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="The main content of the disaster relief information"
                ),
                Property(
                    name="source",
                    data_type=DataType.TEXT,
                    description="The source website or organization"
                ),
                Property(
                    name="timestamp",
                    data_type=DataType.TEXT,
                    description="When the information was collected"
                )
            ]
        )
        print("Successfully created DisasterInfo collection!")

        # Get the DisasterInfo collection
        disaster_collection = client.collections.get("DisasterInfo")

        # Scrape and add information from each URL
        for url in disaster_relief_urls:
            print(f"Scraping {url}...")
            info = scrape_disaster_info(url)
            if info:
                try:
                    disaster_collection.data.insert(info)
                    print(f"Successfully added information from {url}")
                except Exception as e:
                    print(f"Error adding data from {url}: {str(e)}")
            time.sleep(1)  # Be nice to the servers

        print("Successfully added disaster relief information!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if "already exists" in str(e):
            print("The DisasterInfo collection already exists in the schema.")
finally:
    # Close the client connection when done
    if client:
        print("Closing Weaviate client connection...")
        client.close()
        print("Weaviate client connection closed.")
