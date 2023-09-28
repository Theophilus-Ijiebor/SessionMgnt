import os, logging, time
import psycopg2
from langchain.text_splitter import TextSplitter, TokenTextSplitter
from langchain.vectorstores.pgvector import PGVector
from dotenv import load_dotenv
from langchain.document_loaders import AzureBlobStorageContainerLoader
import boto3
from langchain.embeddings import BedrockEmbeddings
    
load_dotenv()

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host="vector-1.c6snayjzs6va.us-east-1.rds.amazonaws.com",
    port="5432",
    database="postgres",
    user="postgres",
    password="oHBDfZ8lLmg8atgFQPih"
)

# AWS Bedrock Instance
try:
    # Using environment variable
    sts = boto3.client('sts',
                       aws_access_key_id=os.getenv("ACCESS_KEY"),
                       aws_secret_access_key=os.getenv("SECRET_KEY"),
                       )
    response = sts.assume_role(RoleArn="arn:aws:iam::934297252078:role/RocheBedrockRole",
                               RoleSessionName="ATOM-Bedrock")
except:
    # Using local credentials file
    # Create a session with the specified profile
    sts = boto3.client('sts')
    response = sts.assume_role(RoleArn="arn:aws:iam::934297252078:role/RocheBedrockRole",
                               RoleSessionName="ATOM-Bedrock")


new_session = boto3.Session(aws_access_key_id=response['Credentials']['AccessKeyId'],aws_secret_access_key=response['Credentials']['SecretAccessKey'],aws_session_token=response['Credentials']['SessionToken'])
bedrock = new_session.client(service_name='bedrock',region_name='us-east-1',endpoint_url='https://bedrock.us-east-1.amazonaws.com')

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def delete_table(tablename):

    #establishing the connection
    conn = psycopg2.connect(database="postgres", user='postgres', password='oHBDfZ8lLmg8atgFQPih', host='vector-1.c6snayjzs6va.us-east-1.rds.amazonaws.com', port= '5432'
    )
    #Setting auto commit false
    conn.autocommit = True

    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    #Doping EMPLOYEE table if already exists
    cursor.execute("DROP TABLE "+tablename)
    print("Table dropped... ")

    #Commit your changes in the database
    conn.commit()

    #Closing the connection
    conn.close()

def create_vectors(project):
    start_time = time.time()
    res = ""

    # STATIC PATHS
    DOCUMENT_PATH: str = "./data/"+project+"/originals/"
    VECTOR_STORE_PATH: str = "./data/"+project+"/storage/"
    BEDROCK_EMBEDDING_MODEL_NAME: str = "amazon.titan-tg1-large"

    # INDEXING PARAMETERS
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 20
    
    loader = AzureBlobStorageContainerLoader(conn_str=os.getenv("CONNECTION_STRING"), container="sample")
    
    print("Splitting into Chunks...")
    text_splitter: TextSplitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    print("Now calling load_and_split...")
    documents = loader.load_and_split(text_splitter=text_splitter)
    print("Documents loaded...")

    print("Now calling embeddings...")

    embeddings = BedrockEmbeddings(client=bedrock, credentials_profile_name="bedrock-admin", model_id=BEDROCK_EMBEDDING_MODEL_NAME)

    print("Embeddings done...")

    delete_table("langchain_pg_embedding")

    db = PGVector.from_documents(
    embedding=embeddings,
    documents=documents,
    connection_string=CONNECTION_STRING
    ) 

    end_time = time.time()

    time_taken = end_time - start_time
    minutes, seconds = divmod(time_taken, 60)
    time_msg = f"Time taken: {int(minutes)} minutes {int(seconds)} seconds"
    print(time_msg)

    return time_msg


if __name__ == "__main__":
    create_vectors("roche")
