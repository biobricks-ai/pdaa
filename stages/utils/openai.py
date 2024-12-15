import sys
sys.path.append('.')
import dotenv
import openai
import stages.utils.simple_cache as simple_cache
import pathlib
dotenv.load_dotenv()

cache_dir = pathlib.Path('cache/util/openai') 
cache_dir.mkdir(parents=True, exist_ok=True)

client = openai.OpenAI()
@simple_cache.simple_cache(cache_dir / 'embed')
def embed(text):
    return client.embeddings.create(input=text, model="text-embedding-3-large").data[0].embedding