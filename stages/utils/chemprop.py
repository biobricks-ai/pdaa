import aiohttp
import asyncio
import requests
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import wraps

def make_safe(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {"result": None, "error": str(e)}
    return wrapper


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_chemprop_prediction(inchi: str, property_token: str) -> dict:
    base_url = "http://chemprop-transformer-alb-2126755060.us-east-1.elb.amazonaws.com/predict"
    params = {"property_token": property_token, "inchi": inchi}
    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Raise exception for bad status codes
    return response.json()

def get_chemprop_prediction_safe(inchi: str, property_token: str, retries: int = 5, delay: int = 2) -> dict:
    return make_safe(get_chemprop_prediction)(inchi, property_token, retries, delay)

def chemprop_predict_all(inchi: str) -> list[dict]:
    base_url = "http://chemprop-transformer-alb-2126755060.us-east-1.elb.amazonaws.com/predict_all"
    params = {"inchi": inchi}
    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Raise exception for bad status codes
    return response.json()

async def chemprop_predict_all_async(inchi: str) -> dict:
    base_url = "http://chemprop-transformer-alb-2126755060.us-east-1.elb.amazonaws.com/predict_all"
    params = {"inchi": inchi}
    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            response.raise_for_status()  # Raise exception for bad status codes
            return await response.json()
                
async def get_chemprop_prediction_async(inchi: str, property_token: str, retries: int = 5, delay: int = 2) -> dict:
    """
    Get prediction from ChemProp API for a given InChI and property token with retry logic.
    
    Args:
        inchi: InChI string of the molecule
        property_token: Property token ID for the prediction
        retries: Number of retry attempts
        delay: Delay (in seconds) between retries
        
    Returns:
        dict: Dictionary containing 'result' and 'error' keys. Result contains the API response if successful,
              error contains error message if failed.
    """
    base_url = "http://chemprop-transformer-alb-2126755060.us-east-1.elb.amazonaws.com/predict"
    params = {
        "property_token": property_token,
        "inchi": inchi
    }

    async with aiohttp.ClientSession() as session:
        for attempt in range(1, retries + 1):
            try:
                async with session.get(base_url, params=params) as response:
                    response.raise_for_status()  # Raise exception for bad status codes
                    result = await response.json()
                    return {"result": result, "error": None}
            except Exception as e:
                if attempt < retries:
                    await asyncio.sleep(delay)  # Wait before retrying
                else:
                    return {"result": None, "error": f"Failed after {retries} retries: {str(e)}"}



# Example usage
# async def main(inchi="InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H", property_token="5042"):
#     result = await get_chemprop_prediction_async(inchi, property_token)
#     print(result)
# asyncio.run(main(inchi=inchi, property_token="5042"))