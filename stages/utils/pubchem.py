import time
import requests
import stages.utils.simple_cache as simple_cache
import pathlib
import threading

cachedir = pathlib.Path('cache') / 'utils' / 'pubchem' 
cachedir.mkdir(parents=True, exist_ok=True)

CALLS = 1
RATE_LIMIT = 0.33

def lookup_chemical_inchi(chemical_name):
    # First get the PubChem CID by searching the name
    search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{chemical_name}/cids/JSON"
    response = requests.get(search_url)
    response.raise_for_status()

    # if it is a 404, return None
    if response.status_code == 404:
        return None
    
    cid = response.json()['IdentifierList']['CID'][0]
        
    # Then get the InChI using the CID
    inchi_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/InChI/JSON"
    response = requests.get(inchi_url)
    response.raise_for_status()
    return response.json()['PropertyTable']['Properties'][0]['InChI']

class PubchemTools:
    """A thread-safe wrapper for PubChem API calls with caching.
    
    Example:
        >>> # Create a PubchemTools instance
        >>> pubchem = PubchemTools()
        >>> 
        >>> # Look up InChI for aspirin
        >>> inchi = pubchem.lookup_chemical_inchi("aspirin")
        >>> print(inchi)
        'InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)'
        >>> 
        >>> # Subsequent lookups will use cached result
        >>> inchi2 = pubchem.lookup_chemical_inchi("aspirin")  # Returns cached value
        >>> assert inchi == inchi2
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        self.cache = simple_cache.load_cache(cachedir / 'lookup_chemical_inchi')

    def lookup_chemical_inchi(self, chemical_name):
        with self.lock:
            hash_key = simple_cache.get_cache_hash(lookup_chemical_inchi, chemical_name)
            if hash_key in self.cache:
                return self.cache[hash_key]
            
            # add a .33 second delay
            time.sleep(0.33)
            inchi = lookup_chemical_inchi(chemical_name)
            self.cache[hash_key] = inchi
            return inchi
    