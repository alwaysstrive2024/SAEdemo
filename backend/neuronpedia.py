import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
NEURONPEDIA_API_KEY = os.getenv("X-API-KEY")
NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org/api/feature"

# Cache to avoid redundant API calls
# Key: (model_id, sae_id, feature_id)
_LABEL_CACHE: Dict[tuple, str] = {}

def fetch_single_label(model_id: str, sae_id: str, feature_id: int) -> tuple[int, str]:
    cache_key = (model_id, sae_id, feature_id)
    if cache_key in _LABEL_CACHE:
        return feature_id, _LABEL_CACHE[cache_key]

    url = f"{NEURONPEDIA_BASE_URL}/{model_id}/{sae_id}/{feature_id}"
    try:
        response = requests.get(
            url, 
            headers={'x-api-key': NEURONPEDIA_API_KEY}, 
            timeout=5.0
        )
        if response.status_code == 200:
            data = response.json()
            explanation = data.get("explanation")
            
            if explanation:
                label = explanation
            else:
                # Fallback to pos_str top 3 words
                pos_str = data.get("pos_str", [])
                if isinstance(pos_str, list) and len(pos_str) > 0:
                    label = ", ".join(str(x) for x in pos_str[:3])
                elif isinstance(pos_str, str) and pos_str:
                    label = pos_str
                else:
                    label = f"Concept {feature_id}"
            
            _LABEL_CACHE[cache_key] = label
            return feature_id, label
        else:
            return feature_id, f"Concept {feature_id}"
    except Exception as e:
        print(f"[Neuronpedia] API error for feature {feature_id}: {e}")
        return feature_id, f"Concept {feature_id}"

def get_concept_labels_batch(model_id: str, sae_id: str, feature_ids: List[int]) -> Dict[int, str]:
    """Fetch labels for multiple feature IDs concurrently."""
    if not feature_ids:
        return {}
        
    if not model_id or not sae_id:
        return {fid: f"Concept {fid}" for fid in feature_ids}

    results = {}
    # Use ThreadPoolExecutor to fetch in parallel (max 20 workers to be polite to the API)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(fetch_single_label, model_id, sae_id, fid): fid 
            for fid in feature_ids
        }
        for future in as_completed(futures):
            fid, label = future.result()
            results[fid] = label
            
    return results
