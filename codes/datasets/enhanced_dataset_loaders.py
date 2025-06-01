import json
from typing import List, Dict, Any, Optional

class DatasetLoader:
    def load(self, path: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def preprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError

class TriviaQALoader(DatasetLoader):
    def load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def preprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []
        for item in data:
            processed.append({
                "question": item.get("question"),
                "answer": item.get("answer"),
                "passages": item.get("passages", [])
            })
        return processed

class HotpotQALoader(DatasetLoader):
    def load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def preprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []
        for item in data:
            processed.append({
                "question": item.get("question"),
                "context": item.get("context", []),
                "answer": item.get("answer"),
                "supporting_facts": item.get("supporting_facts", [])
            })
        return processed

class BioASQLoader(DatasetLoader):
    def load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def preprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []
        for item in data:
            processed.append({
                "question": item.get("question"),
                "context": item.get("context", ""),
                "answer": item.get("exact_answer", ""),
                "documents": item.get("documents", [])
            })
        return processed

class SciFactLoader(DatasetLoader):
    def load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def preprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []
        for item in data:
            processed.append({
                "claim": item.get("claim"),
                "evidence": item.get("evidence", []),
                "label": item.get("label", "")
            })
        return processed

def get_loader(dataset_name: str) -> Optional[DatasetLoader]:
    loaders = {
        "hotpotqa": HotpotQALoader(),
        "bioasq": BioASQLoader(),
        "scifact": SciFactLoader(),
        "triviaq": TriviaQALoader(),
    }
    return loaders.get(dataset_name.lower())

# Example usage:
# loader = get_loader("hotpotqa")
# data = loader.load("path/to/hotpotqa.json")
# processed_data = loader.preprocess(data)