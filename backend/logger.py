from pymongo import MongoClient
from datetime import datetime

try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["pest_detection"]
    collection = db["logs"]
except Exception as e:
    print(f"[Logger] Failed to connect to MongoDB: {e}")
    collection = None

def log_detection(class_name, confidence, source):
    """
    Logs a pest detection event into MongoDB.
    
    Args:
        class_name (str): Name of the detected class (e.g., 'Cockroach').
        confidence (float): Confidence score (0.0 to 1.0).
        source (str): Source of the detection ('Live', 'Image', etc.)
    """
    if collection is None:
        print("[Logger] No MongoDB connection. Skipping logging.")
        return

    try:
        log_data = {
            "class_name": class_name,
            "confidence": round(float(confidence), 4),
            "source": source,
            "timestamp": datetime.utcnow()
        }
        collection.insert_one(log_data)
        print(f"[Logger] Logged: {log_data}")
    except Exception as e:
        print(f"[Logger] Error logging detection: {e}")


def get_logs(limit=50):
    return list(collection.find().sort("timestamp", -1).limit(limit))