import os
import sys
import time
import pickle
# Add custom libs path
sys.path.insert(0, r"C:\tmp\libs")

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils import process_and_index_document, save_index, load_index, RAW_DIR, INDEX_DIR, DATA_DIR

METADATA_FILE = os.path.join(DATA_DIR, "metadata.pkl")

class NewDocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        filepath = event.src_path
        filename = os.path.basename(filepath)
        
        # Filter for supported extensions
        if not filename.lower().endswith(('.txt', '.pdf', '.docx')):
            return
            
        print(f"New file detected: {filename}")
        self.process_file(filepath)

    def process_file(self, filepath):
        # Load current state
        index_path = os.path.join(INDEX_DIR, "faiss.index")
        index = load_index(index_path)
        
        metadata = {}
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "rb") as f:
                metadata = pickle.load(f)
        
        print("Processing...")
        try:
            # Wait a brief moment to ensure file write is complete
            time.sleep(1)
            
            index, metadata, error = process_and_index_document(filepath, index, metadata)
            
            if error:
                print(f"Error processing {filepath}: {error}")
            else:
                # Save state
                save_index(index, index_path)
                with open(METADATA_FILE, "wb") as f:
                    pickle.dump(metadata, f)
                print(f"Successfully indexed {filepath}")
                
        except Exception as e:
            print(f"Exception during processing: {e}")

def start_watching():
    event_handler = NewDocumentHandler()
    observer = Observer()
    observer.schedule(event_handler, RAW_DIR, recursive=False)
    observer.start()
    print(f"Watching for new documents in {RAW_DIR}...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # Ensure raw dir exists
    os.makedirs(RAW_DIR, exist_ok=True)
    start_watching()
