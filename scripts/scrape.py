# Huge credit to woctezuma on GitHub for the steamreviews code, it works flawlessly: https://github.com/woctezuma/download-steam-reviews.
import os
import steamreviews

# making a path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

app_id = 1771300  # KCD2 (use 379430 for KCD1)
request_params = {'language': 'english'}

review_dict, query_count = steamreviews.download_reviews_for_app_id(
    app_id,
    chosen_request_params=request_params,
    save_directory=DATA_DIR
)
