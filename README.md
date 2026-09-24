# Guest Notes — Hotel Review Studio

A Streamlit app with two tabs:

1. **Ask the reviews:** open-ended Pinecone RAG chatbot across hotels, with follow-up questions and numbered review sources. No hotel dropdown.
2. **Create a promotion:** select a hotel from the promotion tab’s dropdown, find three guest-praised features, and create an email for the exact offer “2 days, 3 nights”. Generate a matching hero image with OpenAI `gpt-image-1`. Preview and download HTML, a MIME email draft with the image embedded, and the PNG. No email is sent.

The app uses a dark theme configured in `.streamlit/config.toml`. Enter an optional booking/enquiry URL to add a clickable email CTA. Without one, the email asks readers to contact the hotel. HTML exports use inline images for browser preview; `.eml` drafts use CID attachments for email clients.

## Run locally

Python 3.11+ is recommended; the dependency ranges also support Python 3.9.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Keep your existing `.env` beside `app.py` with real values for `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME`. Do not commit it. The app reads `hotel_reviews.json` or `Hotel Reviews.json` for its hotel selector. Set `HOTEL_REVIEWS_PATH` in `.env` to use another path.

Complete indexing in `hotel_reviews_rag.ipynb` first. In the sidebar, click **Connect / refresh reviews**, and select the `hotel-reviews-…` namespace printed by the notebook. Different dataset subsets have different namespaces. Hotels listed in the JSON may not yet be indexed if an upload stopped early.

The app uses the notebook's existing 512-dimensional cosine index and `text-embedding-3-small` embeddings. It does not upload or delete vectors. Chat uses `gpt-4.1-mini`. Chat and analysis run only on user submission; image generation runs only when its button is clicked. Downloading an image does not regenerate it. Changing the promotion hotel clears its creative; the open-ended conversation is preserved. Changing the review collection clears both.

If OpenAI reports **no credits remaining**, add API credits before trying again. Both query embeddings and image generation require available credits; retries cannot fix exhausted credits. Your API project must also have access to the selected image model.

Positive features are drawn from a deduplicated sample of up to 60 retrieved reviews, not a whole-dataset statistical ranking. If the sample cannot support three distinct features, generation stops. Promotions are illustrations based on historical guest opinions, not actual property photographs. Check the generated wording before publishing.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests mock external services and do not consume API credits.
