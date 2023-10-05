import psycopg
import torch
from datasets import load_dataset
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

total_doc_count = 50000

#
# Connect to the database, and create our objects
#
conn = psycopg.connect("dbname=pg_marvin user=postgres")
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)
conn.execute('CREATE TABLE IF NOT EXISTS wiki (id bigserial PRIMARY KEY, passage_text text, embedding vector(768));')
conn.execute('DELETE FROM wiki;')

#
# Get the dataset. In this case, we'll use snippets from Wikipedia
# downloaded from HuggingFace
#
wiki_data = load_dataset('vblagoje/wikipedia_snippets_streamed', split='train', streaming=True).shuffle(seed=960)

# We only want 'History' docs
history = wiki_data.filter(lambda d: d['section_title'].startswith('History'))

# Create an array of the doc passages
count = 0
docs = []
for d in tqdm(history, total=total_doc_count, desc='Retrieving data'):
    docs.append(d["passage_text"])

    # Quit when we've got enough
    if count == total_doc_count:
        break

    count += 1

#
# Setup a transformer to generate vectors from our text
#
device = 'cuda' if torch.cuda.is_available() else 'cpu'

retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)

#
# Loop over the docs, and load them with embeddings into PostgreSQL
# TODO: Encode passages in batches, and then insert.
#
with conn.cursor() as cur:
    with cur.copy("COPY wiki (passage_text, embedding) FROM STDIN") as copy:
        for doc in tqdm(docs, total=len(docs), desc='Loading data   '):
            copy.write_row((doc, str(retriever.encode(doc).tolist())))

#
# Cleanup
#
conn.commit()
conn.close()
