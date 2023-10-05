import textwrap
import sys

import psycopg
import torch
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration


question = sys.argv[1]

#
# Connect to the database, and create our objects
#
conn = psycopg.connect("dbname=pg_marvin user=postgres")
register_vector(conn)

#
# We use the same model we used to generate our embeddings
# originally to create the query to search the database.
#
device = 'cuda' if torch.cuda.is_available() else 'cpu'

retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)

#
# We'll use the BART tokenizer and generator to write the
# answer text from us, given the context retrieved from the
# database.
#
tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa').to(device)


def run_query(query, top_k):
    # Encode the query, and use it to find the closest matches in the database
    xq = retriever.encode([query]).tolist()
    res = conn.execute('SELECT * FROM wiki ORDER BY embedding <-> %s::vector LIMIT %s', (xq[0], top_k)).fetchall()
    return res


def format_query(query, context):
    # Add the <P> tag to all passages and then join them, then return the
    # question and passages for BART to generate an answer.
    ctx = [f"<P> {r[1]}" for r in context]
    ctx = " ".join(ctx)

    query = f"question: {query} context: {ctx}"
    return query


def generate_answer(query):
    # Tokenize the query to get the input IDs
    inputs = tokenizer([query], return_tensors="pt").to(device)

    # Use the generator to predict output IDs
    ids = generator.generate(inputs["input_ids"], num_beams=4,
                             length_penalty=2.0,
                             max_length=100,
                             min_length=5,
                             no_repeat_ngram_size=3)

    # Use the tokenizer to decode the output IDs to create the response
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return answer


context = run_query(question, top_k=5)
result = format_query(question, context)

print('== Result =====================================================================')
print('Question: \n{}\n'.format(textwrap.fill(question, width=78, initial_indent='    ', subsequent_indent='    ')))
print('Answer: \n{}'.format(
    textwrap.fill(generate_answer(result), width=78, initial_indent='    ', subsequent_indent='    ')))
print('===============================================================================')

#
# Cleanup
#
conn.close()
