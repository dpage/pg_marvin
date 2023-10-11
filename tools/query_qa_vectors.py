import argparse
import textwrap

import psycopg
import torch
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline


def read_command_line():
    parser = argparse.ArgumentParser(
        description='Ask a question to be answered from vectors of text stored in PostgreSQL.')

    parser.add_argument('store', help='the name of the vector store to use. '
                                      'Equates to a table in PostgreSQL')

    parser.add_argument('question', help='the question to be answered')

    parser.add_argument('-m', '--max', default=200, help='maximum length of the output in tokens (default: 200)')

    parser.add_argument('-s', '--summary', action='store_true', help='include summarised output')

    parser.add_argument('-t', '--top_k', default=5, help='number of top results to consider')

    parser.add_argument('-o', '--host', default='127.0.0.1',
                        help='the hostname or IP address of the PostgreSQL server (default: 127.0.0.1)')
    parser.add_argument('-p', '--port', default='5432',
                        help='the port number for the PostgreSQL server (default: 5432)')
    parser.add_argument('-d', '--db', default='postgres', help='the name of the database to use (default: postgres)')
    parser.add_argument('-u', '--user', default='postgres', help='the name of the database to use (default: postgres)')

    args = parser.parse_args()

    return args


def connect(args):
    c = psycopg.connect('host={} port={} dbname={} user={}'.format(args.host, args.port, args.db, args.user))
    register_vector(c)

    return c


def run_query(query, store, top_k):
    # We use the same model we used to generate our embeddings
    # originally to create the query to search the database.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base', device=device)

    # Encode the query, and use it to find the closest matches in the database
    xq = retriever.encode([query]).tolist()
    res = conn.execute('SELECT passage_text FROM {} ORDER BY embedding <-> %s::vector LIMIT %s'.format(store),
                       (xq[0], top_k)).fetchall()
    return res


def format_query(query, context):
    # Add the <P> tag to all passages and then join them, then return the
    # question and passages for BART to generate an answer.
    ctx = [f"<P> {r[0]}" for r in context]
    ctx = " ".join(ctx)

    query = f"question: {query} context: {ctx}"
    return query


def generate_answer(query, max_length):
    # Tokenize the query to get the input IDs
    inputs = tokenizer([query], return_tensors='pt').to(device)

    # Use the generator to predict output IDs
    ids = generator.generate(inputs['input_ids'], num_beams=4,
                             length_penalty=2.0,
                             max_length=max_length,
                             min_length=5,
                             no_repeat_ngram_size=3)

    # Use the tokenizer to decode the output IDs to create the response
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return answer


if __name__ == '__main__':
    args = read_command_line()

    # Connect to the database
    conn = connect(args)

    # We'll use the BART tokenizer and generator to write the
    # answer text from us, given the context retrieved from the
    # database.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
    generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa').to(device)

    context = run_query(args.question, args.store, top_k=args.top_k)
    result = format_query(args.question, context)
    answer = generate_answer(result, args.max)

    if args.summary:
        summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
        summary = summarizer(answer)[0]['summary_text']

    print('== Result =====================================================================')
    print('Question: \n{}'.format(
        textwrap.fill(args.question, width=78, initial_indent='    ', subsequent_indent='    ')))
    print('\nAnswer: \n{}'.format(
        textwrap.fill(answer, width=78, initial_indent='    ', subsequent_indent='    ')))
    if args.summary:
        print('\nSummary: \n{}'.format(
            textwrap.fill(summary, width=78, initial_indent='    ', subsequent_indent='    ')))
    print('===============================================================================')

    # Cleanup
    conn.close()
