import argparse
import math

import fitz
import psycopg
import torch
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def read_command_line():
    parser = argparse.ArgumentParser(
        description='Preprocess and load vectors of text into PostgreSQL.')

    parser.add_argument('store', help='the name of the vector store to use. '
                                      'Equates to a table in PostgreSQL')

    parser.add_argument('file', help='the file to load text from')

    parser.add_argument('-c', '--clear', action='store_true', help='clear existing data from the store before loading')

    parser.add_argument('-o', '--host', default='127.0.0.1',
                        help='the hostname or IP address of the PostgreSQL server (default: 127.0.0.1)')
    parser.add_argument('-p', '--port', default='5432',
                        help='the port number for the PostgreSQL server (default: 5432)')
    parser.add_argument('-d', '--db', default='postgres', help='the name of the database to use (default: postgres)')
    parser.add_argument('-u', '--user', default='postgres', help='the name of the database to use (default: postgres)')

    args = parser.parse_args()

    return args


def connect(args):
    # Connect to the database, and create our objects
    c = psycopg.connect('host={} port={} dbname={} user={}'.format(args.host, args.port, args.db, args.user))
    c.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(c)
    c.execute(
        'CREATE TABLE IF NOT EXISTS {} (id bigserial PRIMARY KEY, passage_text text, embedding vector(768));'.format(
            args.store))

    if args.clear:
        c.execute('DELETE FROM {};'.format(args.store))

    return c


def load_pdf(file):
    pdf = fitz.open(file)

    d = []
    for page in pdf:
        blocks = page.get_text('blocks')
        for block in blocks:
            # Make sure there are no NULLs
            t = block[4].replace('\x00', '\\0')

            # Is this long enough?
            if len(t.split()) > 20:
                d.append({'text': t})

    pdf.close()

    return d


def load_txt(file):
    f = open(file, 'r')
    raw = f.read()

    d = raw.split('\n\n')

    def remove_nulls(s):
        return s.replace('\x00', '\\0')

    def dictify(s):
        return {'text': s}

    # Remove short items
    d = list(filter(lambda x: len(x.split()) >= 20, d))

    # Make sure there are no NULLs
    d = list(map(remove_nulls, d))

    # Make each element into the required dictionary
    d = list(map(dictify, d))

    f.close()

    return d


if __name__ == '__main__':
    args = read_command_line()

    conn = connect(args)

    # Load the data
    if args.file.endswith('.pdf'):
        docs = load_pdf(args.file)
    if args.file.endswith('.txt'):
        docs = load_txt(args.file)
    else:
        print('Cannot determine format for {}. The filename must end with .pdf or .txt.'.format(args.file))

    # Setup a transformer to generate vectors from our text
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base', device=device)

    # Loop over the docs, and create the embeddings in batches
    batch_size = 256
    for i in tqdm(range(0, len(docs), batch_size), total=math.ceil(len(docs) / batch_size),
                  desc='Generating embeddings'):
        batch = docs[i:i + batch_size]

        text = [d.get('text', None) for d in batch]
        embeddings = retriever.encode(text).tolist()

        p = i
        for e in embeddings:
            docs[p]['embedding'] = e
            p = p + 1

    # Load the database
    with conn.cursor() as cur:
        with cur.copy('COPY {} (passage_text, embedding) FROM STDIN'.format(args.store)) as copy:
            for doc in tqdm(docs, total=len(docs), desc='Loading data         '):
                copy.write_row((doc['text'], str(doc['embedding'])))

    # Cleanup
    conn.commit()
    conn.close()
