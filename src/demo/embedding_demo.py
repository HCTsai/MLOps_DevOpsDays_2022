import sys
sys.path.append('..')
from text2vec import SBert, cos_sim, semantic_search


embedder = SBert("hfl/chinese-roberta-wwm-ext")
cls_corpus= [
    '優勢強項 (strength)','劣勢弱項弱點 (weakness)',
    '機會潛力 (opportunity)','威脅競爭 (threat)'
]

out_class = ["strength","weakness","opportunity","threat"]

cls_embedding = embedder.encode(cls_corpus)

print (embedder.encode(["市場需求量極為龐大"]))


# Query sentences:
queries = ['市場需求量極為龐大',
    '我們有很強的執行力與完善的技術能力',
    '我們人才無法以支撐產品銷售',
    '我們抓到了一個商業模式切入點',
    '競爭者多，留給我們的時間不多']

for query in queries:
    query_embedding = embedder.encode(query)
    hits = semantic_search(query_embedding, cls_embedding, top_k=5)
    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")
    hits = hits[0]  # Get the hits for the first query
    for hit in hits:
        print(cls_corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))