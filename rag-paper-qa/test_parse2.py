from rdflib import Graph
g = Graph()
try:
    g.parse(r'C:\Users\SINORFI\Desktop\RAGforResearch\rag-paper-qa\Ontology\neuroAI.owx', format='xml')
    print('ok', len(g))
except Exception as e:
    print('err')
    print(e)
