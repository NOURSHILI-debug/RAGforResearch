from lxml import etree
tree = etree.parse(r'C:\Users\SINORFI\Desktop\RAGforResearch\rag-paper-qa\Ontology\neuroAI.owx')
ns = {'owl': 'http://www.w3.org/2002/07/owl#', 'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'}
count = 0
for ann in tree.xpath('//owl:AnnotationAssertion', namespaces=ns):
    prop = ann.xpath('owl:AnnotationProperty/@abbreviatedIRI', namespaces=ns)
    if prop and 'rdfs:label' in prop[0]:
        iri = ann.xpath('owl:IRI/text()', namespaces=ns)
        lit = ann.xpath('owl:Literal/text()', namespaces=ns)
        if iri and lit:
            print(iri[0] + ' => ' + lit[0])
            count += 1
            if count > 50: break
