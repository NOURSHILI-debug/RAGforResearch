import xml.etree.ElementTree as ET
tree = ET.parse(r'C:\Users\SINORFI\Desktop\RAGforResearch\rag-paper-qa\Ontology\neuroAI.owx')
root = tree.getroot()
ns = {'owl': 'http://www.w3.org/2002/07/owl#'}
print('classes:', len(root.findall('owl:Declaration/owl:Class', ns)))
print('props:', len(root.findall('owl:Declaration/owl:ObjectProperty', ns)))
print('individuals:', len(root.findall('owl:Declaration/owl:NamedIndividual', ns)))
print('classassert:', len(root.findall('owl:ClassAssertion', ns)))
print('objassert:', len(root.findall('owl:ObjectPropertyAssertion', ns)))
