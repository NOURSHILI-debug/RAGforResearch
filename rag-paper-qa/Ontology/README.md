# Ontology

Place the NeuroAI ontology file here (`neuroAI.owx` in OWL/XML format).

The ontology is parsed by `src/ontology_query.py` using Python's standard library XML parser. It extracts:
- Classes with human-readable labels
- Individuals with human-readable labels and types
- Object properties with human-readable labels
- Object property assertions (relationships)

Query it using:
```python
from ontology_query import OntologyQuery
oq = OntologyQuery()
oq.list_classes()
oq.list_individuals()
oq.get_relations(property_name='inspires')
oq.query_by_concept('hippocampal')
```
