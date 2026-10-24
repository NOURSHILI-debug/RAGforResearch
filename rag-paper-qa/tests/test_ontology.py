import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ontology_query import OntologyQuery


def test_load():
    oq = OntologyQuery()
    assert len(oq.list_classes()) >= 1
    assert len(oq.list_individuals()) >= 1
    assert len(oq.list_properties()) >= 1


def test_relations():
    oq = OntologyQuery()
    rels = oq.get_relations()
    assert isinstance(rels, list)
    if rels:
        assert 'property' in rels[0]
        assert 'subject' in rels[0]
        assert 'object' in rels[0]


def test_query_concept():
    oq = OntologyQuery()
    ctx = oq.query_by_concept('hippocampal')
    assert 'relations' in ctx
    assert 'matches' in ctx
