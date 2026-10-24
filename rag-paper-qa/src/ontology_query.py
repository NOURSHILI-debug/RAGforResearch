from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional


OWL = "http://www.w3.org/2002/07/owl#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"


def _iri_local(iri: str) -> str:
    if "#" in iri:
        return iri.split("#")[-1]
    if "/" in iri:
        return iri.rstrip("/").split("/")[-1]
    return iri


class OntologyQuery:
    def __init__(self, ontology_path: Path | str = None):
        if ontology_path is None:
            # Try common locations
            candidates = [
                Path("Ontology/neuroAI.owx"),
                Path("../Ontology/neuroAI.owx"),
                Path("rag-paper-qa/Ontology/neuroAI.owx"),
            ]
            self.path = None
            for c in candidates:
                if c.exists():
                    self.path = c
                    break
            if self.path is None:
                # also try relative to this file
                self.path = Path(__file__).parent.parent / "Ontology" / "neuroAI.owx"
        else:
            self.path = Path(ontology_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Ontology not found: {self.path}")
        try:
            self.tree = ET.parse(str(self.path))
        except Exception as e:
            raise ValueError(f"Failed to parse ontology as XML: {e}")
        self.root = self.tree.getroot()
        self.ns = {"owl": OWL, "rdfs": RDFS}
        self._load()

    def _load(self):
        # labels
        self.labels: Dict[str, str] = {}
        for ann in self.root.findall("owl:AnnotationAssertion", self.ns):
            prop = ann.find("owl:AnnotationProperty", self.ns)
            if prop is None:
                continue
            prop_iri = prop.attrib.get("{%s}abbreviatedIRI" % OWL) or prop.attrib.get("abbreviatedIRI") or ""
            if "rdfs:label" not in prop_iri:
                continue
            iri_el = ann.find("owl:IRI", self.ns)
            if iri_el is None:
                continue
            iri = iri_el.attrib.get("IRI") or (iri_el.text or "").strip()
            if not iri:
                continue
            lit = ann.find("owl:Literal", self.ns)
            if lit is None or not lit.text:
                continue
            self.labels[iri] = lit.text.strip()

        # declarations
        self.classes: Dict[str, str] = {}
        for decl in self.root.findall("owl:Declaration/owl:Class", self.ns):
            iri = decl.attrib.get("IRI")
            if iri:
                self.classes[iri] = self.labels.get(iri, _iri_local(iri))

        self.object_props: Dict[str, str] = {}
        for decl in self.root.findall("owl:Declaration/owl:ObjectProperty", self.ns):
            iri = decl.attrib.get("IRI")
            if iri:
                self.object_props[iri] = self.labels.get(iri, _iri_local(iri))

        self.individuals: Dict[str, str] = {}
        for decl in self.root.findall("owl:Declaration/owl:NamedIndividual", self.ns):
            iri = decl.attrib.get("IRI")
            if iri:
                self.individuals[iri] = self.labels.get(iri, _iri_local(iri))

        # class assertions
        self.individual_types: Dict[str, List[str]] = {}
        for ca in self.root.findall("owl:ClassAssertion", self.ns):
            cls_el = ca.find("owl:Class", self.ns)
            ind_el = ca.find("owl:NamedIndividual", self.ns)
            if cls_el is None or ind_el is None:
                continue
            cls_iri = cls_el.attrib.get("IRI")
            ind_iri = ind_el.attrib.get("IRI")
            if not cls_iri or not ind_iri:
                continue
            self.individual_types.setdefault(ind_iri, []).append(cls_iri)

        # object property assertions
        self.relations: List[Dict[str, str]] = []
        for opa in self.root.findall("owl:ObjectPropertyAssertion", self.ns):
            prop_el = opa.find("owl:ObjectProperty", self.ns)
            subj_el = opa.find("owl:NamedIndividual", self.ns)
            # there might be multiple NamedIndividual elements (subject and object)
            inds = opa.findall("owl:NamedIndividual", self.ns)
            if prop_el is None or len(inds) < 2:
                continue
            prop_iri = prop_el.attrib.get("IRI")
            subj_iri = inds[0].attrib.get("IRI")
            obj_iri = inds[1].attrib.get("IRI")
            if not prop_iri or not subj_iri or not obj_iri:
                continue
            self.relations.append(
                {
                    "property_iri": prop_iri,
                    "property": self.object_props.get(prop_iri, self.labels.get(prop_iri, _iri_local(prop_iri))),
                    "subject_iri": subj_iri,
                    "subject": self.individuals.get(subj_iri, self.labels.get(subj_iri, _iri_local(subj_iri))),
                    "object_iri": obj_iri,
                    "object": self.individuals.get(obj_iri, self.labels.get(obj_iri, _iri_local(obj_iri))),
                }
            )

    def list_classes(self) -> List[Dict[str, str]]:
        return [{"iri": k, "label": v} for k, v in sorted(self.classes.items(), key=lambda x: x[1])]

    def list_individuals(self) -> List[Dict[str, str]]:
        return [{"iri": k, "label": v, "types": [self.classes.get(t, _iri_local(t)) for t in self.individual_types.get(k, [])]} for k, v in sorted(self.individuals.items(), key=lambda x: x[1])]

    def list_properties(self) -> List[Dict[str, str]]:
        return [{"iri": k, "label": v} for k, v in sorted(self.object_props.items(), key=lambda x: x[1])]

    def get_relations(self, property_name: Optional[str] = None, subject: Optional[str] = None, object: Optional[str] = None) -> List[Dict[str, str]]:
        results = self.relations
        if property_name:
            pn = property_name.lower()
            results = [r for r in results if pn in r["property"].lower()]
        if subject:
            sn = subject.lower()
            results = [r for r in results if sn in r["subject"].lower() or sn in r["subject_iri"].lower()]
        if object:
            on = object.lower()
            results = [r for r in results if on in r["object"].lower() or on in r["object_iri"].lower()]
        return results

    def query_by_concept(self, concept: str) -> Dict[str, Any]:
        concept_l = concept.lower()
        # find matching individuals/classes
        matches = []
        for iri, label in self.individuals.items():
            if concept_l in label.lower() or any(tok in label.lower() for tok in concept_l.split() if len(tok) > 3):
                if (iri, label) not in [(m[1], m[2]) for m in matches]:
                    matches.append(("individual", iri, label))
        for iri, label in self.classes.items():
            if concept_l in label.lower() or any(tok in label.lower() for tok in concept_l.split() if len(tok) > 3):
                if (iri, label) not in [(m[1], m[2]) for m in matches]:
                    matches.append(("class", iri, label))
        rels = []
        # get all relations involving any match
        match_iris = {m[1] for m in matches}
        for r in self.relations:
            if (concept_l in r["subject"].lower() or concept_l in r["object"].lower() or concept_l in r["property"].lower() or
                    r["subject_iri"] in match_iris or r["object_iri"] in match_iris):
                rels.append(r)
        # deduplicate
        seen = set()
        uniq = []
        for r in rels:
            key = (r["subject_iri"], r["property_iri"], r["object_iri"])
            if key not in seen:
                uniq.append(r)
                seen.add(key)
        return {"matches": matches, "relations": uniq}
