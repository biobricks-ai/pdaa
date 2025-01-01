import rdflib
from rdflib import RDF
from collections import defaultdict

def generate_alias(uri):
    """Helper function to create a valid alias for a URI."""
    return uri.split("/")[-1].replace("#", "_").replace(":", "_")

def rdf2uml(input_file, output_file="output.puml"):
    """
    Generates a PlantUML text file from an RDF/NT file, 
    showing only objects (instances of classes) and 
    relationships between them.

    Args:
        input_file (str): Path to the input NT file.
        output_file (str): Path to the output PlantUML file.
    """
    graph = rdflib.Graph()
    graph.parse(input_file, format='nt')

    diagram = """@startuml
skinparam linetype ortho
skinparam dpi 150
skinparam maxMessageSize 100
skinparam padding 2
skinparam roundcorner 5
skinparam defaultFontSize 12
skinparam arrowColor #666666
skinparam classAttributeIconSize 0
"""

    entity_classes = defaultdict(lambda: "external")
    for s, _, o in graph.triples((None, RDF.type, None)):
        entity_classes[s] = o

    class_predicates = defaultdict(set)
    for s, p, o in graph.triples((None, None, None)):
        class_predicates[entity_classes[s]].add(p)

    predicate_exclusions = {RDF.type}
    for class_name, predicates in class_predicates.items():
        alias = generate_alias(class_name)
        diagram += f"class {alias} as \"{class_name}\" {{\n"
        for predicate in predicates - predicate_exclusions:
            diagram += f"  \"{predicate}\"\n"
        diagram += "}\n"

    diagram += "class external { }\n"

    relationships = set()
    for s, p, o in graph.triples((None, None, None)):
        if isinstance(o, rdflib.Literal): continue
        source_alias = generate_alias(entity_classes[s])
        target_alias = generate_alias(entity_classes[o])
        relationships.add((source_alias, target_alias, p))

    # remove RDF.type
    relationships = [r for r in relationships if r[2] != RDF.type]
    for source_alias, target_alias, predicate in relationships:
        predicate_alias = generate_alias(predicate)
        diagram += f"{source_alias} --> {target_alias} : \"{predicate_alias}\"\n"

    diagram += "@enduml\n"

    with open(output_file, "w") as f:
        f.write(diagram)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert RDF to PlantUML diagram')
    parser.add_argument('input', help='Input RDF/NT file path')
    parser.add_argument('--output', default='output.puml', help='Output PlantUML file path')
    args = parser.parse_args()

    rdf2uml(args.input, args.output)
