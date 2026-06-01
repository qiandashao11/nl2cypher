from neo4j_executor import Neo4jExecutor

with Neo4jExecutor() as executor:
    result = executor.execute("MATCH (a)-[r]->(b) RETURN DISTINCT type(r) AS rel_type, labels(a) AS from_labels, labels(b) AS to_labels, keys(r) AS rel_property_keys LIMIT 50;")
    print(result)
