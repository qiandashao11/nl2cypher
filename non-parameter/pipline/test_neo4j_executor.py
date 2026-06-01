# test_neo4j_executor.py
from neo4j_executor import Neo4jExecutor

# Use a with-statement to manage the database connection
with Neo4jExecutor(uri="neo4j://localhost:7687", user="neo4j", password="") as executor:
    # Test Neo4j connection
    if executor.test_connection():
        print("✅ Connection test passed")
        
        # Execute the Cypher query
        result = executor.execute("MATCH (n) RETURN count(n) AS count LIMIT 1")
        
        if result["success"]:
            print(f"✅ Query executed: {result['data']}")
        else:
            print(f"❌ Query failed: {result['error']}")
    else:
        print("❌ Connection test failed")
