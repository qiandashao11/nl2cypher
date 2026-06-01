# ==================== neo4j_executor.py ====================
"""
Neo4j query executor
Execute Cypher queries and return structured results
"""
from neo4j import GraphDatabase
from typing import Dict, List, Any


class Neo4jExecutor:
    """Neo4j query executor"""
    
    def __init__(self, 
                 uri: str = "neo4j://localhost:7687",
                 user: str = "neo4j",
                 password: str = ""):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j server URI
            user: username
            password: password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"✅ Connected to Neo4j: {uri}")
    
    def execute(self, 
                cypher: str, 
                database: str = "neo4j",
                parameters: dict = None) -> Dict[str, Any]:
        """
        Execute Cypher query
        
        Args:
            cypher: Cypher query string
            database: database name
            parameters: query parameters
            
        Returns:
            dictionary containing query results:
            {
                "success": bool,
                "data": List[dict],
                "count": int,
                "error": str (only on failure)
            }
        """
        try:
            with self.driver.session(database=database) as session:
                result = session.run(cypher, parameters or {})
                records = [record.data() for record in result]
                
                return {
                    "success": True,
                    "data": records,
                    "count": len(records)
                }
        except Exception as e:
            return {
                "success": False,
                "data": [],
                "count": 0,
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """Test connection"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS num")
                return result.single()["num"] == 1
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def close(self):
        """Close connection"""
        self.driver.close()
        print("✅ Neo4j connection closed")
    
    def __enter__(self):
        """Support with-statements"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support with-statements"""
        self.close()


if __name__ == "__main__":
    # Test
    with Neo4jExecutor() as executor:
        if executor.test_connection():
            print("✅ Connection test passed")
            
            result = executor.execute("MATCH (n) RETURN count(n) AS count LIMIT 1")
            if result["success"]:
                print(f"✅ Query executed: {result['data']}")
            else:
                print(f"❌ Query failed: {result['error']}")