# test_neo4j_executor.py
from neo4j_executor import Neo4jExecutor

# 使用 with 语句来管理数据库连接
with Neo4jExecutor(uri="neo4j://localhost:7687", user="neo4j", password="") as executor:
    # 测试 Neo4j 连接
    if executor.test_connection():
        print("✅ Connection test passed")
        
        # 执行 Cypher 查询
        result = executor.execute("MATCH (n) RETURN count(n) AS count LIMIT 1")
        
        if result["success"]:
            print(f"✅ Query executed: {result['data']}")
        else:
            print(f"❌ Query failed: {result['error']}")
    else:
        print("❌ Connection test failed")
