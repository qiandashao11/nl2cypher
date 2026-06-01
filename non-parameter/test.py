from neo4j_executor import Neo4jExecutor

executor = Neo4jExecutor(
    uri="neo4j://localhost:7687",
    user="neo4j",
    password="你的密码"
)

# 测试连接
if executor.test_connection():
    print("连接成功")

# 执行查询
result = executor.execute("MATCH (:Gene {ENTITY:'ACSL3'})-[:CO_OCCURS]-(m:MeSH) RETURN DISTINCT m.ENTITY AS mesh_entity ORDER BY m.ENTITY ASC")

print(result)

executor.close()
