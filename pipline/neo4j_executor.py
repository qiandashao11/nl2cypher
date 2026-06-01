# ==================== neo4j_executor.py ====================
"""
Neo4j查询执行器
执行Cypher查询并返回结构化结果
"""
from neo4j import GraphDatabase
from typing import Dict, List, Any


class Neo4jExecutor:
    """Neo4j查询执行器"""
    
    def __init__(self, 
                 uri: str = "neo4j://localhost:7687",
                 user: str = "neo4j",
                 password: str = ""):
        """
        初始化Neo4j连接
        
        Args:
            uri: Neo4j服务器地址
            user: 用户名
            password: 密码
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
        执行Cypher查询
        
        Args:
            cypher: Cypher查询字符串
            database: 数据库名称
            parameters: 查询参数
            
        Returns:
            包含查询结果的字典:
            {
                "success": bool,
                "data": List[dict],
                "count": int,
                "error": str (仅在失败时)
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
        """测试连接"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS num")
                return result.single()["num"] == 1
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        self.driver.close()
        print("✅ Neo4j connection closed")
    
    def __enter__(self):
        """支持with语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持with语句"""
        self.close()


if __name__ == "__main__":
    # 测试
    with Neo4jExecutor() as executor:
        if executor.test_connection():
            print("✅ Connection test passed")
            
            result = executor.execute("MATCH (n) RETURN count(n) AS count LIMIT 1")
            if result["success"]:
                print(f"✅ Query executed: {result['data']}")
            else:
                print(f"❌ Query failed: {result['error']}")