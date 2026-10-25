import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


def test_crete_collection():
    collection_name = "students"
    # 连接串从环境变量读取，禁止硬编码账号密码
    mongo_url = os.getenv("MONGO_URL")
    if not mongo_url:
        raise EnvironmentError("缺少必需的环境变量: MONGO_URL")
    client = MongoClient(mongo_url)
    # 连接数据库（不存在时自动创建）
    db = client["my_db"]
    # 选择集合
    collection = db[collection_name]

    print("连接成功")
    return collection


def test_insert_into_collection(collection):
    # 插入一条记录
    result = collection.insert_one({
        "name": "zhangsan",
        "age": 22,
        "major": "计算机科学"
    })
    print(f"插入成功，ID：{result.inserted_id}")
    # 插入多久条记录
    results = collection.insert_many([
        {"name": "李四", "age": 22, "major": "软件工程"},
        {"name": "王五", "age": 21, "major": "计算机科学"}
    ])
    print(f"插入 {len(results.inserted_ids)} 条记录")


def test_query(collection):
    # 查询全部
    for doc in collection.find():
        print(doc)


if __name__ == "__main__":

    collection = test_crete_collection()

    # test_insert_into_collection(collection)

    test_query(collection)
