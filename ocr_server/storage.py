import sqlite3
import json
import os
from datetime import datetime

# 数据库放在 target 目录
DB_DIR = os.path.join(os.path.dirname(__file__), "target")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "recognitions.db")


def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recognitions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            image_name TEXT,
            image_path TEXT,
            annotated_image_path TEXT,
            status TEXT DEFAULT 'pending',
            result_json TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_recognition(rec_id: str, image_name: str, image_filename: str, annotated_filename: str, result_json: dict, status: str = "success"):
    """保存识别结果"""
    created_at = datetime.now().isoformat()

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO recognitions (id, created_at, image_name, image_path, annotated_image_path, status, result_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (rec_id, created_at, image_name, image_filename, annotated_filename, status, json.dumps(result_json, ensure_ascii=False))
    )
    conn.commit()
    conn.close()


def update_recognition_status(rec_id: str, status: str):
    """更新识别状态"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE recognitions SET status = ? WHERE id = ?", (status, rec_id))
    conn.commit()
    conn.close()


def get_recognition(rec_id: str) -> dict | None:
    """根据 ID 获取识别结果"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT * FROM recognitions WHERE id = ?", (rec_id,)
    )
    row = cur.fetchone()
    conn.close()

    if row:
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "image_name": row["image_name"],
            "image_path": row["image_path"],
            "annotated_image_path": row["annotated_image_path"],
            "status": row["status"],
            "result_json": json.loads(row["result_json"])
        }
    return None


def list_recognitions(limit: int = 50) -> list:
    """列出最近的识别记录"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT id, created_at, image_name, image_path, annotated_image_path, status FROM recognitions ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()

    return [{"id": row["id"], "created_at": row["created_at"], "image_name": row["image_name"], "image_path": row["image_path"], "annotated_image_path": row["annotated_image_path"], "status": row["status"]} for row in rows]


# 初始化
init_db()
