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
            result_json TEXT NOT NULL,
            rec_type TEXT DEFAULT 'ocr',
            deleted INTEGER DEFAULT 0
        )
    """)
    # 如果已存在表但没有 rec_type 列，则添加
    try:
        conn.execute("ALTER TABLE recognitions ADD COLUMN rec_type TEXT DEFAULT 'ocr'")
    except sqlite3.OperationalError:
        pass  # 列已存在
    # 如果已存在表但没有 deleted 列，则添加
    try:
        conn.execute("ALTER TABLE recognitions ADD COLUMN deleted INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # 列已存在
    conn.commit()
    conn.close()


def save_recognition(rec_id: str, image_name: str, image_filename: str, annotated_filename: str, result_json: dict, status: str = "success", rec_type: str = "ocr"):
    """保存识别结果"""
    created_at = datetime.now().isoformat()

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO recognitions (id, created_at, image_name, image_path, annotated_image_path, status, result_json, rec_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (rec_id, created_at, image_name, image_filename, annotated_filename, status, json.dumps(result_json, ensure_ascii=False), rec_type)
    )
    conn.commit()
    conn.close()


def update_recognition_status(rec_id: str, status: str):
    """更新识别状态"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE recognitions SET status = ? WHERE id = ? AND deleted = 0", (status, rec_id))
    conn.commit()
    conn.close()


def delete_recognition(rec_id: str):
    """软删除识别记录"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE recognitions SET deleted = 1 WHERE id = ?", (rec_id,))
    conn.commit()
    conn.close()


def get_recognition(rec_id: str) -> dict | None:
    """根据 ID 获取识别结果"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT * FROM recognitions WHERE id = ? AND deleted = 0", (rec_id,)
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
            "result_json": json.loads(row["result_json"]),
            "rec_type": row["rec_type"]
        }
    return None


def list_recognitions(limit: int = 50) -> list:
    """列出最近的识别记录"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT id, created_at, image_name, image_path, annotated_image_path, status, rec_type FROM recognitions WHERE deleted = 0 ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()

    return [{"id": row["id"], "created_at": row["created_at"], "image_name": row["image_name"], "image_path": row["image_path"], "annotated_image_path": row["annotated_image_path"], "status": row["status"], "rec_type": row["rec_type"]} for row in rows]


# 初始化
init_db()
