"""
database.py — Database Layer for NeuroScan AI (MySQL with SQLite Fallback)
Connected to local MySQL Server (root/root @ localhost:3306 -> neuroscan_db)

Provides persistence for:
1. User authentication & roles (Doctor, Patient, Admin) with secure password hashing.
2. Patient demographic profiles & Medical Record Numbers (MRN).
3. Diagnostic MRI scan history & volumetric morphology metrics.
4. Generated DICOM-grade PDF reports stored as LONGBLOBs.
5. Comprehensive system error logs & guardrail rejections.
"""

import hashlib
import json
import os
import datetime
import traceback
import re
import bcrypt
import sqlite3
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# MySQL / AWS RDS Configuration (dynamically loaded from .env)
MYSQL_HOST = os.getenv("DB_HOST", "localhost")
MYSQL_USER = os.getenv("DB_USER", "root")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD", "root")
MYSQL_PORT = int(os.getenv("DB_PORT", 3306))
MYSQL_DB = os.getenv("DB_NAME", "neuroscan_db")

# SQLite Fallback Path
SQLITE_DB_PATH = r"C:\TumorOI\neuroscan.db"

# Flag to track active DB backend
USE_MYSQL = True

try:
    import pymysql
    import pymysql.cursors
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False
    USE_MYSQL = False


def get_db_connection():
    """Returns an active MySQL connection or falls back to SQLite."""
    global USE_MYSQL
    if PYMYSQL_AVAILABLE and USE_MYSQL:
        try:
            conn = pymysql.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                port=MYSQL_PORT,
                database=MYSQL_DB,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True
            )
            return conn, "mysql"
        except Exception:
            # Fallback to SQLite if MySQL service is not running or DB not ready
            pass

    # SQLite Fallback
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"


# Password Policy Configuration
MIN_PASSWORD_LENGTH = 8
COMMON_WEAK_PASSWORDS = {
    "password", "password123", "password1234", "123456", "12345678", "123456789",
    "12345", "qwerty", "qwertyuiop", "admin", "admin123", "administrator",
    "doctor", "patient", "welcome", "welcome1", "neuroscan", "neuro2025",
    "brain123", "patient123", "hospital", "medical", "clinic123", "iloveyou",
    "secret", "pass123", "letmein", "default"
}


def validate_password_policy(password: str, username: str = "", full_name: str = "") -> tuple[bool, list[str]]:
    """
    Validates a password against clinical security & complexity policies:
    1. Minimum 8 characters
    2. At least 1 uppercase letter (A-Z)
    3. At least 1 lowercase letter (a-z)
    4. At least 1 numeric digit (0-9)
    5. At least 1 special character (!@#$%^&*...)
    6. Not in common weak passwords dictionary
    7. Does not contain username or full name substrings
    
    Returns:
        (is_valid: bool, issues: list[str])
    """
    issues = []

    if not password:
        return False, ["Password cannot be empty."]

    if len(password) < MIN_PASSWORD_LENGTH:
        issues.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long (currently {len(password)}).")

    if not re.search(r"[A-Z]", password):
        issues.append("Password must contain at least one uppercase letter (A-Z).")

    if not re.search(r"[a-z]", password):
        issues.append("Password must contain at least one lowercase letter (a-z).")

    if not re.search(r"\d", password):
        issues.append("Password must contain at least one numeric digit (0-9).")

    if not re.search(r"[!@#$%^&*(),.?\":{}|<>_+\-=[\]/\\;`~]", password):
        issues.append("Password must contain at least one special character (!@#$%^&*...).")

    # Weak password blacklist check
    pwd_clean = password.lower().strip()
    if pwd_clean in COMMON_WEAK_PASSWORDS:
        issues.append("Password is too common or easily guessable. Please choose a stronger passphrase.")

    # Check if username is contained inside password
    if username and len(username) >= 3 and username.lower() in pwd_clean:
        issues.append("Password must not contain your username.")

    # Check if name is contained inside password
    if full_name:
        for part in full_name.lower().split():
            if len(part) >= 4 and part in pwd_clean:
                issues.append("Password must not contain parts of your name.")
                break

    return (len(issues) == 0, issues)


def hash_password(password: str) -> str:
    """
    Securely hash a password using bcrypt with a cryptographic salt (work factor 12).
    Falls back to Argon2 or PBKDF2-HMAC-SHA256 if bcrypt is unavailable.
    """
    if not isinstance(password, str):
        password = str(password)

    # Primary: bcrypt
    try:
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")
    except Exception:
        pass

    # Secondary: Argon2
    try:
        from argon2 import PasswordHasher
        ph = PasswordHasher()
        return ph.hash(password)
    except Exception:
        pass

    # Tertiary Fallback: PBKDF2-HMAC-SHA256 (600,000 iterations)
    salt_bytes = os.urandom(16)
    kdf = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 600000)
    return f"pbkdf2_sha256$600000${salt_bytes.hex()}${kdf.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> tuple[bool, bool]:
    """
    Verifies a plaintext password against a stored hash.
    Supports bcrypt ($2b$, $2a$), Argon2 ($argon2id$), PBKDF2, and legacy salted SHA-256.
    
    Returns:
        (is_valid: bool, needs_rehash: bool)
    """
    if not plain_password or not hashed_password:
        return False, False

    if not isinstance(plain_password, str):
        plain_password = str(plain_password)

    # 1. Bcrypt verification
    if hashed_password.startswith(("$2b$", "$2a$", "$2y$")):
        try:
            valid = bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
            return valid, False
        except Exception:
            return False, False

    # 2. Argon2 verification
    if hashed_password.startswith("$argon2"):
        try:
            from argon2 import PasswordHasher
            ph = PasswordHasher()
            ph.verify(hashed_password, plain_password)
            return True, False
        except Exception:
            return False, False

    # 3. PBKDF2 verification
    if hashed_password.startswith("pbkdf2_sha256$"):
        try:
            parts = hashed_password.split("$")
            iterations = int(parts[1])
            salt_bytes = bytes.fromhex(parts[2])
            expected_hash = parts[3]
            computed = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt_bytes, iterations).hex()
            return computed == expected_hash, False
        except Exception:
            return False, False

    # 4. Legacy SHA-256 fallback (Salt: neuroscan_ai_secure_salt_2026)
    legacy_salt = "neuroscan_ai_secure_salt_2026"
    legacy_hash = hashlib.sha256((plain_password + legacy_salt).encode("utf-8")).hexdigest()
    if legacy_hash == hashed_password:
        # Valid legacy hash; flag for automatic upgrade/rehash to bcrypt
        return True, True

    # 5. Direct plaintext check (in case unhashed legacy strings exist)
    if plain_password == hashed_password:
        return True, True

    return False, False


def init_db():
    """Initialize database tables on MySQL (or SQLite fallback) and seed default accounts."""
    global USE_MYSQL

    # Step 1: Attempt MySQL DB Creation
    if PYMYSQL_AVAILABLE:
        try:
            root_conn = pymysql.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                port=MYSQL_PORT,
                charset='utf8mb4',
                autocommit=True
            )
            with root_conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
            root_conn.close()
            USE_MYSQL = True
        except Exception as e:
            print(f"[MySQL Init Notice]: Unable to connect to MySQL ({e}). Using SQLite fallback.")
            USE_MYSQL = False

    conn, engine = get_db_connection()

    if engine == "mysql":
        with conn.cursor() as cursor:
            # 1. Users Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(50) NOT NULL,
                full_name VARCHAR(255),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # 2. Patients Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INT AUTO_INCREMENT PRIMARY KEY,
                full_name VARCHAR(255) NOT NULL,
                age INT NOT NULL,
                gender VARCHAR(50) NOT NULL,
                mrn VARCHAR(100) UNIQUE,
                user_id INT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # 3. Scans & Diagnostics History Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id INT,
                doctor_username VARCHAR(100),
                filename VARCHAR(255) NOT NULL,
                is_valid_mri BOOLEAN NOT NULL,
                guardrail_reason TEXT,
                predicted_class VARCHAR(50),
                confidence FLOAT,
                probabilities_json TEXT,
                tumor_area_cm2 FLOAT,
                tumor_area_mm2 FLOAT,
                tumor_pixel_count INT,
                coverage_pct FLOAT,
                circularity FLOAT,
                compactness FLOAT,
                solidity FLOAT,
                shape_label VARCHAR(50),
                s3_key VARCHAR(255),
                s3_url TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE SET NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # 4. Error Logs Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                error_type VARCHAR(100) NOT NULL,
                error_severity VARCHAR(50) NOT NULL,
                error_message TEXT NOT NULL,
                stack_trace TEXT,
                component VARCHAR(100) NOT NULL,
                username VARCHAR(100),
                filename VARCHAR(255),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # 5. Diagnostic Reports Table (with LONGBLOB for high-res PDF storage)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INT AUTO_INCREMENT PRIMARY KEY,
                report_code VARCHAR(100) UNIQUE NOT NULL,
                scan_id INT,
                patient_id INT,
                patient_name VARCHAR(255) NOT NULL,
                patient_age INT,
                patient_gender VARCHAR(50),
                doctor_username VARCHAR(100),
                predicted_class VARCHAR(50) NOT NULL,
                confidence FLOAT NOT NULL,
                tumor_area_cm2 FLOAT,
                pdf_blob LONGBLOB,
                pdf_filename VARCHAR(255),
                s3_key VARCHAR(255),
                s3_url TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (scan_id) REFERENCES scans (id) ON DELETE SET NULL,
                FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE SET NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Safe auto-migration for existing databases
            try:
                cursor.execute("ALTER TABLE reports ADD COLUMN s3_key VARCHAR(255);")
                cursor.execute("ALTER TABLE reports ADD COLUMN s3_url TEXT;")
            except Exception:
                pass

            try:
                cursor.execute("ALTER TABLE scans ADD COLUMN s3_key VARCHAR(255);")
                cursor.execute("ALTER TABLE scans ADD COLUMN s3_url TEXT;")
            except Exception:
                pass



            # 6. Activity & Audit Logs Table (tracks logins, scans, downloads)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) NOT NULL,
                action VARCHAR(100) NOT NULL,
                role VARCHAR(50),
                details TEXT,
                status VARCHAR(50) DEFAULT 'SUCCESS',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)



            # Seed Default Accounts in MySQL if empty
            cursor.execute("SELECT COUNT(*) as cnt FROM users")
            row = cursor.fetchone()
            if row['cnt'] == 0:
                default_users = [
                    ("doctor",   "brain123",  "doctor",   "Dr. Marcus Vance, M.D."),
                    ("admin",    "neuro2025", "admin",    "Chief AI Systems Administrator"),
                    ("patient",  "patient123","patient",  "Eleanor Campbell"),
                    ("demo",     "demo",      "doctor",   "Demo Clinical Radiologist"),
                ]
                for uname, pwd, r, fname in default_users:
                    cursor.execute("""
                    INSERT INTO users (username, password_hash, role, full_name)
                    VALUES (%s, %s, %s, %s)
                    """, (uname, hash_password(pwd), r, fname))
        conn.close()

    else:
        # SQLite Engine Initialization
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            mrn TEXT UNIQUE,
            user_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            doctor_username TEXT,
            filename TEXT NOT NULL,
            is_valid_mri BOOLEAN NOT NULL,
            guardrail_reason TEXT,
            predicted_class TEXT,
            confidence REAL,
            probabilities_json TEXT,
            tumor_area_cm2 REAL,
            tumor_area_mm2 REAL,
            tumor_pixel_count INTEGER,
            coverage_pct REAL,
            circularity REAL,
            compactness REAL,
            solidity REAL,
            shape_label TEXT,
            s3_key TEXT,
            s3_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS error_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_type TEXT NOT NULL,
            error_severity TEXT NOT NULL,
            error_message TEXT NOT NULL,
            stack_trace TEXT,
            component TEXT NOT NULL,
            username TEXT,
            filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_code TEXT UNIQUE NOT NULL,
            scan_id INTEGER,
            patient_id INTEGER,
            patient_name TEXT NOT NULL,
            patient_age INTEGER,
            patient_gender TEXT,
            doctor_username TEXT,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            tumor_area_cm2 REAL,
            pdf_blob BLOB,
            pdf_filename TEXT,
            s3_key TEXT,
            s3_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        try:
            cursor.execute("ALTER TABLE reports ADD COLUMN s3_key TEXT;")
            cursor.execute("ALTER TABLE reports ADD COLUMN s3_url TEXT;")
            conn.commit()
        except Exception:
            pass

        try:
            cursor.execute("ALTER TABLE scans ADD COLUMN s3_key TEXT;")
            cursor.execute("ALTER TABLE scans ADD COLUMN s3_url TEXT;")
            conn.commit()
        except Exception:
            pass

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (


            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            role TEXT,
            details TEXT,
            status TEXT DEFAULT 'SUCCESS',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()


        cursor.execute("SELECT COUNT(*) as cnt FROM users")
        if cursor.fetchone()['cnt'] == 0:
            default_users = [
                ("doctor",   "brain123",  "doctor",   "Dr. Marcus Vance, M.D."),
                ("admin",    "neuro2025", "admin",    "Chief AI Systems Administrator"),
                ("patient",  "patient123","patient",  "Eleanor Campbell"),
                ("demo",     "demo",      "doctor",   "Demo Clinical Radiologist"),
            ]
            for uname, pwd, r, fname in default_users:
                cursor.execute("""
                INSERT INTO users (username, password_hash, role, full_name)
                VALUES (?, ?, ?, ?)
                """, (uname, hash_password(pwd), r, fname))
            conn.commit()
        conn.close()


def get_active_engine_name():
    """Return 'MySQL (neuroscan_db)' or 'SQLite'."""
    conn, engine = get_db_connection()
    conn.close()
    return "MySQL (neuroscan_db)" if engine == "mysql" else "SQLite (neuroscan.db)"


# ── User & Authentication Operations ─────────────────────────────────────────

def authenticate_user(username: str, password: str):
    """
    Authenticate user against MySQL / SQLite using secure bcrypt / Argon2 verification.
    Transparently upgrades legacy SHA-256 hashes to bcrypt upon successful authentication.
    """
    clean_u = username.strip() if username else ""
    if not clean_u or not password:
        return None

    conn, engine = get_db_connection()
    sql = "SELECT id, username, password_hash, role, full_name, created_at FROM users WHERE username = %s" if engine == "mysql" else \
          "SELECT id, username, password_hash, role, full_name, created_at FROM users WHERE username = ?"
    
    user_row = None
    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sql, (clean_u,))
            user_row = cur.fetchone()
    else:
        cur = conn.cursor()
        cur.execute(sql, (clean_u,))
        row = cur.fetchone()
        user_row = dict(row) if row else None

    if not user_row:
        conn.close()
        return None

    stored_hash = user_row.get("password_hash", "")
    is_valid, needs_rehash = verify_password(password, stored_hash)

    if is_valid:
        # Transparently upgrade legacy SHA-256 or unhashed records to secure bcrypt in database
        if needs_rehash:
            try:
                new_bcrypt_hash = hash_password(password)
                update_sql = "UPDATE users SET password_hash = %s WHERE id = %s" if engine == "mysql" else \
                             "UPDATE users SET password_hash = ? WHERE id = ?"
                if engine == "mysql":
                    with conn.cursor() as cur:
                        cur.execute(update_sql, (new_bcrypt_hash, user_row["id"]))
                else:
                    cur = conn.cursor()
                    cur.execute(update_sql, (new_bcrypt_hash, user_row["id"]))
                    conn.commit()
            except Exception:
                pass
        conn.close()
        # Pop hash from memory for security
        user_row.pop("password_hash", None)
        return user_row

    conn.close()
    return None


def create_user(username: str, password: str, role: str, full_name: str = "", enforce_policy: bool = False):
    """
    Register a new user account in database with salted bcrypt hashing.
    Optionally enforces clinical password complexity policy.
    """
    clean_u = username.strip() if username else ""
    if not clean_u or not password:
        return None, "Username and password cannot be empty."

    if enforce_policy:
        is_valid, issues = validate_password_policy(password, username=clean_u, full_name=full_name)
        if not is_valid:
            return None, " ".join(issues)

    conn, engine = get_db_connection()
    sql = "INSERT INTO users (username, password_hash, role, full_name) VALUES (%s, %s, %s, %s)" if engine == "mysql" else \
          "INSERT INTO users (username, password_hash, role, full_name) VALUES (?, ?, ?, ?)"
    params = (clean_u, hash_password(password), role, full_name.strip() or clean_u.title())

    try:
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute(sql, params)
                user_id = cur.lastrowid
            conn.close()
            return user_id, None
        else:
            cur = conn.cursor()
            cur.execute(sql, params)
            conn.commit()
            user_id = cur.lastrowid
            conn.close()
            return user_id, None
    except Exception as e:
        conn.close()
        err_msg = str(e)
        if "Duplicate entry" in err_msg or "UNIQUE constraint failed" in err_msg:
            return None, "Username already exists."
        return None, err_msg


def get_all_users():
    """Retrieve all registered users."""
    conn, engine = get_db_connection()
    sql = "SELECT id, username, role, full_name, created_at FROM users ORDER BY id ASC"
    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        conn.close()
        return rows
    else:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]


def update_user_role(username: str, new_role: str):
    """Update a user's role in MySQL/SQLite."""
    conn, engine = get_db_connection()
    sql = "UPDATE users SET role = %s WHERE username = %s" if engine == "mysql" else \
          "UPDATE users SET role = ? WHERE username = ?"
    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sql, (new_role, username))
        conn.close()
    else:
        cur = conn.cursor()
        cur.execute(sql, (new_role, username))
        conn.commit()
        conn.close()


def delete_user(username: str):
    """Delete a user account."""
    conn, engine = get_db_connection()
    sql = "DELETE FROM users WHERE username = %s" if engine == "mysql" else \
          "DELETE FROM users WHERE username = ?"
    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sql, (username,))
        conn.close()
    else:
        cur = conn.cursor()
        cur.execute(sql, (username,))
        conn.commit()
        conn.close()


# ── Patient Operations ───────────────────────────────────────────────────────

def create_or_get_patient(full_name: str, age: int, gender: str, user_id: int = None):
    """Find existing patient by name & age or create a new record."""
    if not full_name or not full_name.strip():
        full_name = "Anonymous Patient"

    conn, engine = get_db_connection()
    sel_sql = "SELECT id, full_name, age, gender, mrn FROM patients WHERE full_name = %s AND age = %s AND gender = %s" if engine == "mysql" else \
              "SELECT id, full_name, age, gender, mrn FROM patients WHERE full_name = ? AND age = ? AND gender = ?"
    params = (full_name.strip(), age, gender)

    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sel_sql, params)
            row = cur.fetchone()
            if row:
                conn.close()
                return row['id']

            mrn = f"MRN-{datetime.datetime.now().strftime('%Y%m')}-{int(datetime.datetime.now().timestamp()) % 10000:04d}"
            ins_sql = "INSERT INTO patients (full_name, age, gender, mrn, user_id) VALUES (%s, %s, %s, %s, %s)"
            cur.execute(ins_sql, (full_name.strip(), age, gender, mrn, user_id))
            patient_id = cur.lastrowid
        conn.close()
        return patient_id
    else:
        cur = conn.cursor()
        cur.execute(sel_sql, params)
        row = cur.fetchone()
        if row:
            conn.close()
            return row['id']

        mrn = f"MRN-{datetime.datetime.now().strftime('%Y%m')}-{int(datetime.datetime.now().timestamp()) % 10000:04d}"
        ins_sql = "INSERT INTO patients (full_name, age, gender, mrn, user_id) VALUES (?, ?, ?, ?, ?)"
        cur.execute(ins_sql, (full_name.strip(), age, gender, mrn, user_id))
        conn.commit()
        patient_id = cur.lastrowid
        conn.close()
        return patient_id


def get_all_patients():
    """Retrieve all patient records."""
    conn, engine = get_db_connection()
    sql = "SELECT id, full_name, age, gender, mrn, created_at FROM patients ORDER BY id DESC"
    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        conn.close()
        return rows
    else:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]


def get_patient_by_user_id(user_id: int):
    """Retrieve patient record associated with a specific user_id."""
    if not user_id:
        return None
    conn, engine = get_db_connection()
    sql = "SELECT id, full_name, age, gender, mrn, created_at FROM patients WHERE user_id = %s" if engine == "mysql" else \
          "SELECT id, full_name, age, gender, mrn, created_at FROM patients WHERE user_id = ?"
    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            row = cur.fetchone()
        conn.close()
        return row
    else:
        cur = conn.cursor()
        cur.execute(sql, (user_id,))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None


def create_patient_account(username: str, password: str, full_name: str, age: int, gender: str, doctor_username: str = ""):
    """Create a new Patient account and demographic profile with password policy validation."""
    clean_u = username.strip() if username else ""
    if not clean_u or not password or not full_name or not full_name.strip():
        return None, "All fields (Username, Password, Full Name, Age, Gender) are required."

    is_valid, issues = validate_password_policy(password, username=clean_u, full_name=full_name.strip())
    if not is_valid:
        return None, " ".join(issues)

    user_id, err = create_user(clean_u, password, "patient", full_name.strip(), enforce_policy=False)
    if err:
        return None, err

    # Create patient record linked to user_id
    patient_id = create_or_get_patient(full_name.strip(), age, gender, user_id=user_id)
    
    # Log audit event
    action_type = "PATIENT_ONBOARDED" if doctor_username else "PATIENT_REGISTERED"
    actor_role = "doctor" if doctor_username else "patient"
    actor_user = doctor_username if doctor_username else clean_u
    log_detail = f"Doctor onboarded patient '{full_name.strip()}' (Username: @{clean_u})" if doctor_username else \
                 f"Patient self-registered account '{full_name.strip()}' (@{clean_u}, Age: {age}, Gender: {gender})"

    log_activity(
        username=actor_user,
        action=action_type,
        role=actor_role,
        details=log_detail,
        status="SUCCESS"
    )
    return patient_id, None



# ── Scan & Diagnostic History Operations ─────────────────────────────────────

def save_scan_record(
    filename: str,
    is_valid_mri: bool,
    guardrail_reason: str = "",
    patient_id: int = None,
    doctor_username: str = "",
    predicted_class: str = None,
    confidence: float = None,
    probabilities_dict: dict = None,
    area_data: dict = None,
    shape_data: dict = None,
    s3_key: str = None,
    s3_url: str = None,
):
    """Save an MRI scan diagnostic inference record to database."""
    conn, engine = get_db_connection()

    probs_json = json.dumps(probabilities_dict) if probabilities_dict else None
    area_cm2 = area_data.get('area_cm2') if area_data else None
    area_mm2 = area_data.get('area_mm2') if area_data else None
    px_count = area_data.get('pixel_count') if area_data else None
    cov_pct = area_data.get('coverage_pct') if area_data else None

    circ = shape_data.get('circularity') if shape_data else None
    comp = shape_data.get('compactness') if shape_data else None
    sol = shape_data.get('solidity') if shape_data else None
    shape_lbl = shape_data.get('shape_label') if shape_data else None

    sql = """
    INSERT INTO scans (
        patient_id, doctor_username, filename, is_valid_mri, guardrail_reason,
        predicted_class, confidence, probabilities_json,
        tumor_area_cm2, tumor_area_mm2, tumor_pixel_count, coverage_pct,
        circularity, compactness, solidity, shape_label, s3_key, s3_url
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """ if engine == "mysql" else """
    INSERT INTO scans (
        patient_id, doctor_username, filename, is_valid_mri, guardrail_reason,
        predicted_class, confidence, probabilities_json,
        tumor_area_cm2, tumor_area_mm2, tumor_pixel_count, coverage_pct,
        circularity, compactness, solidity, shape_label, s3_key, s3_url
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = (
        patient_id, doctor_username, filename, is_valid_mri, guardrail_reason,
        predicted_class, confidence, probs_json,
        area_cm2, area_mm2, px_count, cov_pct,
        circ, comp, sol, shape_lbl, s3_key, s3_url
    )

    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sql, params)
            scan_id = cur.lastrowid
        conn.close()
        return scan_id
    else:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        scan_id = cur.lastrowid
        conn.close()
        return scan_id


def get_latest_scan_for_patient(patient_id: int = None, patient_name: str = None):
    """Retrieve the most recent scan record for a specific patient."""
    conn, engine = get_db_connection()
    if not patient_id and not patient_name:
        return None

    if patient_id:
        sql = """
        SELECT s.*, p.full_name as patient_name, p.age as patient_age, p.gender as patient_gender, p.mrn
        FROM scans s
        LEFT JOIN patients p ON s.patient_id = p.id
        WHERE s.patient_id = %s
        ORDER BY s.id DESC
        LIMIT 1
        """ if engine == "mysql" else """
        SELECT s.*, p.full_name as patient_name, p.age as patient_age, p.gender as patient_gender, p.mrn
        FROM scans s
        LEFT JOIN patients p ON s.patient_id = p.id
        WHERE s.patient_id = ?
        ORDER BY s.id DESC
        LIMIT 1
        """
        params = (patient_id,)
    else:
        sql = """
        SELECT s.*, p.full_name as patient_name, p.age as patient_age, p.gender as patient_gender, p.mrn
        FROM scans s
        LEFT JOIN patients p ON s.patient_id = p.id
        WHERE p.full_name = %s
        ORDER BY s.id DESC
        LIMIT 1
        """ if engine == "mysql" else """
        SELECT s.*, p.full_name as patient_name, p.age as patient_age, p.gender as patient_gender, p.mrn
        FROM scans s
        LEFT JOIN patients p ON s.patient_id = p.id
        WHERE p.full_name = ?
        ORDER BY s.id DESC
        LIMIT 1
        """
        params = (patient_name,)

    try:
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
            conn.close()
            return row
        else:
            cur = conn.cursor()
            cur.execute(sql, params)
            row = cur.fetchone()
            conn.close()
            return dict(row) if row else None
    except Exception as e:
        print(f"[GET LATEST SCAN ERROR]: {e}")
        conn.close()
        return None


def get_all_scans_for_patient(patient_id: int = None, patient_name: str = None, limit: int = 20):
    """Retrieve all scan records for a specific patient."""
    conn, engine = get_db_connection()
    if not patient_id and not patient_name:
        return []

    if patient_id:
        sql = """
        SELECT s.*, p.full_name as patient_name, p.age as patient_age, p.gender as patient_gender, p.mrn
        FROM scans s
        LEFT JOIN patients p ON s.patient_id = p.id
        WHERE s.patient_id = %s
        ORDER BY s.id DESC
        LIMIT %s
        """ if engine == "mysql" else """
        SELECT s.*, p.full_name as patient_name, p.age as patient_age, p.gender as patient_gender, p.mrn
        FROM scans s
        LEFT JOIN patients p ON s.patient_id = p.id
        WHERE s.patient_id = ?
        ORDER BY s.id DESC
        LIMIT ?
        """
        params = (patient_id, limit)
    else:
        sql = """
        SELECT s.*, p.full_name as patient_name, p.age as patient_age, p.gender as patient_gender, p.mrn
        FROM scans s
        LEFT JOIN patients p ON s.patient_id = p.id
        WHERE p.full_name = %s
        ORDER BY s.id DESC
        LIMIT %s
        """ if engine == "mysql" else """
        SELECT s.*, p.full_name as patient_name, p.age as patient_age, p.gender as patient_gender, p.mrn
        FROM scans s
        LEFT JOIN patients p ON s.patient_id = p.id
        WHERE p.full_name = ?
        ORDER BY s.id DESC
        LIMIT ?
        """
        params = (patient_name, limit)

    try:
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            conn.close()
            return rows
        else:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            conn.close()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"[GET ALL SCANS ERROR]: {e}")
        conn.close()
        return []


def get_recent_scans(limit: int = 25):
    """Retrieve recent MRI diagnostic scan records."""
    conn, engine = get_db_connection()
    sql = """
    SELECT s.*, p.full_name as patient_name, p.age as patient_age, p.gender as patient_gender
    FROM scans s
    LEFT JOIN patients p ON s.patient_id = p.id
    ORDER BY s.id DESC
    LIMIT %s
    """ if engine == "mysql" else """
    SELECT s.*, p.full_name as patient_name, p.age as patient_age, p.gender as patient_gender
    FROM scans s
    LEFT JOIN patients p ON s.patient_id = p.id
    ORDER BY s.id DESC
    LIMIT ?
    """

    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
        conn.close()
        return rows
    else:
        cur = conn.cursor()
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]


# ── Report Management Operations ─────────────────────────────────────────────

def save_report(
    report_code: str,
    patient_name: str,
    predicted_class: str,
    confidence: float,
    pdf_bytes: bytes,
    pdf_filename: str,
    scan_id: int = None,
    patient_id: int = None,
    patient_age: int = None,
    patient_gender: str = None,
    doctor_username: str = "doctor",
    tumor_area_cm2: float = None,
    s3_key: str = None,
    s3_url: str = None,
):
    """Store a generated PDF clinical diagnostic report, its binary blob, and AWS S3 cloud URL into database."""
    conn, engine = get_db_connection()
    ins_sql = """
    INSERT INTO reports (
        report_code, scan_id, patient_id, patient_name, patient_age, patient_gender,
        doctor_username, predicted_class, confidence, tumor_area_cm2,
        pdf_blob, pdf_filename, s3_key, s3_url
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """ if engine == "mysql" else """
    INSERT INTO reports (
        report_code, scan_id, patient_id, patient_name, patient_age, patient_gender,
        doctor_username, predicted_class, confidence, tumor_area_cm2,
        pdf_blob, pdf_filename, s3_key, s3_url
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = (
        report_code, scan_id, patient_id, patient_name, patient_age, patient_gender,
        doctor_username, predicted_class, confidence, tumor_area_cm2,
        pdf_bytes, pdf_filename, s3_key, s3_url
    )

    try:
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute(ins_sql, params)
                rep_id = cur.lastrowid
            conn.close()
            return rep_id
        else:
            cur = conn.cursor()
            cur.execute(ins_sql, params)
            conn.commit()
            rep_id = cur.lastrowid
            conn.close()
            return rep_id
    except Exception as e:
        conn.close()
        print(f"[REPORT SAVE FAILED]: {e}")
        return None


def get_all_reports(limit: int = 50):
    """Retrieve all generated diagnostic reports including AWS S3 cloud links."""
    conn, engine = get_db_connection()
    sql = """
    SELECT id, report_code, scan_id, patient_id, patient_name, patient_age, patient_gender,
           doctor_username, predicted_class, confidence, tumor_area_cm2, pdf_filename, s3_key, s3_url, created_at
    FROM reports
    ORDER BY id DESC
    LIMIT %s
    """ if engine == "mysql" else """
    SELECT id, report_code, scan_id, patient_id, patient_name, patient_age, patient_gender,
           doctor_username, predicted_class, confidence, tumor_area_cm2, pdf_filename, s3_key, s3_url, created_at
    FROM reports
    ORDER BY id DESC
    LIMIT ?
    """

    if engine == "mysql":

        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
        conn.close()
        return rows
    else:
        cur = conn.cursor()
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]



def get_report_pdf_blob(report_id: int):
    """Retrieve raw PDF binary blob for download."""
    conn, engine = get_db_connection()
    sql = "SELECT pdf_blob, pdf_filename, report_code FROM reports WHERE id = %s" if engine == "mysql" else \
          "SELECT pdf_blob, pdf_filename, report_code FROM reports WHERE id = ?"

    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sql, (report_id,))
            row = cur.fetchone()
        conn.close()
        if row:
            return row["pdf_blob"], row["pdf_filename"]
        return None, None
    else:
        cur = conn.cursor()
        cur.execute(sql, (report_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            return row["pdf_blob"], row["pdf_filename"]
        return None, None


# ── Error & System Logging Operations ────────────────────────────────────────

def log_error(
    error_type: str,
    severity: str,
    message: str,
    stack_trace: str = None,
    component: str = "general",
    username: str = None,
    filename: str = None,
):
    """Log an application, pipeline, guardrail, or runtime exception into database."""
    try:
        conn, engine = get_db_connection()
        sql = """
        INSERT INTO error_logs (
            error_type, error_severity, error_message, stack_trace, component, username, filename
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """ if engine == "mysql" else """
        INSERT INTO error_logs (
            error_type, error_severity, error_message, stack_trace, component, username, filename
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (error_type, severity.upper(), message, stack_trace, component, username, filename)

        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.close()
        else:
            cur = conn.cursor()
            cur.execute(sql, params)
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"[ERROR LOGGER FAILED]: {e}")


def get_error_logs(limit: int = 50):
    """Retrieve recent system error logs."""
    conn, engine = get_db_connection()
    sql = """
    SELECT id, error_type, error_severity, error_message, stack_trace, component, username, filename, created_at
    FROM error_logs
    ORDER BY id DESC
    LIMIT %s
    """ if engine == "mysql" else """
    SELECT id, error_type, error_severity, error_message, stack_trace, component, username, filename, created_at
    FROM error_logs
    ORDER BY id DESC
    LIMIT ?
    """

    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
        conn.close()
        return rows
    else:
        cur = conn.cursor()
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]


def clear_error_logs():
    """Clear all system error logs."""
    conn, engine = get_db_connection()
    sql = "DELETE FROM error_logs"
    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.close()
    else:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        conn.close()


# ── Activity & Audit Log Operations ──────────────────────────────────────────

def log_activity(
    username: str,
    action: str,
    role: str = "",
    details: str = "",
    status: str = "SUCCESS"
):
    """Record an audit trail event (Login, Logout, Scan run, PDF download, Role update)."""
    try:
        conn, engine = get_db_connection()
        sql = """
        INSERT INTO activity_logs (username, action, role, details, status)
        VALUES (%s, %s, %s, %s, %s)
        """ if engine == "mysql" else """
        INSERT INTO activity_logs (username, action, role, details, status)
        VALUES (?, ?, ?, ?, ?)
        """
        params = (username, action, role, details, status)

        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.close()
        else:
            cur = conn.cursor()
            cur.execute(sql, params)
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"[ACTIVITY LOGGER FAILED]: {e}")


def get_activity_logs(limit: int = 100):
    """Retrieve recent user activity and login audit logs."""
    conn, engine = get_db_connection()
    sql = """
    SELECT id, username, action, role, details, status, created_at
    FROM activity_logs
    ORDER BY id DESC
    LIMIT %s
    """ if engine == "mysql" else """
    SELECT id, username, action, role, details, status, created_at
    FROM activity_logs
    ORDER BY id DESC
    LIMIT ?
    """

    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
        conn.close()
        return rows
    else:
        cur = conn.cursor()
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]


def clear_activity_logs():
    """Clear user activity logs."""
    conn, engine = get_db_connection()
    sql = "DELETE FROM activity_logs"
    if engine == "mysql":
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.close()
    else:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        conn.close()

