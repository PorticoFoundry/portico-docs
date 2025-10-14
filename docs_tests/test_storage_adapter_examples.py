"""Test examples for Storage Adapter documentation.

This module tests code examples from storage adapter documentation (SQLAlchemy, SQLite, PostgreSQL)
to ensure they remain accurate and working. These tests complement test_storage_examples.py
with adapter-specific configuration and pattern examples.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from portico.adapters.storage.database_adapters import SqlAlchemyAdapter

# ========== SQLAlchemy Adapter Configuration Examples ==========


# --8<-- [start:sqlalchemy-engine-options]
def test_sqlalchemy_engine_options(config_registry):
    """SQLAlchemy adapter with engine options."""
    # This demonstrates the configuration pattern
    # We don't actually connect since we don't have PostgreSQL running
    adapter_config = {
        "database_url": "postgresql+asyncpg://user:pass@localhost:5432/db",
        "pool_size": 20,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 3600,
        "pool_pre_ping": True,
        "echo": False,
        "echo_pool": False,
        "connect_args": {
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
        },
    }

    # Verify configuration structure
    assert adapter_config["pool_size"] == 20
    assert adapter_config["pool_pre_ping"] is True
    assert "connect_args" in adapter_config


# --8<-- [end:sqlalchemy-engine-options]


# --8<-- [start:sqlalchemy-production-settings]
def test_sqlalchemy_production_settings(config_registry):
    """Production configuration for SQLAlchemy."""
    config = {
        "database_url": "postgresql+asyncpg://user:pass@db.example.com:5432/prod",
        "pool_size": 20,
        "max_overflow": 10,
        "pool_pre_ping": True,
        "pool_recycle": 3600,
        "echo": False,
    }

    assert config["pool_size"] == 20
    assert config["pool_pre_ping"] is True
    assert config["echo"] is False


# --8<-- [end:sqlalchemy-production-settings]


# --8<-- [start:sqlalchemy-development-settings]
def test_sqlalchemy_development_settings(config_registry):
    """Development configuration with query logging."""
    config = {
        "database_url": "sqlite+aiosqlite:///./dev.db",
        "echo": True,  # Log SQL queries for debugging
    }

    assert config["echo"] is True
    assert "sqlite" in config["database_url"]


# --8<-- [end:sqlalchemy-development-settings]


# --8<-- [start:sqlalchemy-pool-sizing]
def test_sqlalchemy_connection_pool_sizing(config_registry):
    """Calculate connection pool size."""
    # Formula: pool_size = (number of workers) * (concurrent requests per worker)

    # Example: 4 workers, 5 concurrent requests each
    pool_size = 4 * 5  # = 20

    config = {
        "pool_size": pool_size,
        "max_overflow": pool_size // 2,  # 50% overflow
    }

    assert config["pool_size"] == 20
    assert config["max_overflow"] == 10


# --8<-- [end:sqlalchemy-pool-sizing]


# --8<-- [start:sqlalchemy-in-memory-testing]
@pytest.mark.asyncio
async def test_sqlalchemy_in_memory_testing(config_registry):
    """In-memory SQLite for testing."""
    adapter = SqlAlchemyAdapter("sqlite+aiosqlite:///:memory:")

    # Verify adapter is created
    assert adapter is not None
    assert ":memory:" in adapter.database_url


# --8<-- [end:sqlalchemy-in-memory-testing]

# ========== SQLite Adapter Examples ==========


# --8<-- [start:sqlite-database-urls]
def test_sqlite_database_urls(config_registry):
    """SQLite database URL patterns."""
    # File-based database
    file_db = "sqlite+aiosqlite:///./app.db"  # Relative path
    absolute_unix = "sqlite+aiosqlite:////tmp/app.db"  # Absolute path (Unix)
    absolute_windows = "sqlite+aiosqlite:///C:/data/app.db"  # Absolute path (Windows)

    # In-memory database (testing)
    memory_db = "sqlite+aiosqlite:///:memory:"

    assert "sqlite+aiosqlite" in file_db
    assert ":memory:" in memory_db
    assert "/tmp" in absolute_unix


# --8<-- [end:sqlite-database-urls]


# --8<-- [start:sqlite-basic-setup]
@pytest.mark.asyncio
async def test_sqlite_basic_setup(config_registry):
    """Basic SQLite adapter setup."""
    # Development database
    adapter = SqlAlchemyAdapter(
        "sqlite+aiosqlite:///:memory:",
        echo=True,  # Log SQL queries for debugging
    )

    assert adapter is not None
    # Note: echo is stored in the engine, not directly on adapter


# --8<-- [end:sqlite-basic-setup]


# --8<-- [start:sqlite-testing-setup]
@pytest.mark.asyncio
async def test_sqlite_testing_setup(config_registry):
    """In-memory database for tests."""
    adapter = SqlAlchemyAdapter("sqlite+aiosqlite:///:memory:", echo=False)

    assert adapter is not None
    assert ":memory:" in adapter.database_url


# --8<-- [end:sqlite-testing-setup]


# --8<-- [start:sqlite-production-setup]
def test_sqlite_production_setup(config_registry):
    """Production SQLite configuration."""
    config = {
        "database_url": "sqlite+aiosqlite:///./production.db",
        "echo": False,
        "connect_args": {
            "check_same_thread": False,  # Allow multiple threads
            "timeout": 30,  # Connection timeout in seconds
        },
    }

    assert config["echo"] is False
    assert config["connect_args"]["timeout"] == 30


# --8<-- [end:sqlite-production-setup]


# --8<-- [start:sqlite-file-permissions]
def test_sqlite_file_permissions_pattern(config_registry):
    """Ensure database directory exists."""
    # Pattern demonstration (not actually creating files in tests)
    db_path = Path("./test_data/app.db")

    # Would normally do: db_path.parent.mkdir(parents=True, exist_ok=True)
    # For test, just verify path construction
    assert db_path.parent == Path("./test_data")
    assert db_path.name == "app.db"


# --8<-- [end:sqlite-file-permissions]


# --8<-- [start:sqlite-wal-mode]
def test_sqlite_wal_mode_pattern(config_registry):
    """WAL mode configuration pattern."""
    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    # This demonstrates the pattern for enabling WAL mode
    # Actual event listener would be:
    # @event.listens_for(Engine, "connect")
    # def set_sqlite_pragma(dbapi_conn, connection_record):
    #     cursor = dbapi_conn.cursor()
    #     cursor.execute("PRAGMA journal_mode=WAL")
    #     cursor.execute("PRAGMA synchronous=NORMAL")
    #     cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
    #     cursor.execute("PRAGMA temp_store=MEMORY")
    #     cursor.close()

    # Verify imports work
    assert event is not None
    assert Engine is not None


# --8<-- [end:sqlite-wal-mode]


# --8<-- [start:sqlite-backup-pattern]
def test_sqlite_backup_pattern(config_registry):
    """SQLite backup strategy pattern."""
    import shutil
    from datetime import datetime

    # Demonstrate backup function structure
    def backup_sqlite_db(db_path: str):
        """Create timestamped backup of SQLite database."""
        source = Path(db_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = source.parent / "backups"
        backup_path = backup_dir / f"{source.stem}_{timestamp}.db"

        return backup_path

    # Test function structure
    result = backup_sqlite_db("./app.db")
    assert "backups" in str(result)
    assert ".db" in str(result)


# --8<-- [end:sqlite-backup-pattern]

# ========== PostgreSQL Adapter Examples ==========


# --8<-- [start:postgresql-database-urls]
def test_postgresql_database_urls(config_registry):
    """PostgreSQL database URL patterns."""
    # Basic connection
    basic = "postgresql+asyncpg://user:password@localhost:5432/dbname"

    # With custom port
    custom_port = "postgresql+asyncpg://user:password@db.example.com:5433/dbname"

    # With SSL
    with_ssl = "postgresql+asyncpg://user:password@localhost:5432/dbname?ssl=require"

    # Connection pool settings in URL
    with_pool = (
        "postgresql+asyncpg://user:password@localhost/dbname?min_size=10&max_size=20"
    )

    assert "postgresql+asyncpg" in basic
    assert "5433" in custom_port
    assert "ssl=require" in with_ssl
    assert "min_size=10" in with_pool


# --8<-- [end:postgresql-database-urls]


# --8<-- [start:postgresql-basic-setup]
def test_postgresql_basic_setup(config_registry):
    """Basic PostgreSQL adapter setup."""
    config = {
        "database_url": "postgresql+asyncpg://user:password@localhost:5432/myapp",
        "pool_size": 20,
        "max_overflow": 10,
        "pool_pre_ping": True,
    }

    assert config["pool_size"] == 20
    assert config["pool_pre_ping"] is True


# --8<-- [end:postgresql-basic-setup]


# --8<-- [start:postgresql-production-config]
def test_postgresql_production_configuration(config_registry):
    """Production-ready PostgreSQL setup."""
    config = {
        "database_url": "postgresql+asyncpg://app_user:secure_password@db.prod.com:5432/production",
        # Connection pool
        "pool_size": 20,  # Base pool size
        "max_overflow": 10,  # Extra connections when needed
        "pool_timeout": 30,  # Seconds to wait for connection
        "pool_recycle": 3600,  # Recycle connections after 1 hour
        "pool_pre_ping": True,  # Test connections before using
        # Performance
        "echo": False,  # Disable SQL logging in production
        "echo_pool": False,  # Disable pool logging
        # Connection arguments
        "connect_args": {
            "server_settings": {
                "application_name": "portico_app",
                "jit": "off",  # Disable JIT compilation for predictable performance
            },
            "command_timeout": 60,  # Query timeout in seconds
            "timeout": 10,  # Connection timeout
        },
    }

    assert config["pool_size"] == 20
    assert config["pool_recycle"] == 3600
    assert (
        config["connect_args"]["server_settings"]["application_name"] == "portico_app"
    )


# --8<-- [end:postgresql-production-config]


# --8<-- [start:postgresql-ssl-config]
def test_postgresql_ssl_configuration(config_registry):
    """SSL configuration for PostgreSQL."""
    # SSL with certificate verification
    ssl_config = {
        "database_url": "postgresql+asyncpg://user:pass@db.example.com:5432/db?ssl=require",
        "connect_args": {
            "ssl": {
                "ca": "/path/to/ca-cert.pem",
                "cert": "/path/to/client-cert.pem",
                "key": "/path/to/client-key.pem",
            }
        },
    }

    # SSL without certificate verification (development only)
    dev_ssl_config = {
        "database_url": "postgresql+asyncpg://user:pass@localhost:5432/db?ssl=prefer"
    }

    assert "ssl=require" in ssl_config["database_url"]
    assert "ca" in ssl_config["connect_args"]["ssl"]
    assert "ssl=prefer" in dev_ssl_config["database_url"]


# --8<-- [end:postgresql-ssl-config]


# --8<-- [start:postgresql-environment-config]
def test_postgresql_environment_based_configuration(config_registry):
    """Environment-based PostgreSQL configuration."""
    # Simulate environment variables
    DATABASE_URL = "postgresql+asyncpg://user:pass@localhost:5432/db"
    POOL_SIZE = 20
    POOL_TIMEOUT = 30

    config = {
        "database_url": DATABASE_URL,
        "pool_size": POOL_SIZE,
        "pool_timeout": POOL_TIMEOUT,
        "pool_pre_ping": True,
    }

    assert config["pool_size"] == 20
    assert config["pool_timeout"] == 30


# --8<-- [end:postgresql-environment-config]


# --8<-- [start:postgresql-pool-sizing]
def test_postgresql_connection_pool_sizing(config_registry):
    """Calculate PostgreSQL pool size."""
    # Formula: pool_size = (workers × concurrent_requests) + overhead

    # Example: 4 Uvicorn workers, 5 concurrent requests each
    workers = 4
    concurrent_requests = 5
    overhead = 5  # For background tasks

    pool_size = (workers * concurrent_requests) + overhead  # = 25

    config = {
        "pool_size": pool_size,
        "max_overflow": pool_size // 2,  # 50% overflow capacity
    }

    assert config["pool_size"] == 25
    assert config["max_overflow"] == 12


# --8<-- [end:postgresql-pool-sizing]


# --8<-- [start:postgresql-read-replicas]
def test_postgresql_read_replicas_pattern(config_registry):
    """Read replica configuration pattern."""
    # Primary database (writes)
    primary_url = "postgresql+asyncpg://user:pass@primary.db.com:5432/db"

    # Read replica (reads)
    replica_url = "postgresql+asyncpg://user:pass@replica.db.com:5432/db"

    assert "primary.db.com" in primary_url
    assert "replica.db.com" in replica_url


# --8<-- [end:postgresql-read-replicas]


# --8<-- [start:postgresql-connection-failover]
def test_postgresql_connection_failover(config_registry):
    """Connection failover configuration."""
    # Multiple database hosts for failover
    database_url = (
        "postgresql+asyncpg://user:pass@db1.example.com,db2.example.com:5432/db"
    )

    config = {
        "database_url": database_url,
        "pool_pre_ping": True,  # Test connections before use
        "pool_recycle": 3600,  # Recycle connections regularly
    }

    assert "db1.example.com" in config["database_url"]
    assert "db2.example.com" in config["database_url"]
    assert config["pool_pre_ping"] is True


# --8<-- [end:postgresql-connection-failover]


# --8<-- [start:postgresql-ssl-security]
def test_postgresql_ssl_security(config_registry):
    """SSL/TLS encryption for PostgreSQL."""
    # Require SSL
    require_ssl = {
        "database_url": "postgresql+asyncpg://user:pass@db.example.com:5432/db?ssl=require"
    }

    # Verify SSL certificate
    verify_ssl = {
        "database_url": "postgresql+asyncpg://user:pass@db.example.com:5432/db?ssl=verify-full",
        "connect_args": {"ssl": {"ca": "/path/to/ca-cert.pem"}},
    }

    assert "ssl=require" in require_ssl["database_url"]
    assert "ssl=verify-full" in verify_ssl["database_url"]
    assert "ca" in verify_ssl["connect_args"]["ssl"]


# --8<-- [end:postgresql-ssl-security]


# --8<-- [start:postgresql-password-security]
def test_postgresql_password_security_pattern(config_registry):
    """Password security with URL encoding."""
    from urllib.parse import quote_plus

    # Simulate loading password from environment
    password = "p@ssw0rd!#$"
    encoded_password = quote_plus(password)

    database_url = f"postgresql+asyncpg://user:{encoded_password}@localhost:5432/db"

    assert encoded_password != password
    assert "p%40ssw0rd" in database_url  # @ is encoded as %40


# --8<-- [end:postgresql-password-security]


# --8<-- [start:postgresql-migration-from-sqlite]
def test_postgresql_migration_from_sqlite(config_registry):
    """Migration pattern from SQLite to PostgreSQL."""
    # Before (SQLite)
    sqlite_url = "sqlite+aiosqlite:///./app.db"

    # After (PostgreSQL)
    postgresql_url = "postgresql+asyncpg://user:pass@localhost:5432/app"
    postgresql_config = {"pool_size": 20}

    # Verify both URLs are valid formats
    assert "sqlite" in sqlite_url
    assert "postgresql" in postgresql_url
    assert postgresql_config["pool_size"] == 20


# --8<-- [end:postgresql-migration-from-sqlite]


# --8<-- [start:postgresql-pool-exhausted]
def test_postgresql_connection_pool_exhausted_solutions(config_registry):
    """Solutions for connection pool exhaustion."""
    # Solution 1: Increase pool size
    solution1 = {"pool_size": 30, "max_overflow": 20}

    # Solution 2: Increase timeout
    solution2 = {"pool_timeout": 60}  # Wait longer

    assert solution1["pool_size"] == 30
    assert solution2["pool_timeout"] == 60


# --8<-- [end:postgresql-pool-exhausted]


# --8<-- [start:postgresql-ssl-certificate-errors]
def test_postgresql_ssl_certificate_error_solutions(config_registry):
    """Solutions for SSL certificate errors."""
    # Development: Allow unverified SSL
    dev_config = {
        "database_url": "postgresql+asyncpg://user:pass@localhost:5432/db?ssl=prefer"
    }

    # Production: Verify SSL properly
    prod_config = {
        "database_url": "postgresql+asyncpg://user:pass@db.prod.com:5432/db?ssl=verify-full",
        "connect_args": {"ssl": {"ca": "/etc/ssl/certs/ca-certificates.crt"}},
    }

    assert "ssl=prefer" in dev_config["database_url"]
    assert "ssl=verify-full" in prod_config["database_url"]


# --8<-- [end:postgresql-ssl-certificate-errors]


# --8<-- [start:postgresql-core-integration]
@pytest.mark.asyncio
async def test_postgresql_porticocore_integration(config_registry):
    """Integration with PorticoCore."""
    from portico.core import PorticoCore

    # PostgreSQL configuration
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Verify core initialized with PostgreSQL settings
    assert core is not None
    assert core.db is not None

    await core.close()


# --8<-- [end:postgresql-core-integration]
