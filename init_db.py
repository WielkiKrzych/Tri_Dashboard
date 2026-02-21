#!/usr/bin/env python3
"""
Database Initialization Script.

Usage:
    python init_db.py           # Initialize database if missing
    python init_db.py --reset   # Delete existing database and recreate
"""
import sys
import argparse
import logging
from modules.config import Config
from modules.db.session_store import SessionStore

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def init_db(reset: bool = False):
    db_path = Config.DB_PATH
    logger.info("Target Database: %s", db_path)

    if db_path.exists():
        if reset:
            logger.warning("--reset flag provided. Deleting existing database...")
            db_path.unlink()
            logger.info("Database file deleted.")
        else:
            logger.info("Database already exists. Run with --reset to recreate.")
            return

    logger.info("Initializing SessionStore...")
    store = SessionStore()
    
    if db_path.exists():
        tables = store.get_session_count()
        logger.info("Database initialized successfully at %s.", db_path)
        logger.info("  Current session count: %d", tables)
    else:
        logger.error("Failed to create database file.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Tri_Dashboard Database")
    parser.add_argument("--reset", action="store_true", help="Delete existing database and recreate")
    args = parser.parse_args()
    
    init_db(args.reset)
