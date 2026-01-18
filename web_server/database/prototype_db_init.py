from sqlalchemy import create_engine, Column, String, TIMESTAMP, Integer, CheckConstraint, ForeignKey, Float
import os
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Documents(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    file_name = Column(String(255), nullable=False)
    upload_time = Column(TIMESTAMP, nullable=False)
    size_kb = Column(Float)
    mime_type = Column(String)
    source = Column(String)
    file_path = Column(String)


class Ocr_jobs(Base):
    __tablename__ = 'ocr_jobs'
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    started_time = Column(TIMESTAMP, nullable=False)
    completed_time = Column(TIMESTAMP)
    status = Column(String, nullable=False)

    __table_args__ = CheckConstraint(status.in_(["queued", "processing", "succeeded", "failed"]),
                        name="ck_ocr_jobs_status")


class Ocr_results(Base):
    __tablename__ = 'ocr_results'
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    client_name = Column(String, nullable=False)


class Logs(Base):
    __tablename__ = 'logs'
    id = Column(Integer, primary_key=True)
    time_created = Column(TIMESTAMP)
    related_table = Column(String)
    log_information = Column(String)

    __table_args__ = CheckConstraint(related_table.in_(["documents", "ocr_jobs", "ocr_results", "users", "log",
                                                        "multiple", "none"]),
                                     name="ck_ocr_jobs_status")


def setup_database():
    URL = os.getenv("DATABASE_URL")
    print(URL)
    engine = create_engine(URL, echo=True)
    Base.metadata.create_all(engine)
