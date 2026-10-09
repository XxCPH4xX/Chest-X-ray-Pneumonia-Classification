from sqlalchemy import JSON, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[str] = mapped_column(String, index=True)
    label: Mapped[str] = mapped_column(String, index=True)
    payload: Mapped[dict] = mapped_column(JSON)


class History:
    def __init__(self, database_url):
        if not database_url.startswith("sqlite:///"):
            raise ValueError("Only local SQLite history is supported")
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)

    def save(self, result: dict):
        # Do not store images or original filenames.
        payload = {k: v for k, v in result.items() if k != "images"}
        payload["gradcam_available"] = False
        with Session(self.engine) as session:
            session.add(
                PredictionRecord(
                    id=result["prediction_id"],
                    created_at=result["created_at"],
                    label=result["predicted_class"],
                    payload=payload,
                )
            )
            session.commit()

    def list(self, limit: int, offset: int, label: str | None, search: str):
        query = select(PredictionRecord)
        if label:
            query = query.where(PredictionRecord.label == label)
        if search:
            query = query.where(PredictionRecord.id.contains(search, autoescape=True))
        with Session(self.engine) as session:
            count = session.scalar(select(func.count()).select_from(query.subquery()))
            records = session.scalars(
                query.order_by(PredictionRecord.created_at.desc()).offset(offset).limit(limit)
            ).all()
            return {"enabled": True, "total": count, "items": [r.payload for r in records]}

    def get(self, identifier: str):
        with Session(self.engine) as session:
            record = session.get(PredictionRecord, identifier)
            return record.payload if record else None

    def delete(self, identifier: str) -> bool:
        with Session(self.engine) as session:
            record = session.get(PredictionRecord, identifier)
            if record is None:
                return False
            session.delete(record)
            session.commit()
            return True
