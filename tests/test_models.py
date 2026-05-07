"""Тесты ORM-модели Violation на in-memory SQLite."""

from models import Violation


def test_insert_and_read_violation(db_session):
    v = Violation(
        video_name="test.mp4",
        track_id=42,
        frame_idx=100,
        bbox="1,2,3,4",
        ratio_no_helmet=0.9,
        image_path="/tmp/x.jpg",
    )
    db_session.add(v)
    db_session.commit()

    stored = db_session.query(Violation).filter_by(track_id=42).first()
    assert stored is not None
    assert stored.video_name == "test.mp4"
    assert stored.ratio_no_helmet == 0.9


def test_created_at_set_automatically(db_session):
    v = Violation(
        video_name="a.mp4",
        track_id=1,
        frame_idx=1,
        bbox="1,1,1,1",
        ratio_no_helmet=1.0,
        image_path=None,
    )
    db_session.add(v)
    db_session.commit()
    assert v.created_at is not None


def test_multiple_violations_for_same_video(db_session):
    for track_id in range(5):
        db_session.add(
            Violation(
                video_name="bulk.mp4",
                track_id=track_id,
                frame_idx=track_id * 10,
                bbox="0,0,1,1",
                ratio_no_helmet=0.95,
                image_path=f"/tmp/{track_id}.jpg",
            )
        )
    db_session.commit()

    count = db_session.query(Violation).filter_by(video_name="bulk.mp4").count()
    assert count == 5
