"""Researchers' notes on a video (spec 028 R9)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from videoannotator import video_notes
from videoannotator.database import models  # noqa: F401  (registers the tables)
from videoannotator.database.database import Base

ALICE = {"id": "u1", "username": "alice"}
BOB = {"id": "u2", "email": "bob@lab.org"}


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def video(tmp_path) -> Path:
    path = tmp_path / "p01_t1.mp4"
    path.write_bytes(b"video")
    return path


def _note(db, video, start=1.0, end=None, text="infant looks away", **kw):
    return video_notes.create(
        db, path=video, start=start, end=end, text=text, user=ALICE, **kw
    )


def test_a_note_is_trimmed_and_records_its_author(db, video):
    note = _note(db, video, text="  infant looks away \n")
    assert note.text == "infant looks away"
    assert (note.author_id, note.author_name) == ("u1", "alice")
    assert note.video_path == str(video.resolve())
    assert note.video_name == "p01_t1.mp4"


def test_a_span_given_end_first_is_stored_in_order(db, video):
    note = _note(db, video, start=5.0, end=2.0)
    assert (note.start_sec, note.end_sec) == (2.0, 5.0)


@pytest.mark.parametrize(
    ("start", "end", "text", "duration", "code"),
    [
        (1.0, None, "   ", None, "NOTE_TEXT_REQUIRED"),
        (1.0, None, "x" * 2001, None, "NOTE_TEXT_TOO_LONG"),
        (-1.0, None, "x", None, "NOTE_TIME_INVALID"),
        (2.0, 2.0, "x", None, "NOTE_TIME_INVALID"),
        (1.0, 30.0, "x", 20.0, "NOTE_TIME_INVALID"),
    ],
)
def test_notes_that_cant_be_saved(db, video, start, end, text, duration, code):
    with pytest.raises(video_notes.NoteError) as e:
        _note(db, video, start=start, end=end, text=text, duration=duration)
    assert e.value.code == code


def test_notes_are_listed_by_start_then_creation_and_two_may_share_a_time(db, video):
    later = _note(db, video, start=9.0, text="later")
    first = _note(db, video, start=3.0, text="first")
    second = _note(db, video, start=3.0, text="second")
    assert [n.id for n in video_notes.list_for(db, video)] == [
        first.id,
        second.id,
        later.id,
    ]


def test_notes_are_found_by_the_resolved_path(db, video, tmp_path):
    link = tmp_path / "link.mp4"
    link.symlink_to(video)
    _note(db, video)
    assert len(video_notes.for_media_path(db, link)) == 1
    assert video_notes.for_media_path(db, None) == []


def test_anyone_may_edit_and_is_recorded_as_the_editor(db, video):
    note = _note(db, video)
    changed = video_notes.update(db, str(note.id), user=BOB, text="looks back")
    assert changed.text == "looks back"
    assert changed.author_id == "u1"
    assert changed.updated_by == "u2"
    assert changed.updated_at is not None
    moment = video_notes.update(db, str(note.id), user=BOB, end=None)
    assert moment.end_sec is None


def test_sha256_is_kept_when_known(db, video):
    assert _note(db, video, sha256="abc").video_sha256 == "abc"


def test_delete_and_not_found(db, video):
    note = _note(db, video)
    video_notes.delete(db, str(note.id))
    assert video_notes.list_for(db, video) == []
    with pytest.raises(video_notes.NoteNotFound):
        video_notes.get(db, str(note.id))
    with pytest.raises(video_notes.NoteNotFound):
        video_notes.get(db, "not-a-uuid")


def test_to_dict_has_the_contract_fields(db, video):
    data = video_notes.to_dict(_note(db, video))
    assert set(data) == {
        "id",
        "video_path",
        "start_sec",
        "end_sec",
        "text",
        "author_id",
        "author_name",
        "created_at",
        "updated_at",
        "updated_by",
    }
    assert data["created_at"].endswith("+00:00")
