"""Presets as workflows (spec 029 R7): export choices, where they came from,
applying one to a project (a copy), and New run starting from one."""

from __future__ import annotations

CHOICES = {"table": "parquet", "elan": False, "methods": True}
SOURCE = {"kind": "project", "id": "d1", "name": "QS"}


def _workflow(api, name="Faces 40", **fields):
    r = api.client.post(
        "/api/v1/presets/",
        json={
            "name": name,
            "selected_pipelines": ["face_analysis"],
            "config": {"face_analysis": {"min_face_size": 40}},
        }
        | fields,
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_export_choices_and_source_round_trip(api):
    made = _workflow(api, export_choices=CHOICES, source=SOURCE)

    assert (made["export_choices"], made["source"]) == (CHOICES, SOURCE)
    exported = api.client.get(f"/api/v1/presets/{made['id']}").json()
    imported = api.client.post(
        "/api/v1/presets/", json=exported | {"name": "Faces 40 copy"}
    ).json()
    assert (imported["export_choices"], imported["source"]) == (CHOICES, SOURCE)

    updated = api.client.put(
        f"/api/v1/presets/{made['id']}", json={"export_choices": {"table": "csv"}}
    ).json()
    assert updated["export_choices"] == {"table": "csv"}
    assert updated["source"] == SOURCE


def test_a_preset_from_before_has_neither(api):
    made = _workflow(api)

    assert (made["export_choices"], made["source"]) == (None, None)


def test_new_run_starts_from_a_chosen_workflow(api):
    workflow = _workflow(api)
    project = api.project("QS")

    found = api.client.get(
        "/api/v1/videos/run-settings",
        params={"project": project["id"], "workflow": workflow["id"]},
    ).json()

    assert found["selected_pipelines"] == ["face_analysis"]
    assert found["source"]["kind"] == "workflow"
    assert found["source"]["name"] == "Faces 40"


def test_applying_copies_and_later_edits_change_nothing(api):
    workflow = _workflow(api)
    project = api.project("QS")

    applied = api.client.put(
        f"/api/v1/datasets/{project['id']}",
        json={
            "current_settings": {
                "selected_pipelines": workflow["selected_pipelines"],
                "config": workflow["config"],
                "source": {
                    "kind": "workflow",
                    "id": workflow["id"],
                    "name": workflow["name"],
                },
            }
        },
    )
    assert applied.status_code == 200, applied.text
    api.client.put(
        f"/api/v1/presets/{workflow['id']}",
        json={"name": "Faces 50", "config": {"face_analysis": {"min_face_size": 50}}},
    )

    found = api.client.get(
        "/api/v1/videos/run-settings", params={"project": project["id"]}
    ).json()

    assert found["config"] == {"face_analysis": {"min_face_size": 40}}
    assert found["source"]["from"]["kind"] == "workflow"
    assert found["source"]["from"]["name"] == "Faces 40"
