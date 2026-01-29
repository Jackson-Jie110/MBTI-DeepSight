from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Q:
    dimension: str
    agree_pole: str
    text: str


@dataclass(frozen=True)
class _A:
    value: int
    question: _Q


def test_generate_dynamic_insights_strength_values_contrast():
    from app.services.reporting import generate_dynamic_insights

    dimensions = {
        "EI": {
            "first_pole": "E",
            "second_pole": "I",
            "first_percent": 82,
            "second_percent": 18,
            "gap_percent": 64,
        },
        "SN": {
            "first_pole": "S",
            "second_pole": "N",
            "first_percent": 55,
            "second_percent": 45,
            "gap_percent": 10,
        },
    }

    answers = [
        _A(value=5, question=_Q(dimension="SN", agree_pole="S", text="我更愿意依据事实与经验做判断。")),
        _A(value=5, question=_Q(dimension="EI", agree_pole="I", text="我需要独处来恢复能量。")),
    ]

    insights = generate_dynamic_insights(dimensions, answers)
    assert isinstance(insights, list)
    assert any("非常均衡" in x for x in insights)
    assert any("极致" in x for x in insights)
    assert any("坚定立场" in x for x in insights)
    assert any("尽管你整体偏向" in x for x in insights)


def test_build_report_context_includes_insights():
    from app.services.reporting import build_report_context

    dimensions = {
        "EI": {
            "first_pole": "E",
            "second_pole": "I",
            "first_percent": 82,
            "second_percent": 18,
            "gap_percent": 64,
        }
    }

    answers = [_A(value=5, question=_Q(dimension="EI", agree_pole="I", text="我需要独处来恢复能量。"))]
    ctx = build_report_context("ENTJ", dimensions, answers=answers, boundary_notes=[])
    assert "insights" in ctx
    assert isinstance(ctx["insights"], list)


def test_ai_stream_endpoint_returns_sse(client, db):
    from app.models import Question

    pole = {"EI": "E", "SN": "S", "TF": "T", "JP": "J"}
    for dim in ["EI", "SN", "TF", "JP"]:
        for i in range(5):
            db.add(
                Question(
                    dimension=dim,
                    agree_pole=pole[dim],
                    text=f"[{dim}] 测试题 {i + 1}",
                    is_active=True,
                    source="test",
                )
            )
    db.commit()

    client.get("/")
    csrf = client.cookies.get("csrf_token")
    assert csrf

    client.post(
        "/start",
        data={"mode": "20", "resume_expiry": "7d", "csrf_token": csrf},
        follow_redirects=False,
    )

    from app.models import Test, TestItem
    from app.services.tokens import hash_token

    secret = "dev-secret-change-me"
    test_token = client.cookies.get("test_token")
    test_hash = hash_token(test_token, secret=secret)
    test_row = db.query(Test).filter(Test.test_token_hash == test_hash).one()
    items = (
        db.query(TestItem).filter(TestItem.test_id == test_row.id).order_by(TestItem.position.asc()).all()
    )

    for idx, it in enumerate(items, start=1):
        client.post(
            "/test",
            data={
                "csrf_token": csrf,
                "position": str(idx),
                "question_id": str(it.question_id),
                "value": "5",
                "nav": "next",
            },
            follow_redirects=False,
        )

    client.get("/finish")
    finish = client.post(
        "/finish",
        data={"csrf_token": csrf, "share_expiry": "permanent"},
        follow_redirects=False,
    )
    assert finish.status_code == 303
    assert "/result/" in finish.headers["location"]
    share_token = finish.headers["location"].split("/result/")[1]

    sse = client.get(f"/result/ai_stream/{share_token}")
    assert sse.status_code == 200
    assert sse.headers["content-type"].startswith("text/event-stream")
    assert "data:" in sse.text

