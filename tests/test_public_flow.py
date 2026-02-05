from __future__ import annotations


def test_home_page_ok(client):
    r = client.get("/")
    assert r.status_code == 200


def _seed_questions(db, per_dim: int):
    from app.models import Question

    pole = {"EI": "E", "SN": "S", "TF": "T", "JP": "J"}
    for dim in ["EI", "SN", "TF", "JP"]:
        for i in range(per_dim):
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


def test_start_creates_session_and_redirects_to_test(client, db):
    _seed_questions(db, per_dim=5)

    home = client.get("/")
    assert home.status_code == 200

    csrf = client.cookies.get("csrf_token")
    assert csrf

    r = client.post(
        "/start",
        data={"mode": "20", "resume_expiry": "7d", "csrf_token": csrf},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"].startswith("/test")

    page = client.get("/test")
    assert page.status_code == 200
    assert "测试题" in page.text


def test_start_auto_seeds_when_empty(client):
    home = client.get("/")
    assert home.status_code == 200
    csrf = client.cookies.get("csrf_token")
    assert csrf

    r = client.post(
        "/start",
        data={"mode": "20", "resume_expiry": "7d", "csrf_token": csrf},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"].startswith("/test")


def test_answer_next_navigates(client, db):
    _seed_questions(db, per_dim=5)
    client.get("/")
    csrf = client.cookies.get("csrf_token")

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
    item1 = db.query(TestItem).filter(TestItem.test_id == test_row.id, TestItem.position == 1).one()

    r = client.post(
        "/test",
        data={
            "csrf_token": csrf,
            "position": "1",
            "question_id": str(item1.question_id),
            "value": "5",
            "nav": "next",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"].startswith("/test")


def test_complete_flow_generates_result(client, db):
    _seed_questions(db, per_dim=5)
    client.get("/")
    csrf = client.cookies.get("csrf_token")

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
    assert len(items) == 20

    for idx, it in enumerate(items, start=1):
        nav = "next"
        r = client.post(
            "/test",
            data={
                "csrf_token": csrf,
                "position": str(idx),
                "question_id": str(it.question_id),
                "value": "5",
                "nav": nav,
            },
            follow_redirects=False,
        )
        assert r.status_code == 303

    finish = client.get("/finish")
    assert finish.status_code == 200

    r2 = client.post(
        "/finish",
        data={"csrf_token": csrf, "share_expiry": "permanent"},
        follow_redirects=False,
    )
    assert r2.status_code == 303
    assert "/result/" in r2.headers["location"]

    result = client.get(r2.headers["location"])
    assert result.status_code == 200
    assert "ESTJ" in result.text
    assert 'id="ai-result-section"' in result.text
    assert "/result/ai_content/" in result.text
