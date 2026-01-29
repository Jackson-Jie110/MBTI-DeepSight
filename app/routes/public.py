from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request, Query
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, joinedload

from app.db import get_db
from app.models import Answer, Question, Test, TestItem
from app.seeding import seed_questions_if_empty
from app.services import ai
from app.services.reporting import build_report_context
from app.services.selection import select_balanced
from app.services.scoring import is_near_boundary, score_all
from app.services.tokens import expiry_from_choice, hash_token, new_url_token

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))

CSRF_COOKIE = "csrf_token"
TEST_COOKIE = "test_token"


def _app_secret() -> str:
    return os.getenv("MBTI_APP_SECRET", "dev-secret-change-me")


def _csrf_token(request: Request) -> tuple[str, bool]:
    csrf = request.cookies.get(CSRF_COOKIE)
    if csrf:
        return csrf, False
    return new_url_token(16), True


def _require_csrf(request: Request, csrf_token: str) -> None:
    cookie = request.cookies.get(CSRF_COOKIE)
    if not cookie or cookie != csrf_token:
        raise HTTPException(status_code=403, detail="CSRF 校验失败")


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get_test_from_cookie(request: Request, db: Session) -> Test:
    secret = _app_secret()

    token = request.cookies.get(TEST_COOKIE)
    if token:
        token_hash = hash_token(token, secret=secret)
        test_row = db.query(Test).filter(Test.test_token_hash == token_hash).one_or_none()
        if test_row:
            return _validate_in_progress_test(db, test_row)

    raise HTTPException(status_code=401, detail="缺少测试会话")


def _validate_in_progress_test(db: Session, test_row: Test) -> Test:
    if test_row.status != "in_progress":
        raise HTTPException(status_code=400, detail="该测试已结束")
    if test_row.resume_expires_at and datetime.now(timezone.utc) > _as_utc(test_row.resume_expires_at):
        test_row.status = "expired"
        db.commit()
        raise HTTPException(status_code=401, detail="该测试已过期")
    return test_row


@router.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    db: Session = Depends(get_db),
    error: str | None = Query(None),
):
    # Minimal home page; enhanced later with resume/continue logic.
    has_in_progress = False
    try:
        _get_test_from_cookie(request, db)
        has_in_progress = True
    except HTTPException:
        has_in_progress = False

    error_message = None
    if error == "question_bank_insufficient":
        error_message = "题库题目不足，无法抽取所需题量；请到管理后台补题后再试。"
    if error == "bad_request":
        error_message = "请求参数不合法，请重新选择后再提交。"

    csrf, should_set = _csrf_token(request)
    response = templates.TemplateResponse(
        request,
        "home.html",
        {
            "now": datetime.now(timezone.utc),
            "has_in_progress": has_in_progress,
            "csrf_token": csrf,
            "error_message": error_message,
        },
    )
    if should_set:
        response.set_cookie(CSRF_COOKIE, csrf, httponly=True, samesite="lax")
    return response


@router.post("/start")
def start(
    request: Request,
    db: Session = Depends(get_db),
    mode: int = Form(...),
    resume_expiry: str = Form(...),
    csrf_token: str = Form(...),
):
    _require_csrf(request, csrf_token)

    if mode not in (20, 40, 60):
        return RedirectResponse(url="/?error=bad_request", status_code=303)

    try:
        delta = expiry_from_choice(resume_expiry)
    except ValueError:
        return RedirectResponse(url="/?error=bad_request", status_code=303)
    now = datetime.now(timezone.utc)
    resume_expires_at = None if delta is None else (now + delta)

    secret = _app_secret()

    test_token = new_url_token()
    legacy_resume_token = new_url_token()
    legacy_resume_code = new_url_token(8)

    seed_questions_if_empty(db)
    active_questions = db.query(Question).filter(Question.is_active.is_(True)).all()
    try:
        picked = select_balanced(active_questions, total=mode)
    except ValueError:
        return RedirectResponse(url="/?error=question_bank_insufficient", status_code=303)

    test_row = Test(
        mode=mode,
        target_count=mode,
        extra_max=10,
        status="in_progress",
        resume_expires_at=resume_expires_at,
        test_token_hash=hash_token(test_token, secret=secret),
        resume_token_hash=hash_token(legacy_resume_token, secret=secret),
        resume_code_hash=hash_token(legacy_resume_code, secret=secret),
    )
    db.add(test_row)
    db.flush()

    for idx, q in enumerate(picked, start=1):
        db.add(TestItem(test_id=test_row.id, position=idx, question_id=q.id, is_extra=False))

    db.commit()

    response = RedirectResponse(url="/test", status_code=303)
    max_age = None if delta is None else int(delta.total_seconds())
    response.set_cookie(TEST_COOKIE, test_token, httponly=True, samesite="lax", max_age=max_age)
    return response


@router.get("/test", response_class=HTMLResponse)
def test_page(request: Request, db: Session = Depends(get_db), pos: int | None = None):
    test_row = _get_test_from_cookie(request, db)

    items = (
        db.query(TestItem)
        .filter(TestItem.test_id == test_row.id)
        .order_by(TestItem.position.asc())
        .all()
    )
    if not items:
        raise HTTPException(status_code=400, detail="该测试没有题目")

    answers = db.query(Answer).filter(Answer.test_id == test_row.id).all()
    answer_map = {a.question_id: a.value for a in answers}

    total = len(items)
    if pos is None:
        next_pos = None
        for it in items:
            if it.question_id not in answer_map:
                next_pos = it.position
                break
        pos = next_pos or items[-1].position

    pos = max(1, min(total, int(pos)))
    item = items[pos - 1]
    question = db.query(Question).filter(Question.id == item.question_id).one()
    picked_value = answer_map.get(question.id)

    csrf, should_set = _csrf_token(request)

    response = templates.TemplateResponse(
        request,
        "test.html",
        {
            "position": pos,
            "total": total,
            "question": question,
            "picked_value": picked_value,
            "is_extra": item.is_extra,
            "csrf_token": csrf,
        },
    )
    if should_set:
        response.set_cookie(CSRF_COOKIE, csrf, httponly=True, samesite="lax")
    return response


@router.post("/test")
def test_submit(
    request: Request,
    db: Session = Depends(get_db),
    csrf_token: str = Form(...),
    position: int = Form(...),
    question_id: int = Form(...),
    value: int | None = Form(None),
    nav: str = Form(...),
):
    _require_csrf(request, csrf_token)
    test_row = _get_test_from_cookie(request, db)

    item = (
        db.query(TestItem)
        .filter(TestItem.test_id == test_row.id, TestItem.position == int(position))
        .one_or_none()
    )
    if not item or item.question_id != int(question_id):
        raise HTTPException(status_code=400, detail="题目位置不匹配")

    if value is not None:
        v = int(value)
        if v < 1 or v > 5:
            raise HTTPException(status_code=400, detail="选项不合法")

        existing = (
            db.query(Answer)
            .filter(Answer.test_id == test_row.id, Answer.question_id == int(question_id))
            .one_or_none()
        )
        if existing:
            existing.value = v
            existing.answered_at = datetime.now(timezone.utc)
        else:
            db.add(
                Answer(
                    test_id=test_row.id,
                    question_id=int(question_id),
                    value=v,
                    answered_at=datetime.now(timezone.utc),
                )
            )
        db.commit()

    if nav == "exit":
        return RedirectResponse(url="/", status_code=303)

    total = db.query(TestItem).filter(TestItem.test_id == test_row.id).count()
    if nav == "prev":
        new_pos = max(1, int(position) - 1)
        return RedirectResponse(url=f"/test?pos={new_pos}", status_code=303)
    if nav == "next":
        new_pos = int(position) + 1
        if new_pos > total:
            return RedirectResponse(url="/finish", status_code=303)
        return RedirectResponse(url=f"/test?pos={new_pos}", status_code=303)

    raise HTTPException(status_code=400, detail="未知导航操作")


def _load_test_questions(db: Session, test_id: int) -> tuple[list[TestItem], list[Question]]:
    rows = (
        db.query(TestItem, Question)
        .join(Question, Question.id == TestItem.question_id)
        .filter(TestItem.test_id == test_id)
        .order_by(TestItem.position.asc())
        .all()
    )
    items = [it for it, _ in rows]
    questions = [q for _, q in rows]
    return items, questions


def _load_answers(db: Session, test_id: int) -> dict[int, int]:
    answers = db.query(Answer).filter(Answer.test_id == test_id).all()
    return {a.question_id: int(a.value) for a in answers}


def _first_missing_position(items: list[TestItem], answer_map: dict[int, int]) -> int | None:
    for it in items:
        if it.question_id not in answer_map:
            return int(it.position)
    return None


@router.get("/finish", response_class=HTMLResponse)
def finish_page(request: Request, db: Session = Depends(get_db)):
    test_row = _get_test_from_cookie(request, db)
    items, questions = _load_test_questions(db, test_row.id)
    answer_map = _load_answers(db, test_row.id)

    missing = _first_missing_position(items, answer_map)
    if missing is not None:
        return RedirectResponse(url=f"/test?pos={missing}", status_code=303)

    scoring = score_all(
        [{"id": q.id, "dimension": q.dimension, "agree_pole": q.agree_pole} for q in questions],
        answer_map,
    )
    dims = scoring["dimensions"]

    extra_count = sum(1 for it in items if it.is_extra)
    if extra_count < int(test_row.extra_max):
        near = [d for d in ["EI", "SN", "TF", "JP"] if is_near_boundary(int(dims[d]["first_percent"]), threshold_gap_percent=10)]
        if near:
            used_ids = {it.question_id for it in items}
            near_sorted = sorted(near, key=lambda d: int(dims[d]["gap_percent"]))
            for dim in near_sorted:
                q = (
                    db.query(Question)
                    .filter(
                        Question.is_active.is_(True),
                        Question.dimension == dim,
                        ~Question.id.in_(used_ids),
                    )
                    .first()
                )
                if q:
                    new_pos = len(items) + 1
                    db.add(TestItem(test_id=test_row.id, position=new_pos, question_id=q.id, is_extra=True))
                    db.commit()
                    return RedirectResponse(url=f"/test?pos={new_pos}", status_code=303)

    csrf, should_set = _csrf_token(request)
    response = templates.TemplateResponse(
        request,
        "finish.html",
        {
            "type_code": scoring["type"],
            "dimensions": [dims[d] for d in ["EI", "SN", "TF", "JP"]],
            "csrf_token": csrf,
        },
    )
    if should_set:
        response.set_cookie(CSRF_COOKIE, csrf, httponly=True, samesite="lax")
    return response


@router.post("/finish")
def finish_submit(
    request: Request,
    db: Session = Depends(get_db),
    csrf_token: str = Form(...),
    share_expiry: str = Form(...),
):
    _require_csrf(request, csrf_token)
    test_row = _get_test_from_cookie(request, db)

    items, questions = _load_test_questions(db, test_row.id)
    answer_map = _load_answers(db, test_row.id)
    missing = _first_missing_position(items, answer_map)
    if missing is not None:
        return RedirectResponse(url=f"/test?pos={missing}", status_code=303)

    scoring = score_all(
        [{"id": q.id, "dimension": q.dimension, "agree_pole": q.agree_pole} for q in questions],
        answer_map,
    )
    dims = scoring["dimensions"]

    boundary_notes: list[str] = []
    for d in ["EI", "SN", "TF", "JP"]:
        fp = int(dims[d]["first_percent"])
        sp = int(dims[d]["second_percent"])
        if is_near_boundary(fp, threshold_gap_percent=10):
            boundary_notes.append(f"{d} 接近边界（{dims[d]['first_pole']} {fp}% / {dims[d]['second_pole']} {sp}%）")

    delta = expiry_from_choice(share_expiry)
    now = datetime.now(timezone.utc)
    share_expires_at = None if delta is None else (now + delta)

    secret = _app_secret()
    share_token = new_url_token()

    test_row.share_token_hash = hash_token(share_token, secret=secret)
    test_row.share_expires_at = share_expires_at
    test_row.result_type = scoring["type"]
    test_row.result_json = {
        "type": scoring["type"],
        "dimensions": dims,
        "boundary_notes": boundary_notes,
    }
    test_row.status = "completed"
    test_row.completed_at = now

    db.commit()

    return RedirectResponse(url=f"/result/{share_token}", status_code=303)


@router.get("/result/{share_token}", response_class=HTMLResponse)
def result_page(request: Request, share_token: str, db: Session = Depends(get_db)):
    secret = _app_secret()
    token_hash = hash_token(share_token, secret=secret)
    test_row = (
        db.query(Test)
        .options(joinedload(Test.answers).joinedload(Answer.question))
        .filter(Test.share_token_hash == token_hash)
        .one_or_none()
    )
    if not test_row or test_row.status != "completed" or not test_row.result_json:
        raise HTTPException(status_code=404, detail="结果不存在")

    if test_row.share_expires_at and datetime.now(timezone.utc) > _as_utc(test_row.share_expires_at):
        return templates.TemplateResponse(request, "result_expired.html", {"type_code": test_row.result_type})

    result = dict(test_row.result_json)
    type_code = result.get("type") or test_row.result_type
    dims = result.get("dimensions") or {}
    boundary_notes = result.get("boundary_notes") or []

    report = build_report_context(
        str(type_code),
        dims,
        boundary_notes=list(boundary_notes),
        answers=list(getattr(test_row, "answers", []) or []),
    )

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "type_code": type_code,
            "dimensions": [dims.get(d) for d in ["EI", "SN", "TF", "JP"] if dims.get(d)],
            "report": report,
            "share_url": str(request.base_url)[:-1] + f"/result/{share_token}",
            "share_token": share_token,
        },
    )


@router.get("/result/ai_stream/{share_token}")
async def ai_stream(request: Request, share_token: str, db: Session = Depends(get_db)):
    # 1. 验证 Token
    secret = _app_secret()
    token_hash = hash_token(share_token, secret=secret)

    # 预加载 answers 以便后续分析（即使暂时不用，也保持结构正确）
    test_row = (
        db.query(Test)
        .options(joinedload(Test.answers))
        .filter(Test.share_token_hash == token_hash)
        .one_or_none()
    )

    if not test_row or not test_row.result_json:
        # 找不到记录，返回空流
        async def empty_generator():
            yield "data: ⚠️ 无法找到测试记录，请刷新重试。\n\n"

        return StreamingResponse(empty_generator(), media_type="text/event-stream")

    # 2. 准备数据
    result = dict(test_row.result_json)
    type_code = result.get("type", "Unknown")

    # 简化的洞察列表（暂时先传空列表，确保核心功能跑通，避免因数据处理逻辑崩溃）
    insights = []

    # 3. 启动流式响应
    # 注意：这里直接调用 ai.generate_report_stream，它内部已经有了 try-except 保护
    return StreamingResponse(
        ai.generate_report_stream(type_code, insights),
        media_type="text/event-stream",
    )
