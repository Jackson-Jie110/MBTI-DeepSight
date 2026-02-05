from __future__ import annotations

from datetime import datetime, timezone
import html
import hashlib
import json
import os
import re
import traceback
from pathlib import Path
from urllib.parse import quote_plus

from fastapi import APIRouter, Depends, Form, Request, Query
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, joinedload

from app.db import get_db
from app.models import Answer, Question, Test, TestItem
from app.seeding import seed_questions_if_empty
from app.services.reporting import build_report_context
from app.services.selection import select_balanced
from app.services.scoring import is_near_boundary, score_all
from app.services.tokens import expiry_from_choice, hash_token, new_url_token

try:
    import markdown  # type: ignore
except Exception:  # pragma: no cover
    markdown = None

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover
    AsyncOpenAI = None

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

    rows = (
        db.query(TestItem, Question)
        .join(Question, Question.id == TestItem.question_id)
        .filter(TestItem.test_id == test_row.id)
        .order_by(TestItem.position.asc())
        .all()
    )
    if not rows:
        raise HTTPException(status_code=400, detail="该测试没有题目")

    items = [it for it, _q in rows]
    used_ids = {int(it.question_id) for it in items}
    answers = db.query(Answer).filter(Answer.test_id == test_row.id).all()
    answer_map = {int(a.question_id): int(a.value) for a in answers}

    questions_data = []
    for it, q in rows:
        questions_data.append(
            {
                "id": int(q.id),
                "text": q.text,
                "dimension": q.dimension,
                "agree_pole": q.agree_pole,
                "position": int(it.position),
                "is_extra": bool(it.is_extra),
            }
        )

    # Tie-breaker questions pool (preferred: source == "tie_breaker"; fallback: any unused active questions).
    # Note: the current Question model doesn't have a `category` column; `source` acts as the tie-breaker marker.
    tie_rows = db.query(Question).filter(Question.is_active.is_(True), Question.source == "tie_breaker").all()
    if not tie_rows:
        q = db.query(Question).filter(Question.is_active.is_(True))
        if used_ids:
            q = q.filter(~Question.id.in_(used_ids))
        tie_rows = q.all()

    dim_pair = {"EI": "E-I", "SN": "S-N", "TF": "T-F", "JP": "J-P"}
    tie_breakers: dict[str, list[dict[str, object]]] = {}
    for q in tie_rows:
        key = dim_pair.get(str(q.dimension), str(q.dimension))
        tie_breakers.setdefault(key, []).append(
            {
                "id": int(q.id),
                "text": q.text,
                "dimension": q.dimension,
                "agree_pole": q.agree_pole,
            }
        )

    initial_index = 0
    for idx, it in enumerate(items):
        if int(it.question_id) not in answer_map:
            initial_index = idx
            break
    else:
        initial_index = max(0, len(items) - 1)

    csrf, should_set = _csrf_token(request)

    response = templates.TemplateResponse(
        request,
        "test.html",
        {
            "questions_json": json.dumps(questions_data, ensure_ascii=False),
            "answers_json": json.dumps(answer_map, ensure_ascii=False),
            "tie_breakers_json": json.dumps(tie_breakers, ensure_ascii=False),
            "initial_index": initial_index,
            "csrf_token": csrf,
        },
    )
    if should_set:
        response.set_cookie(CSRF_COOKIE, csrf, httponly=True, samesite="lax")
    return response


@router.post("/test/answers")
async def test_answers(request: Request, db: Session = Depends(get_db)):
    payload = await request.json()
    csrf_token = payload.get("csrf_token")
    if not isinstance(csrf_token, str):
        raise HTTPException(status_code=400, detail="缺少 csrf_token")
    _require_csrf(request, csrf_token)

    test_row = _get_test_from_cookie(request, db)

    intent = payload.get("intent") or "save"
    raw_answers = payload.get("answers")
    if raw_answers is None:
        raise HTTPException(status_code=400, detail="缺少 answers")

    items = (
        db.query(TestItem)
        .filter(TestItem.test_id == test_row.id)
        .order_by(TestItem.position.asc())
        .all()
    )
    item_ids = {int(it.question_id) for it in items}
    if not item_ids:
        raise HTTPException(status_code=400, detail="该测试没有题目")

    # De-duplicate answers by question_id; keep the last value.
    to_upsert_map: dict[int, int] = {}
    if isinstance(raw_answers, dict):
        for k, v in raw_answers.items():
            try:
                qid = int(k)
                val = int(v)
            except Exception:
                continue
            to_upsert_map[qid] = val
    elif isinstance(raw_answers, list):
        for it in raw_answers:
            if not isinstance(it, dict):
                continue
            try:
                qid = int(it.get("question_id"))
                val = int(it.get("value"))
            except Exception:
                continue
            to_upsert_map[qid] = val
    else:
        raise HTTPException(status_code=400, detail="answers 格式不合法")

    to_upsert: list[tuple[int, int]] = list(to_upsert_map.items())

    # Allow client-side scheduled extra (tie-breaker) questions:
    # create TestItem rows for new question ids so server-side scoring includes them.
    extra_qids = [qid for qid, _v in to_upsert if qid not in item_ids]
    if extra_qids:
        existing_extra = sum(1 for it in items if bool(getattr(it, "is_extra", False)))
        remaining = max(0, int(test_row.extra_max) - int(existing_extra))
        if remaining <= 0:
            raise HTTPException(status_code=400, detail="加测题数量已达上限")

        extra_qids = extra_qids[:remaining]
        has_dedicated_tie_pool = (
            db.query(Question.id)
            .filter(Question.is_active.is_(True), Question.source == "tie_breaker")
            .limit(1)
            .scalar()
            is not None
        )

        q_query = db.query(Question).filter(Question.id.in_(extra_qids), Question.is_active.is_(True))
        if has_dedicated_tie_pool:
            q_query = q_query.filter(Question.source == "tie_breaker")
        q_rows = q_query.all()
        q_by_id = {int(q.id): q for q in q_rows}

        max_pos = int(items[-1].position) if items else 0
        allowed_dims = {"EI", "SN", "TF", "JP"}
        for qid in extra_qids:
            q = q_by_id.get(int(qid))
            if not q:
                continue
            if str(q.dimension) not in allowed_dims:
                continue
            max_pos += 1
            db.add(TestItem(test_id=test_row.id, position=max_pos, question_id=int(qid), is_extra=True))
        db.commit()

        # refresh item ids after appending extras
        items = (
            db.query(TestItem)
            .filter(TestItem.test_id == test_row.id)
            .order_by(TestItem.position.asc())
            .all()
        )
        item_ids = {int(it.question_id) for it in items}

    now = datetime.now(timezone.utc)
    existing = db.query(Answer).filter(Answer.test_id == test_row.id).all()
    existing_map = {int(a.question_id): a for a in existing}

    for qid, val in to_upsert:
        if qid not in item_ids:
            continue
        if val < 1 or val > 5:
            continue
        row = existing_map.get(qid)
        if row:
            row.value = val
            row.answered_at = now
        else:
            db.add(Answer(test_id=test_row.id, question_id=qid, value=val, answered_at=now))

    db.commit()

    if intent == "finish":
        saved_ids = {
            int(qid)
            for (qid,) in db.query(Answer.question_id).filter(Answer.test_id == test_row.id).all()
        }
        missing = None
        for pos_it in (
            db.query(TestItem)
            .filter(TestItem.test_id == test_row.id)
            .order_by(TestItem.position.asc())
            .all()
        ):
            if int(pos_it.question_id) not in saved_ids:
                missing = int(pos_it.position)
                break
        if missing is not None:
            return JSONResponse({"status": "incomplete", "redirect": "/test", "missing_position": missing})
        return JSONResponse({"status": "complete", "redirect": "/finish"})

    if intent == "exit":
        return JSONResponse({"status": "exit", "redirect": "/"})

    return JSONResponse({"status": "saved"})


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
        return RedirectResponse(url="/test", status_code=303)

    scoring = score_all(
        [{"id": q.id, "dimension": q.dimension, "agree_pole": q.agree_pole} for q in questions],
        answer_map,
    )
    dims = scoring["dimensions"]

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
        return RedirectResponse(url="/test", status_code=303)

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
            "dimensions_json": json.dumps(dims, ensure_ascii=False),
            "report": report,
            "share_url": str(request.base_url)[:-1] + f"/result/{share_token}",
            "share_token": share_token,
            "result_ai_async_url": str(request.url_for("result_ai_content", share_token=share_token)),
        },
    )


def _clean_ai_markdown(md: str) -> str:
    text = (md or "").replace("TAGS_SHORT_READ_WARNING false", "").replace("TAGS_SHORT_READ_WARNING true", "")
    # Normalize headings: remove leading spaces before #'s, and normalize space after hashes.
    text = re.sub(r"^\s+(#{1,6})", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"(#{1,6})[\s\u3000\u00A0]+", r"\1 ", text, flags=re.MULTILINE)
    text = re.sub(r"(#{1,6} \*\*)[\s\u3000\u00A0]+", r"\1", text, flags=re.MULTILINE)
    return text.strip()


def _markdown_to_html(md: str) -> str:
    # Escape raw HTML first to avoid injection, while keeping Markdown syntax intact.
    safe_md = html.escape(md or "")
    if markdown is None:
        return "<pre>" + safe_md + "</pre>"
    try:
        return markdown.markdown(safe_md, extensions=["extra", "nl2br"])
    except Exception:
        return markdown.markdown(safe_md)


@router.get("/result/ai_content/{share_token}", response_class=HTMLResponse, name="result_ai_content")
async def result_ai_content(request: Request, share_token: str, db: Session = Depends(get_db)):
    secret = _app_secret()
    token_hash = hash_token(share_token, secret=secret)
    test_row = (
        db.query(Test)
        .options(joinedload(Test.answers).joinedload(Answer.question))
        .filter(Test.share_token_hash == token_hash)
        .one_or_none()
    )
    if not test_row or test_row.status != "completed" or not test_row.result_json:
        return templates.TemplateResponse(
            request,
            "partials/result_ai_content.html",
            {"ai_content_html": _markdown_to_html("**⚠️ 结果不存在或已失效。**")},
        )

    if test_row.share_expires_at and datetime.now(timezone.utc) > _as_utc(test_row.share_expires_at):
        return templates.TemplateResponse(
            request,
            "partials/result_ai_content.html",
            {"ai_content_html": _markdown_to_html("**⚠️ 分享链接已过期。**")},
        )

    result = test_row.result_json or {}
    type_code = str(result.get("type") or test_row.result_type or "Unknown")
    dims: dict[str, dict] = dict(result.get("dimensions") or {})
    boundary_notes = list(result.get("boundary_notes") or [])
    answers = list(getattr(test_row, "answers", []) or [])

    def _dimensions_str() -> str:
        parts: list[str] = []
        for dim in ["EI", "SN", "TF", "JP"]:
            info = dims.get(dim) or {}
            fp = info.get("first_percent")
            sp = info.get("second_percent")
            first = info.get("first_pole")
            second = info.get("second_pole")
            if fp is None or sp is None or not first or not second:
                continue
            parts.append(f"{first}:{fp}% / {second}:{sp}%")
        return ", ".join(parts) if parts else "维度数据缺失"

    def _extremes_str() -> str:
        items: list[str] = []
        for idx, a in enumerate(answers, start=1):
            try:
                v = int(getattr(a, "value", 0))
            except Exception:
                continue
            if v not in (1, 5):
                continue
            q = getattr(a, "question", None)
            if not q:
                continue
            text = str(getattr(q, "text", "") or "").strip()
            dim = str(getattr(q, "dimension", "") or "").strip()
            agree = str(getattr(q, "agree_pole", "") or "").strip()
            if len(dim) == 2 and len(agree) == 1:
                opposite = dim[1] if dim[0] == agree else (dim[0] if dim[1] == agree else "?")
            else:
                opposite = "?"
            pole = agree if v == 5 else opposite
            snippet = text[:48] + ("…" if len(text) > 48 else "")
            items.append(f"{idx}. [{dim}/{pole}向] {v}分：{snippet}")
            if len(items) >= 6:
                break
        return "；".join(items) if items else "未发现明显的极值作答（1分/5分）"

    insight_list = build_report_context(
        type_code,
        dims,
        boundary_notes=boundary_notes,
        answers=answers,
    ).get("insights", [])
    insight_list = [str(x).strip() for x in (insight_list or []) if str(x).strip()]

    tags: list[str] = []
    for dim, info in (dims or {}).items():
        try:
            gap = int(info.get("gap_percent"))
        except Exception:
            continue
        if gap < 20:
            tags.append(f"{dim}均衡")
        elif gap > 60:
            tags.append(f"{dim}极致")
    if any(getattr(a, "value", None) == 5 for a in answers):
        tags.append("立场坚定")
    if any("尽管你整体偏向" in s for s in insight_list):
        tags.append("反差发力")
    tags = tags[:6]

    dimensions_str = _dimensions_str()
    extreme_traits = _extremes_str()
    dynamic_insights = "；".join(insight_list[:3]) if insight_list else "暂无动态洞察"
    dynamic_tags = "动态标签：" + (", ".join([f"[{t}]" for t in tags]) if tags else "[稳定作答]")

    user_profile_context = f"""
用户MBTI类型：{type_code}
【维度数据】：{dimensions_str}
【极值特质】：{extreme_traits}
【行为标签】：{dynamic_tags}
【动态洞察】：{dynamic_insights}
请综合上述数据，忽略刻板印象，还原一个鲜活的人。
""".strip()

    system_prompt = f"""
你是一位洞察人性幽暗与光辉的心理学大师，正在使用 DeepSeek-V3.2 模型进行深度侧写。
{user_profile_context}

【指令】
1. 你的语言要像手术刀一样精准，剥开用户的社会面具，直指内心深处的矛盾与渴望。
2. 结合给出的维度数据和极值特质进行推理，不要只罗列优缺点。
3. 严禁输出任何开场白，严禁使用代码块。
4. 【排版严格要求】：
   - 所有标题（###）必须**顶格书写**，严禁在前面加空格，确保左对齐。
   - **关键格式**：标题的井号与文字之间**只能有一个空格**（例如 `### 标题`），严禁出现两个空格或全角空格。
   - 请务必使用 Markdown 加粗语法（如 **关键特质**）高亮文中那些直击灵魂、反直觉或最具冲击力的短句，每段至少包含 1-2 处高亮。

【输出模版】
### ️ 灵魂底色
(用一个带有灰度色彩的意象，如“**暴风雨中的灯塔**”，描述该人格最底层的核心驱动力。约 100 字)

###  镜像与阴影
(指出用户常常欺骗自己的一点，以及外界误解最深的一点。用“世人皆以为...，殊不知...”的句式。约 150 字)

###  给伤口的诗
(针对该人格最容易受挫的软肋，写一段治愈且充满力量的短句。约 80 字)
""".strip()

    user_prompt = f"我的 MBTI 类型是：{type_code}。请开始你的深度解读。"

    try:
        api_key = os.getenv("MBTI_AI_API_KEY")
        base_url = os.getenv("MBTI_AI_BASE_URL", "https://api.siliconflow.cn/v1")
        model = os.getenv("MBTI_AI_MODEL", "deepseek-ai/DeepSeek-V3.2")
        if not api_key:
            raise RuntimeError("系统未配置 MBTI_AI_API_KEY")
        if AsyncOpenAI is None:
            raise RuntimeError("OpenAI SDK 不可用")

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                timeout=60.0,
            )
        finally:
            try:
                await client.close()
            except Exception:
                pass

        ai_text = ""
        try:
            ai_text = resp.choices[0].message.content or ""
        except Exception:
            ai_text = ""

        if not ai_text.strip():
            raise RuntimeError("AI 返回内容为空")

        cleaned_md = _clean_ai_markdown(ai_text)
        ai_content_html = _markdown_to_html(cleaned_md)
    except Exception as e:
        err = str(e).strip()
        if len(err) > 240:
            err = err[:240] + "..."
        ai_content_html = _markdown_to_html(f"**❌ AI 生成失败**\n\n错误：{err}")

    return templates.TemplateResponse(
        request,
        "partials/result_ai_content.html",
        {"ai_content_html": ai_content_html},
    )


def _rpg_radar_from_dimensions(dims: dict[str, object]) -> list[int]:
    def clamp(v: float) -> int:
        return int(max(0, min(100, round(v))))

    def pole_percent(dim_key: str, pole: str) -> int:
        raw = dims.get(dim_key)
        if not isinstance(raw, dict):
            return 50
        first_pole = str(raw.get("first_pole") or "")
        second_pole = str(raw.get("second_pole") or "")
        first_percent = raw.get("first_percent")
        second_percent = raw.get("second_percent")
        try:
            fp = int(first_percent) if first_percent is not None else 50
            sp = int(second_percent) if second_percent is not None else 50
        except Exception:
            fp, sp = 50, 50

        if first_pole == pole:
            return max(0, min(100, fp))
        if second_pole == pole:
            return max(0, min(100, sp))
        return 50

    e = pole_percent("EI", "E")
    s = pole_percent("SN", "S")
    n = pole_percent("SN", "N")
    t = pole_percent("TF", "T")
    f = pole_percent("TF", "F")
    j = pole_percent("JP", "J")
    p = pole_percent("JP", "P")

    creativity = clamp(n * 0.8 + p * 0.2)
    execution = clamp(j * 0.8 + s * 0.2)
    logic = clamp(t)
    empathy = clamp(f)
    adaptability = clamp(p)
    social = clamp(e)

    return [creativity, execution, logic, empathy, adaptability, social]


def get_conflict_pair(dimensions: dict[str, int]) -> tuple[str, str]:
    pairs = [("E", "I"), ("S", "N"), ("T", "F"), ("J", "P")]
    best_pair: tuple[str, str] = ("T", "F")
    best_gap = 10**9
    for a, b in pairs:
        va = int(dimensions.get(a, 50))
        vb = int(dimensions.get(b, 50))
        gap = abs(va - vb)
        if gap < best_gap:
            best_gap = gap
            best_pair = (a, b)
    return best_pair


def get_fallback_data(error_msg: str) -> dict[str, object]:
    safe_err = (error_msg or "").strip()
    if len(safe_err) > 220:
        safe_err = safe_err[:220] + "..."

    return {
        "manual": {
            "do_list": ["(系统) AI 生成失败", "请检查后端日志", "错误详情见下方"],
            "dont_list": ["不要惊慌", "不要频繁刷新", "不要泄露密钥信息"],
            "recharge": f"错误追踪: {safe_err}",
        },
        "war": {
            "title": "系统异常 vs 调试模式",
            "description": (
                f"API 调用或解析失败。具体原因：{safe_err}。"
                "请检查 API Key 余额、网络连接、Base URL 或 JSON 格式约束。"
            ),
        },
    }


def _extract_json_object(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("```"):
        s = s.strip("`").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return s[start : end + 1]


def _normalize_letter_dimensions(dimensions_obj: object) -> dict[str, int]:
    # Accept either:
    # 1) {"E":60,"I":40,...} or
    # 2) {"EI":{"first_pole":"E","first_percent":60,"second_pole":"I","second_percent":40}, ...}
    out: dict[str, int] = {}
    if isinstance(dimensions_obj, dict):
        # direct letters
        letter_keys = {"E", "I", "S", "N", "T", "F", "J", "P"}
        if any(k in dimensions_obj for k in letter_keys):
            for k in letter_keys:
                v = dimensions_obj.get(k)
                try:
                    out[k] = max(0, min(100, int(v)))
                except Exception:
                    out[k] = 50
        else:
            for dim_key in ("EI", "SN", "TF", "JP"):
                raw = dimensions_obj.get(dim_key)
                if not isinstance(raw, dict):
                    continue
                try:
                    fpole = str(raw.get("first_pole"))
                    spole = str(raw.get("second_pole"))
                    fp = int(raw.get("first_percent"))
                    sp = int(raw.get("second_percent"))
                except Exception:
                    continue
                out[fpole] = max(0, min(100, fp))
                out[spole] = max(0, min(100, sp))

    # Ensure all exist.
    for k in ("E", "I", "S", "N", "T", "F", "J", "P"):
        out.setdefault(k, 50)
    return out


def _rpg_radar_from_letter_dimensions(dimensions: dict[str, int]) -> list[int]:
    def clamp(v: float) -> int:
        return int(max(0, min(100, round(v))))

    e = int(dimensions.get("E", 50))
    s = int(dimensions.get("S", 50))
    n = int(dimensions.get("N", 50))
    t = int(dimensions.get("T", 50))
    f = int(dimensions.get("F", 50))
    j = int(dimensions.get("J", 50))
    p = int(dimensions.get("P", 50))

    creativity = clamp(n * 0.8 + p * 0.2)
    execution = clamp(j * 0.8 + s * 0.2)
    logic = clamp(t)
    empathy = clamp(f)
    adaptability = clamp(p)
    social = clamp(e)
    return [creativity, execution, logic, empathy, adaptability, social]


def _analysis_core(mbti_type: str, dimensions_json: str) -> dict[str, object]:
    try:
        parsed = json.loads(dimensions_json)
    except Exception:
        parsed = {}

    letter_dims = _normalize_letter_dimensions(parsed)
    conflict_pair = get_conflict_pair(letter_dims)
    val1 = int(letter_dims.get(conflict_pair[0], 50))
    val2 = int(letter_dims.get(conflict_pair[1], 50))

    # Versus bar widths must sum to 100.
    total = max(1, val1 + val2)
    left_percent = int(round(val1 / total * 100))
    right_percent = 100 - left_percent
    if val2 > val1:
        conflict_pair = (conflict_pair[1], conflict_pair[0])
        val1, val2 = val2, val1
        left_percent, right_percent = right_percent, left_percent

    radar_data = _rpg_radar_from_letter_dimensions(letter_dims)

    return {
        "mbti_type": mbti_type,
        "dimensions_json": dimensions_json,
        "letter_dims": letter_dims,
        "conflict_pair": conflict_pair,
        "val1": val1,
        "val2": val2,
        "war_left_pole": conflict_pair[0],
        "war_left_percent": left_percent,
        "war_right_pole": conflict_pair[1],
        "war_right_percent": right_percent,
        "radar_data": radar_data,
    }


@router.get("/analysis", response_class=HTMLResponse)
def analysis_page_get(
    request: Request,
    db: Session = Depends(get_db),
    type_code: str = Query("", alias="type"),
    dimensions: str | None = Query(None),
):
    if not type_code or not dimensions:
        return RedirectResponse(url="/", status_code=303)

    core = _analysis_core(type_code, dimensions)
    base = str(request.url_for("analysis_async_content"))
    analysis_async_url = f"{base}?type={quote_plus(type_code)}&dimensions={quote_plus(dimensions)}"

    return templates.TemplateResponse(
        request,
        "analysis.html",
        {
            "type_code": type_code,
            "mbti_type": type_code,
            "radar_data_json": json.dumps(core["radar_data"], ensure_ascii=False),
            "analysis_async_url": analysis_async_url,
        },
    )


@router.post("/analysis", response_class=HTMLResponse)
def analysis_page_post(
    request: Request,
    db: Session = Depends(get_db),
    type_code: str = Form("", alias="type"),
    dimensions: str = Form(""),
):
    if not type_code or not dimensions:
        return RedirectResponse(url="/", status_code=303)

    core = _analysis_core(type_code, dimensions)
    base = str(request.url_for("analysis_async_content"))
    analysis_async_url = f"{base}?type={quote_plus(type_code)}&dimensions={quote_plus(dimensions)}"

    return templates.TemplateResponse(
        request,
        "analysis.html",
        {
            "type_code": type_code,
            "mbti_type": type_code,
            "radar_data_json": json.dumps(core["radar_data"], ensure_ascii=False),
            "analysis_async_url": analysis_async_url,
        },
    )


@router.get("/analysis/content", response_class=HTMLResponse, name="analysis_async_content")
async def analysis_async_content(
    request: Request,
    db: Session = Depends(get_db),
    type_code: str = Query("", alias="type"),
    dimensions: str | None = Query(None),
):
    if not type_code or not dimensions:
        return HTMLResponse("", status_code=400)

    core = _analysis_core(type_code, dimensions)
    conflict_pair = core["conflict_pair"]
    val1 = int(core["val1"])
    val2 = int(core["val2"])
    letter_dims = core["letter_dims"]

    prompt = f"""
用户MBTI: {type_code}
各维度分值: {json.dumps(letter_dims, ensure_ascii=False)}
内心最冲突的维度: {conflict_pair[0]} (score: {val1}) vs {conflict_pair[1]} (score: {val2}) - 分值极度接近。

请基于以上数据，生成“用户使用说明书”和“内心维度战争”分析。
必须严格输出纯 JSON 格式，无 Markdown：
{{
    "manual": {{
        "do_list": ["3个让该用户感到被理解的行为"],
        "dont_list": ["3个该用户的绝对雷区"],
        "recharge": "1个具体的快速回血方式"
    }},
    "war": {{
        "title": "冲突维度的具象化比喻 (如：理性的暴君 vs 感性的诗人)",
        "description": "深度分析这种纠结带来的困扰与优势。"
    }}
}}
""".strip()

    fun_data: dict[str, object]
    try:
        if AsyncOpenAI is None:
            raise RuntimeError("OpenAI SDK 未安装或导入失败")

        api_key = os.getenv("MBTI_AI_API_KEY")
        if not api_key:
            raise RuntimeError("未配置 MBTI_AI_API_KEY")

        base_url = os.getenv("MBTI_AI_BASE_URL", "https://api.siliconflow.cn/v1")
        model = os.getenv("MBTI_AI_MODEL", "deepseek-ai/DeepSeek-V3")
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你必须只输出纯 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                timeout=60.0,
            )
        finally:
            try:
                await client.close()
            except Exception:
                pass

        content = ""
        try:
            content = resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"AI 响应为空或结构异常: {e}") from e

        try:
            fun_data_obj = json.loads(content)
        except Exception:
            extracted = _extract_json_object(content)
            if not extracted:
                raise ValueError("AI 未返回可解析的 JSON 对象")
            fun_data_obj = json.loads(extracted)

        if not isinstance(fun_data_obj, dict):
            raise ValueError("AI JSON 顶层不是对象")
        manual = fun_data_obj.get("manual")
        war = fun_data_obj.get("war")
        if not isinstance(manual, dict) or not isinstance(war, dict):
            raise ValueError("AI JSON 缺少 manual/war 对象")
        if not isinstance(manual.get("do_list"), list) or not isinstance(manual.get("dont_list"), list):
            raise ValueError("manual.do_list / manual.dont_list 必须是数组")
        if not isinstance(manual.get("recharge"), str) or not manual.get("recharge"):
            raise ValueError("manual.recharge 必须是非空字符串")
        if not isinstance(war.get("title"), str) or not isinstance(war.get("description"), str):
            raise ValueError("war.title / war.description 必须是字符串")

        fun_data = fun_data_obj
    except Exception as e:
        fun_data = get_fallback_data(str(e))

    return templates.TemplateResponse(
        request,
        "partials/analysis_content.html",
        {
            "fun_data": fun_data,
            "conflict_pair": f"{conflict_pair[0]} vs {conflict_pair[1]}",
            "war_left_pole": core["war_left_pole"],
            "war_left_percent": int(core["war_left_percent"]),
            "war_right_pole": core["war_right_pole"],
            "war_right_percent": int(core["war_right_percent"]),
        },
    )


@router.get("/result/ai_stream/{share_token}")
async def ai_stream(request: Request, share_token: str, db: Session = Depends(get_db)):
    # 最终整合版：灵魂侧写提示词 + 累加流式输出 + 防重连
    async def event_generator():
        client = None

        # 1) 防无限重连：断开后等待 24 小时再重连
        yield "retry: 86400000\n\n"

        try:
            def local_app_secret() -> str:
                # 兼容历史命名：优先 MBTI_APP_SECRET，其次 APP_SECRET
                return os.getenv("MBTI_APP_SECRET") or os.getenv("APP_SECRET", "secret")

            def local_hash_token(token: str, secret: str) -> str:
                # 与 app.services.tokens.hash_token 一致：sha256(secret + token)
                digest = hashlib.sha256()
                digest.update(secret.encode("utf-8"))
                digest.update(token.encode("utf-8"))
                return digest.hexdigest()

            yield "data: [阶段1] 连接已建立，开始验证...\n\n"

            secret = local_app_secret()
            token_hash = local_hash_token(share_token, secret)
            test_row = (
                db.query(Test)
                .options(joinedload(Test.answers).joinedload(Answer.question))
                .filter(Test.share_token_hash == token_hash)
                .first()
            )
            if not test_row:
                yield "data: <span class='text-red-500'>⚠️ 链接已失效，无法获取测试记录。</span>\n\n"
                return

            result = test_row.result_json or {}
            type_code = result.get("type", "Unknown")

            base_url = os.getenv("MBTI_AI_BASE_URL", "https://api.siliconflow.cn/v1")
            api_key = os.getenv("MBTI_AI_API_KEY")
            model = os.getenv("MBTI_AI_MODEL", "deepseek-ai/DeepSeek-V3.2")

            if not api_key:
                yield "data: ⚠️ 系统未配置 AI API Key，请联系管理员。\n\n"
                return

            if AsyncOpenAI is None:
                yield "data: <span class='text-red-500'>❌ 系统错误: OpenAI SDK 不可用</span>\n\n"
                return

            yield f"data: [阶段2] 准备请求 AI（{html.escape(str(model))}）...\n\n"
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)

            # --- 构建“全息画像”数据 ---
            dims: dict[str, dict] = dict(result.get("dimensions") or {})
            boundary_notes = list(result.get("boundary_notes") or [])

            def _dimensions_str() -> str:
                parts: list[str] = []
                for dim in ["EI", "SN", "TF", "JP"]:
                    info = dims.get(dim) or {}
                    fp = info.get("first_percent")
                    sp = info.get("second_percent")
                    first = info.get("first_pole")
                    second = info.get("second_pole")
                    if first and second and fp is not None and sp is not None:
                        parts.append(f"{first}:{int(fp)}% / {second}:{int(sp)}%")
                return ", ".join(parts) if parts else "暂无维度百分比数据"

            def _extremes_str() -> str:
                items: list[str] = []
                for idx, a in enumerate(list(getattr(test_row, "answers", []) or []), start=1):
                    try:
                        v = int(getattr(a, "value", 0))
                    except Exception:
                        continue
                    if v not in (1, 5):
                        continue
                    q = getattr(a, "question", None)
                    if not q:
                        continue
                    text = str(getattr(q, "text", "") or "").strip()
                    dim = str(getattr(q, "dimension", "") or "").strip()
                    agree = str(getattr(q, "agree_pole", "") or "").strip()
                    if len(dim) == 2 and len(agree) == 1:
                        opposite = dim[1] if dim[0] == agree else (dim[0] if dim[1] == agree else "?")
                    else:
                        opposite = "?"
                    pole = agree if v == 5 else opposite
                    snippet = text[:48] + ("…" if len(text) > 48 else "")
                    items.append(f"{idx}. [{dim}/{pole}向] {v}分：{snippet}")
                    if len(items) >= 6:
                        break
                return "；".join(items) if items else "未发现明显的极值作答（1分/5分）"

            answers = list(getattr(test_row, "answers", []) or [])
            insight_list = build_report_context(
                type_code,
                dims,
                boundary_notes=boundary_notes,
                answers=answers,
            ).get("insights", [])
            insight_list = [str(x).strip() for x in (insight_list or []) if str(x).strip()]

            tags: list[str] = []
            for dim, info in (dims or {}).items():
                try:
                    gap = int(info.get("gap_percent"))
                except Exception:
                    continue
                if gap < 20:
                    tags.append(f"{dim}均衡")
                elif gap > 60:
                    tags.append(f"{dim}极致")
            if any(getattr(a, "value", None) == 5 for a in answers):
                tags.append("立场坚定")
            if any("尽管你整体偏向" in s for s in insight_list):
                tags.append("反差发力")
            tags = tags[:6]

            dimensions_str = _dimensions_str()
            extreme_traits = _extremes_str()
            dynamic_insights = "；".join(insight_list[:3]) if insight_list else "暂无动态洞察"
            dynamic_tags = "动态标签：" + (", ".join([f"[{t}]" for t in tags]) if tags else "[稳定作答]")

            user_profile_context = f"""
用户MBTI类型：{type_code}
【维度数据】：{dimensions_str}
【极值特质】：{extreme_traits}
【行为标签】：{dynamic_tags}
【动态洞察】：{dynamic_insights}
请综合上述数据，忽略刻板印象，还原一个鲜活的人。
""".strip()

            # --- 提示词升级：方案 A「深渊凝视者」(格式对齐 + 重点加粗) ---
            system_prompt = f"""
你是一位洞察人性幽暗与光辉的心理学大师，正在使用 DeepSeek-V3.2 模型进行深度侧写。
{user_profile_context}

【指令】
1. 你的语言要像手术刀一样精准，剥开用户的社会面具，直指内心深处的矛盾与渴望。
2. 结合给出的维度数据和极值特质进行推理，不要只罗列优缺点。
3. 严禁输出任何开场白，严禁使用代码块。
4. 【排版严格要求】：
   - 所有标题（###）必须**顶格书写**，严禁在前面加空格，确保左对齐。
   - **关键格式**：标题的井号与文字之间**只能有一个空格**（例如 `### 标题`），严禁出现两个空格或全角空格。
   - 请务必使用 Markdown 加粗语法（如 **关键特质**）高亮文中那些直击灵魂、反直觉或最具冲击力的短句，每段至少包含 1-2 处高亮。

【输出模版】
### ️ 灵魂底色
(用一个带有灰度色彩的意象，如“**暴风雨中的灯塔**”，描述该人格最底层的核心驱动力。约 100 字)

###  镜像与阴影
(指出用户常常欺骗自己的一点，以及外界误解最深的一点。用“世人皆以为...，殊不知...”的句式。约 150 字)

###  给伤口的诗
(针对该人格最容易受挫的软肋，写一段治愈且充满力量的短句。约 80 字)
""".strip()

            user_prompt = f"我的 MBTI 类型是：{type_code}。请开始你的深度解读。"

            stream = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                timeout=60.0,
            )

            yield "data: [阶段3] AI 已接通，开始输出...\n\n"

            async for chunk in stream:
                try:
                    choices = getattr(chunk, "choices", None)
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    if not delta:
                        continue
                    content = getattr(delta, "content", None)
                    if not content:
                        continue
                    safe_payload = json.dumps(str(content))
                    yield f"data: {safe_payload}\n\n"
                except Exception as loop_err:
                    # 忽略单个 token 的异常，避免流整体中断
                    print(f"AI stream chunk error: {loop_err}")
                    continue

        except Exception as e:
            err_msg = str(e)
            print(f"AI Stream Error: {traceback.format_exc()}")
            yield f"data: {json.dumps('❌ 分析中断: ' + err_msg)}\n\n"
        finally:
            if client:
                try:
                    await client.close()
                except Exception:
                    pass

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
