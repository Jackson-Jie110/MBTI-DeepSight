from __future__ import annotations

from datetime import datetime, timezone
import html
import hashlib
import json
import os
import traceback
from pathlib import Path

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

    # Tie-breaker questions pool (fallback to all active questions if no dedicated pool exists).
    tie_rows = (
        db.query(Question)
        .filter(Question.is_active.is_(True), Question.source == "tie_breaker")
        .all()
    )
    if not tie_rows:
        tie_rows = db.query(Question).filter(Question.is_active.is_(True)).all()

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

    item_ids = {
        int(qid)
        for (qid,) in db.query(TestItem.question_id).filter(TestItem.test_id == test_row.id).all()
    }
    if not item_ids:
        raise HTTPException(status_code=400, detail="该测试没有题目")

    to_upsert: list[tuple[int, int]] = []
    if isinstance(raw_answers, dict):
        for k, v in raw_answers.items():
            try:
                qid = int(k)
                val = int(v)
            except Exception:
                continue
            to_upsert.append((qid, val))
    elif isinstance(raw_answers, list):
        for it in raw_answers:
            if not isinstance(it, dict):
                continue
            try:
                qid = int(it.get("question_id"))
                val = int(it.get("value"))
            except Exception:
                continue
            to_upsert.append((qid, val))
    else:
        raise HTTPException(status_code=400, detail="answers 格式不合法")

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

        # Detect tie-break need and append extra question on server side (client renders instantly from preloaded pool).
        items, questions = _load_test_questions(db, test_row.id)
        answer_map = _load_answers(db, test_row.id)
        scoring = score_all(
            [{"id": q.id, "dimension": q.dimension, "agree_pole": q.agree_pole} for q in questions],
            answer_map,
        )
        dims = scoring["dimensions"]

        extra_count = sum(1 for it in items if it.is_extra)
        if extra_count < int(test_row.extra_max):
            near = [
                d
                for d in ["EI", "SN", "TF", "JP"]
                if is_near_boundary(int(dims[d]["first_percent"]), threshold_gap_percent=10)
            ]
            if near:
                used_ids = {int(it.question_id) for it in items}
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
                    if not q:
                        continue
                    new_pos = len(items) + 1
                    db.add(TestItem(test_id=test_row.id, position=new_pos, question_id=q.id, is_extra=True))
                    db.commit()

                    pair = f"{dim[0]}-{dim[1]}"
                    return JSONResponse(
                        {
                            "status": "tie_break",
                            "dimension_pair": pair,
                            "questions": [
                                {
                                    "id": int(q.id),
                                    "text": q.text,
                                    "dimension": q.dimension,
                                    "agree_pole": q.agree_pole,
                                    "position": int(new_pos),
                                    "is_extra": True,
                                }
                            ],
                        }
                    )

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
            "report": report,
            "share_url": str(request.base_url)[:-1] + f"/result/{share_token}",
            "share_token": share_token,
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
