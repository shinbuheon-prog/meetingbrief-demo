"""
MeetingBrief AI — main.py
FastAPI backend: Google OAuth + Claude + Tavily
"""
import os
import re
import json
import datetime
import requests
from pathlib import Path
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware

# ── 設定 ──────────────────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8000/oauth/callback")
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY       = os.environ.get("TAVILY_API_KEY", "")
SECRET_KEY           = os.environ.get("SECRET_KEY", "meeting-brief-secret-2025")
CLAUDE_MODEL         = "claude-haiku-4-5-20251001"

BASE_DIR = Path(__file__).parent

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# ── グローバル変数 ────────────────────────────────────────────────────────────
sessions: dict = {}
meeting_states: dict = {}   # {sid: {mid: {status, transcript, minutes}}}
briefing_cache: dict = {}   # {sid: {mid: {company, result, created_at, mode, language}}}

# ── 会社名パターン ─────────────────────────────────────────────────────────────
COMPANY_PATTERNS = [
    r"株式会社(.{1,15}?)(?:\s|$|との?|様)",
    r"(.{1,15}?)株式会社",
    r"(.{1,15}?)(?:Corp|Inc|Ltd|LLC)\.?",
    r"(?:Corp|Inc|Ltd|LLC)\.?\s+(.{1,15})",
    r"有限会社(.{1,15}?)(?:\s|$|との?|様)",
    r"(.{1,15}?)(?:有限会社|合同会社|LLC)",
    r"(?i)(.{2,20}?)\s+(?:株式?|co\.?|corp\.?|inc\.?|ltd\.?)",
]

def extract_company(text: str) -> str:
    for pat in COMPANY_PATTERNS:
        m = re.search(pat, text)
        if m:
            name = m.group(1).strip()
            if len(name) >= 2:
                return name
    words = text.split()
    return words[0] if words else text

# ── ドメイン → セクション マッピング ─────────────────────────────────────────
DOMAIN_SECTION = {
    "reuters.com": "recent_news", "nikkei.com": "recent_news",
    "prtimes.jp": "recent_news", "bloomberg.com": "recent_news",
    "techcrunch.com": "recent_news",
    "twitter.com": "sns_official", "x.com": "sns_official",
    "linkedin.com": "sns_official", "facebook.com": "sns_official",
    "instagram.com": "sns_official", "youtube.com": "sns_official",
    # テックブログ・採用
    "qiita.com": "business_insight", "zenn.dev": "business_insight",
    "wantedly.com": "business_insight", "openwork.jp": "business_insight",
    "glassdoor.com": "business_insight", "connpass.com": "business_insight",
    "speakerdeck.com": "business_insight", "slideshare.net": "business_insight",
}

def _classify_url(url: str) -> str:
    u = url.lower()
    for domain, section in DOMAIN_SECTION.items():
        if domain in u:
            return section
    if any(k in u for k in ["/ir", "/investor", "/kessan", "/disclosure"]):
        return "financial_info"
    if any(k in u for k in ["/news", "/press", "/release", "/pr"]):
        return "recent_news"
    return "company_overview"

def _dedup(items: list) -> list:
    seen, result = set(), []
    for it in items:
        url = it.get("url", "")
        if url and url not in seen:
            seen.add(url)
            result.append(it)
    return result

# ── fetch_all_sources ─────────────────────────────────────────────────────────
def fetch_all_sources(company: str) -> dict:
    import xml.etree.ElementTree as ET
    from urllib.parse import quote

    bundle: dict = {
        "company_overview": [], "financial_info": [], "market_info": [],
        "competitive_positioning": [], "recent_news": [], "sns_official": [],
        "business_insight": [], "wiki_text": "", "wiki_url": "",
    }

    # ── 1. Wikipedia ──────────────────────────────────────────────────────────
    for lang in ("ja", "en"):
        try:
            api_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(company)}"
            r = requests.get(api_url, timeout=6, headers={"User-Agent": "MeetingBriefAI/1.0"})
            if r.status_code == 200:
                data = r.json()
                extract = data.get("extract", "")
                if len(extract) > 80:
                    bundle["wiki_text"] = f"[Wikipedia/{lang}] {extract[:800]}"
                    bundle["wiki_url"]  = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                    bundle["company_overview"].append({
                        "title": data.get("title", company) + " - Wikipedia",
                        "url": bundle["wiki_url"], "content": extract[:300],
                    })
                    break
        except Exception:
            pass

    # ── 2. Google News RSS ────────────────────────────────────────────────────
    try:
        q = quote(company)
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
        r = requests.get(rss_url, timeout=8, headers={"User-Agent": "MeetingBriefAI/1.0"})
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:6]:
                title = item.findtext("title", "").split(" - ")[0].strip()
                link  = item.findtext("link", "")
                if title and link:
                    bundle["recent_news"].append({"title": title, "url": link, "content": ""})
    except Exception:
        pass

    # ── 3. PR Times RSS ───────────────────────────────────────────────────────
    try:
        q = quote(company)
        pr_url = f"https://prtimes.jp/rss/company_name/search/?name={q}"
        r = requests.get(pr_url, timeout=6, headers={"User-Agent": "MeetingBriefAI/1.0"})
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:3]:
                title = item.findtext("title", "").strip()
                link  = item.findtext("link", "")
                desc  = item.findtext("description", "")[:150]
                if title and link:
                    bundle["recent_news"].append({"title": f"[PR] {title}", "url": link, "content": desc})
    except Exception:
        pass

    # ── 3b. Qiita API ─────────────────────────────────────────────────────────
    try:
        from urllib.parse import quote as _q
        qiita_url = f"https://qiita.com/api/v2/items?query={_q(company)}&per_page=5"
        r = requests.get(qiita_url, timeout=6, headers={"User-Agent": "MeetingBriefAI/1.0"})
        if r.status_code == 200:
            for item in r.json()[:5]:
                title  = item.get("title", "")
                url_q  = item.get("url", "")
                body   = item.get("body", "")[:200]
                tags   = ", ".join(t.get("name", "") for t in item.get("tags", [])[:4])
                if title and url_q:
                    bundle["business_insight"].append({
                        "title": f"[Qiita] {title}",
                        "url": url_q,
                        "content": f"Tags: {tags}. {body[:150]}",
                    })
    except Exception:
        pass

    # ── 4. Tavily (3 calls) ───────────────────────────────────────────────────
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            from tavily import TavilyClient
            tc = TavilyClient(api_key=tavily_key)

            # Call 1: 企業概要・財務・競合・市場 (advanced, 10件)
            resp1 = tc.search(
                query=f"{company} 企業情報 IR 決算 業績 アナリスト評価 競合 市場シェア",
                search_depth="advanced",
                max_results=10,
            )
            for r in resp1.get("results", []):
                section = _classify_url(r.get("url", ""))
                item = {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")[:250]}
                bundle[section].append(item)
                if section == "financial_info":
                    bundle["competitive_positioning"].append(item)

            # Call 2: ニュース・SNS (advanced, 8件)
            resp2 = tc.search(
                query=f"{company} 最新ニュース プレスリリース 公式SNS 投資家 スタートアップ 資金調達",
                search_depth="advanced",
                max_results=8,
            )
            for r in resp2.get("results", []):
                section = _classify_url(r.get("url", ""))
                item = {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")[:200]}
                bundle[section].append(item)

            # Call 3: ビジネスインサイト (basic, 8件)
            resp3 = tc.search(
                query=f"{company} 採用課題 ペインポイント 技術ブログ 事業戦略 カンファレンス",
                search_depth="basic",
                max_results=8,
            )
            for r in resp3.get("results", []):
                section = _classify_url(r.get("url", ""))
                item = {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")[:250]}
                url_r = r.get("url", "").lower()
                if section == "business_insight" or any(kw in url_r for kw in ["blog", "tech", "career", "recruit", "event", "jobs"]):
                    bundle["business_insight"].append(item)
                else:
                    bundle[section].append(item)

        except Exception as e:
            print(f"[Tavily error] {e}")

    # ── 5. 重複排除・上限設定 ─────────────────────────────────────────────────
    limits = {
        "company_overview": 3, "financial_info": 4, "market_info": 3,
        "competitive_positioning": 3, "recent_news": 5, "sns_official": 2,
        "business_insight": 5,
    }
    for key, limit in limits.items():
        bundle[key] = _dedup(bundle[key])[:limit]

    return bundle

# ── call_claude ───────────────────────────────────────────────────────────────
def call_claude(company: str, mode: str = "standard", language: str = "ja") -> dict:
    import anthropic

    sources = fetch_all_sources(company)

    def _fmt_refs(items, max_n=3):
        lines_out = []
        for it in items[:max_n]:
            t, u, c = it.get("title", ""), it.get("url", ""), it.get("content", "")
            lines_out.append(f"  - [{t}]({u})")
            if c:
                lines_out.append(f"    {c[:120]}")
        return "\n".join(lines_out)

    context_block = ""
    if sources["wiki_text"]:
        context_block += f"[会社概要参考]\n{sources['wiki_text'][:600]}\n\n"
    if sources["financial_info"]:
        context_block += f"[財務・IR・アナリスト参考]\n{_fmt_refs(sources['financial_info'])}\n\n"
    if sources["market_info"]:
        context_block += f"[市場・業界参考]\n{_fmt_refs(sources['market_info'])}\n\n"
    if sources["competitive_positioning"]:
        context_block += f"[競合・ポジショニング参考]\n{_fmt_refs(sources['competitive_positioning'])}\n\n"
    if sources["recent_news"]:
        context_block += f"[最新ニュース・プレスリリース参考]\n{_fmt_refs(sources['recent_news'], 5)}\n\n"
    if sources["sns_official"]:
        context_block += f"[公式SNS・コーポレートサイト参考]\n{_fmt_refs(sources['sns_official'])}\n\n"
    if sources["business_insight"]:
        context_block += f"[ビジネスインサイト・採用・ブログ参考]\n{_fmt_refs(sources['business_insight'], 4)}\n\n"

    all_refs = {
        "company_overview":        sources["company_overview"],
        "financial_info":          sources["financial_info"],
        "market_info":             sources["market_info"],
        "competitive_positioning": sources["competitive_positioning"],
        "recent_news":             sources["recent_news"],
        "business_insight":        sources["business_insight"],
    }

    lang_note = "英語で出力すること。" if language == "en" else "日本語で出力すること。"
    mode_note = ""
    if mode == "short":
        mode_note = "各セクション2-3文の簡潔な要約にすること。"
    elif mode == "detail":
        mode_note = "各セクション詳細かつ具体的に記述すること。"

    system_text = (
        f"あなたは商談前ブリーフィング生成の専門AIです。{lang_note}{mode_note}"
        "以下のルールを厳守して generate_briefing ツールを必ず呼び出してください: "
        "1.提供された参考情報を最大限活用し、架空の情報は記載しない。"
        "2.company_overviewは企業の事業・ミッション・規模を簡潔に。"
        "3.financial_infoは業績・財務指標・投資家情報を具体的に。"
        "4.market_infoは市場規模・成長率・トレンドを。"
        "5.competitive_positioningは競合比較・差別化ポイントを。"
        "6.recent_newsは最新ニュース・プレスリリース・重要イベントを。"
        "7.参考情報で提供されたURLを引用した箇所に [参考: タイトル](url) 形式でインラインリンクを付記する。"
        "SNS公式情報がある場合は各セクションに反映する。"
        "8.business_insightでは採用情報・技術ブログ・カンファレンスから"
        "事業課題(ペインポイント)・技術スタック・主要関心事を分析し商談切り口を提示する。"
    )

    tool_schema = {
        "name": "generate_briefing",
        "description": "商談前ブリーフィングを8セクションのJSON形式で生成する",
        "input_schema": {
            "type": "object",
            "properties": {
                "company_overview":        {"type": "string"},
                "financial_info":          {"type": "string"},
                "market_info":             {"type": "string"},
                "competitive_positioning": {"type": "string"},
                "recent_news":             {"type": "string"},
                "proposal_ideas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "idea":     {"type": "string"},
                            "detail":   {"type": "string"},
                            "priority": {"type": "string"},
                        },
                        "required": ["idea", "detail", "priority"],
                    },
                },
                "icebreakers":      {"type": "array", "items": {"type": "string"}},
                "business_insight": {"type": "string"},
            },
            "required": [
                "company_overview", "financial_info", "market_info",
                "competitive_positioning", "recent_news",
                "proposal_ideas", "icebreakers", "business_insight",
            ],
        },
    }

    user_message = (
        f"対象企業: {company}\n\n"
        f"参考情報:\n{context_block}\n\n"
        f"上記情報を参考に、{company}との商談前ブリーフィングを生成してください。"
    )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=[
            {"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}
        ],
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": "generate_briefing"},
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "generate_briefing":
            result = dict(block.input)
            result["references"] = all_refs
            return result

    raise ValueError("generate_briefing tool not called")

# ── call_claude_minutes ───────────────────────────────────────────────────────
def call_claude_minutes(company: str, transcript: str) -> str:
    import anthropic

    system_text = (
        "あなたは商談議事録を作成する専門AIです。"
        "提供された書き起こしから構造化された議事録を日本語で作成してください。"
        "フォーマット: 1.参加者, 2.議題, 3.主要な議論, 4.決定事項, 5.次のアクション"
    )
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=system_text,
        messages=[{
            "role": "user",
            "content": f"企業: {company}\n\n書き起こし:\n{transcript}\n\n議事録を作成してください。"
        }],
    )
    return response.content[0].text if response.content else ""

# ── OAuth ─────────────────────────────────────────────────────────────────────
@app.get("/login/google")
async def login_google():
    scope = "openid email profile https://www.googleapis.com/auth/calendar.readonly"
    url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        "&response_type=code"
        f"&scope={requests.utils.quote(scope)}"
        "&access_type=offline"
        "&prompt=consent"
    )
    return RedirectResponse(url)

@app.get("/oauth/callback")
async def oauth_callback(request: Request, code: str = ""):
    if not code:
        return HTMLResponse("Error: no code", status_code=400)

    token_resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
    )
    tokens = token_resp.json()
    access_token = tokens.get("access_token", "")
    if not access_token:
        return HTMLResponse(f"Error getting token: {tokens}", status_code=400)

    user_resp = requests.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    user_info = user_resp.json()
    sid = user_info.get("sub", "")
    if not sid:
        return HTMLResponse("Error: no user id", status_code=400)

    sessions[sid] = {
        "email":        user_info.get("email", ""),
        "name":         user_info.get("name", ""),
        "picture":      user_info.get("picture", ""),
        "access_token": access_token,
    }
    meeting_states.setdefault(sid, {})
    briefing_cache.setdefault(sid, {})

    response = RedirectResponse(url="/dashboard")
    response.set_cookie("sid", sid, httponly=True, samesite="lax")
    return response

@app.get("/logout")
async def logout(request: Request):
    sid = request.cookies.get("sid", "")
    if sid in sessions:
        del sessions[sid]
    response = RedirectResponse(url="/")
    response.delete_cookie("sid")
    return response

# ── landing ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = BASE_DIR / "landing.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>MeetingBrief AI</h1><a href='/login/google'>Login</a>")

# ── dashboard ─────────────────────────────────────────────────────────────────
def build_dashboard(sid: str) -> str:
    user = sessions.get(sid, {})
    name = user.get("name", "User")
    email = user.get("email", "")
    picture = user.get("picture", "")
    access_token = user.get("access_token", "")

    # カレンダー取得
    meetings = []
    if access_token:
        try:
            now = datetime.datetime.utcnow().isoformat() + "Z"
            end = (datetime.datetime.utcnow() + datetime.timedelta(days=7)).isoformat() + "Z"
            cal_resp = requests.get(
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "timeMin": now, "timeMax": end,
                    "singleEvents": "true", "orderBy": "startTime",
                    "maxResults": "10",
                },
                timeout=8,
            )
            if cal_resp.status_code == 200:
                for ev in cal_resp.json().get("items", []):
                    start = ev.get("start", {})
                    dt = start.get("dateTime", start.get("date", ""))
                    meetings.append({
                        "id":       ev.get("id", ""),
                        "summary":  ev.get("summary", "(無題)"),
                        "start":    dt,
                        "attendees": [a.get("email", "") for a in ev.get("attendees", [])],
                    })
        except Exception as e:
            print(f"[Calendar error] {e}")

    # ブリーフィング履歴
    history = briefing_cache.get(sid, {})

    meetings_json = json.dumps(meetings, ensure_ascii=False)
    history_json  = json.dumps(history,  ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MeetingBrief AI \u2014 Dashboard</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans JP',sans-serif;
  background:#0f172a;color:#e2e8f0;min-height:100vh}}
.header{{background:rgba(15,23,42,0.95);border-bottom:1px solid #1e3a5f;
  padding:12px 24px;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:100}}
.header .logo{{font-size:1.2rem;font-weight:800;color:#60a5fa}}
.header .user{{margin-left:auto;display:flex;align-items:center;gap:10px;font-size:.9rem}}
.header img{{width:32px;height:32px;border-radius:50%}}
.header a{{color:#94a3b8;text-decoration:none;font-size:.85rem}}
.header a:hover{{color:#60a5fa}}

/* Flow bar */
.flow-bar{{background:#0f2441;border-bottom:1px solid #1e3a5f;
  padding:10px 24px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.flow-step{{padding:4px 12px;border-radius:999px;font-size:.78rem;font-weight:600;
  background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.3);color:#93c5fd}}
.flow-step.active{{background:rgba(59,130,246,.35);border-color:#60a5fa;color:#fff}}
.flow-arrow{{color:#475569;font-size:.8rem}}

/* Layout */
.main-wrap{{display:grid;grid-template-columns:1fr 172px;gap:0;min-height:calc(100vh - 110px)}}
.content{{padding:20px 24px;max-width:100%}}
.sidebar{{background:#0a1929;border-left:1px solid #1e3a5f;padding:16px 10px}}

/* Sidebar keywords */
.sidebar-title{{font-size:.72rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px}}
.kw-list{{display:flex;flex-direction:column;gap:6px}}
.kw{{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.2);
  border-radius:8px;padding:6px 8px;font-size:.72rem;color:#93c5fd;text-align:center;line-height:1.3}}

/* Tutorial */
.tutorial-wrap{{margin-bottom:16px}}
.tutorial-toggle{{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.25);
  border-radius:10px;padding:10px 14px;cursor:pointer;display:flex;align-items:center;
  gap:8px;font-size:.85rem;color:#93c5fd;width:100%}}
.tutorial-body{{display:none;background:rgba(15,23,42,.8);border:1px solid #1e3a5f;
  border-radius:10px;padding:14px;margin-top:6px}}
.tutorial-body.open{{display:block}}
.tut-steps{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}}
.tut-step{{background:rgba(59,130,246,.08);border-radius:8px;padding:10px}}
.tut-step h4{{font-size:.8rem;color:#60a5fa;margin-bottom:6px}}
.tut-step p{{font-size:.75rem;color:#94a3b8;line-height:1.5}}
.cal-fmt{{background:#0a1929;border-radius:6px;padding:8px 10px;font-size:.73rem;
  color:#64748b;font-family:monospace;white-space:pre-wrap}}

/* Page title */
.page-title{{font-size:1.25rem;font-weight:700;color:#e2e8f0;margin-bottom:16px}}
.page-title span{{color:#60a5fa}}

/* Meeting card */
.meeting-card{{background:rgba(30,58,95,.25);border:1px solid #1e3a5f;border-radius:14px;
  padding:16px;margin-bottom:14px}}
.card-header{{display:flex;align-items:flex-start;gap:10px;margin-bottom:12px}}
.card-time{{font-size:.78rem;color:#60a5fa;font-weight:600;white-space:nowrap}}
.card-title{{font-size:1rem;font-weight:600;color:#e2e8f0;flex:1}}
.card-attendees{{font-size:.75rem;color:#64748b;margin-top:2px}}
.card-controls{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px}}
select{{background:#0a1929;border:1px solid #1e3a5f;color:#e2e8f0;
  padding:4px 8px;border-radius:6px;font-size:.78rem}}
.btn{{padding:6px 14px;border-radius:8px;border:none;cursor:pointer;
  font-size:.8rem;font-weight:600;transition:all .2s}}
.btn-primary{{background:#2563eb;color:#fff}}
.btn-primary:hover{{background:#1d4ed8}}
.btn-primary:disabled{{background:#1e3a5f;color:#64748b;cursor:not-allowed}}
.btn-sm{{padding:4px 10px;font-size:.75rem}}
.progress-wrap{{height:4px;background:#1e3a5f;border-radius:2px;margin-bottom:10px;overflow:hidden}}
.progress-bar{{height:100%;background:linear-gradient(90deg,#2563eb,#60a5fa);
  width:0%;transition:width .4s;border-radius:2px}}
.status-msg{{font-size:.78rem;color:#94a3b8;margin-bottom:8px}}
.result-wrap{{display:none;margin-top:12px}}
.result-wrap.show{{display:block}}
.sections-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:12px}}
.section-card{{background:rgba(15,23,42,.6);border:1px solid #1e3a5f;border-radius:8px;padding:10px}}
.section-card h4{{font-size:.78rem;color:#60a5fa;margin-bottom:6px;font-weight:700}}
.section-card p{{font-size:.75rem;color:#94a3b8;line-height:1.55;white-space:pre-wrap}}
.section-card ul{{padding-left:14px;font-size:.75rem;color:#94a3b8;line-height:1.55}}
.proposals-list{{display:flex;flex-direction:column;gap:6px}}
.proposal-item{{background:rgba(37,99,235,.1);border:1px solid rgba(37,99,235,.25);
  border-radius:8px;padding:8px 10px}}
.proposal-item .idea{{font-size:.8rem;font-weight:600;color:#60a5fa}}
.proposal-item .detail{{font-size:.75rem;color:#94a3b8;margin-top:2px}}
.proposal-item .priority{{display:inline-block;font-size:.65rem;padding:1px 6px;
  border-radius:4px;background:rgba(245,158,11,.15);color:#f59e0b;margin-top:4px}}
.refs-wrap{{margin-top:10px}}
.refs-title{{font-size:.72rem;color:#475569;margin-bottom:4px}}
.refs-links{{display:flex;flex-wrap:wrap;gap:4px}}
.refs-links a{{font-size:.7rem;color:#60a5fa;text-decoration:none;
  background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.2);
  padding:2px 6px;border-radius:4px}}
.refs-links a:hover{{background:rgba(59,130,246,.2)}}
.inline-ref{{color:#2563eb;text-decoration:none;border-bottom:1px dotted #93c5fd;font-size:.8em;vertical-align:super}}
.inline-ref:hover{{text-decoration:underline}}

/* Post-meeting actions */
.post-actions{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;padding-top:10px;
  border-top:1px solid #1e3a5f}}
.post-actions .btn{{background:rgba(30,58,95,.5);color:#93c5fd;border:1px solid #1e3a5f}}
.post-actions .btn:hover{{background:rgba(37,99,235,.2)}}
.score-badge{{font-size:.75rem;color:#f59e0b}}

/* Transcript / Minutes */
.transcript-area{{width:100%;background:#0a1929;border:1px solid #1e3a5f;color:#e2e8f0;
  border-radius:8px;padding:8px;font-size:.8rem;resize:vertical;font-family:inherit}}
.minutes-out{{background:rgba(15,23,42,.8);border:1px solid #1e3a5f;border-radius:8px;
  padding:10px;font-size:.78rem;color:#94a3b8;white-space:pre-wrap;line-height:1.6}}

/* History */
#history-section{{margin-top:24px;display:none}}
#history-section.show{{display:block}}
.history-title{{font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:10px}}
.history-list{{display:flex;flex-direction:column;gap:8px}}
.history-item{{background:rgba(30,58,95,.2);border:1px solid #1e3a5f;border-radius:10px;
  padding:10px 14px;display:flex;align-items:center;gap:10px;cursor:pointer}}
.history-item:hover{{background:rgba(37,99,235,.15)}}
.history-company{{font-size:.9rem;font-weight:600;color:#e2e8f0}}
.history-meta{{font-size:.75rem;color:#64748b}}
.history-link{{margin-left:auto;font-size:.75rem;color:#60a5fa;text-decoration:none}}

/* Responsive */
@media(max-width:700px){{
  .main-wrap{{grid-template-columns:1fr}}
  .sidebar{{display:none}}
  .sections-grid{{grid-template-columns:1fr}}
  .tut-steps{{grid-template-columns:1fr}}
}}
</style>
</head>
<body>
<div class="header">
  <div class="logo">\u26a1 MeetingBrief AI</div>
  <div class="user">
    {'<img src="' + picture + '" alt="">' if picture else ''}
    <span>{name}</span>
    <a href="/logout">\u30ed\u30b0\u30a2\u30a6\u30c8</a>
  </div>
</div>

<div class="flow-bar">
  <span class="flow-step active">Pre-Meeting</span>
  <span class="flow-arrow">\u2192</span>
  <span class="flow-step">In-Meeting</span>
  <span class="flow-arrow">\u2192</span>
  <span class="flow-step">Post-Meeting</span>
  <span class="flow-arrow">\u2192</span>
  <span class="flow-step">Phase 2 \u8a2d\u8a08\u5b8c\u4e86</span>
</div>

<div class="main-wrap">
<div class="content">

<!-- Tutorial -->
<div class="tutorial-wrap">
  <button class="tutorial-toggle" onclick="toggleTutorial()"
    id="tut-toggle">\u2753 \u4f7f\u3044\u65b9\u30ac\u30a4\u30c9 (\u30af\u30ea\u30c3\u30af\u3067\u5c55\u958b)</button>
  <div class="tutorial-body" id="tut-body">
    <div class="tut-steps">
      <div class="tut-step">
        <h4>1\ufe0f\u20e3 Pre-Meeting</h4>
        <p>\u30ab\u30ec\u30f3\u30c0\u30fc\u306e\u4e88\u5b9a\u304b\u3089\u4f01\u696d\u540d\u3092\u691c\u51fa\u3057\u3001Claude + Tavily 3calls \u3067\u30d6\u30ea\u30fc\u30d5\u30a3\u30f3\u30b0\u3092\u81ea\u52d5\u751f\u6210\u3057\u307e\u3059\u3002</p>
      </div>
      <div class="tut-step">
        <h4>2\ufe0f\u20e3 In-Meeting</h4>
        <p>\u5546\u8ac7\u4e2d\u306b\u66f8\u304d\u8d77\u3053\u3057\u3092\u5165\u529b\u3002\u30df\u30fc\u30c6\u30a3\u30f3\u30b0\u7d42\u4e86\u5f8c\u306b\u300c\u8b70\u4e8b\u9332\u751f\u6210\u300d\u3067\u5373\u5ea7\u306b\u8b70\u4e8b\u9332\u3092\u4f5c\u6210\u3002</p>
      </div>
      <div class="tut-step">
        <h4>3\ufe0f\u20e3 Post-Meeting</h4>
        <p>RelationshipScore \u30fb HubSpot \u540c\u671f\u306f Phase 2 \u3067\u5b9f\u88c5\u4e2d\u3002\u73fe\u5728\u306f\u30b9\u30bf\u30d6\u304c\u5229\u7528\u53ef\u80fd\u3067\u3059\u3002</p>
      </div>
    </div>
    <div class="cal-fmt"># \u30ab\u30ec\u30f3\u30c0\u30fc\u767b\u9332\u30d5\u30a9\u30fc\u30de\u30c3\u30c8
\u4ef6\u540d: [\u4f01\u696d\u540d] \u5546\u8ac7
\u5834\u6240: \u30aa\u30f3\u30e9\u30a4\u30f3 (Zoom/Teams)
\u53c2\u52a0\u8005: \u62c5\u5f53\u8005\u30e1\u30fc\u30eb\u30a2\u30c9\u30ec\u30b9</div>
  </div>
</div>

<div class="page-title">
  \u672c\u65e5\u306e\u5546\u8ac7\u30fb\u76f4\u8fd1\u306e\u4e88\u5b9a
  <span id="cal-count"></span>
</div>

<div id="meetings-container"></div>

<!-- History -->
<div id="history-section">
  <div class="history-title">\u904e\u53bb\u306e\u30d6\u30ea\u30fc\u30d5\u30a3\u30f3\u30b0\u5c65\u6b74</div>
  <div class="history-list" id="history-list"></div>
</div>

</div><!-- .content -->

<!-- Sidebar -->
<div class="sidebar">
  <div class="sidebar-title">Features</div>
  <div class="kw-list">
    <div class="kw">Pre-Meeting</div>
    <div class="kw">Claude tool_use</div>
    <div class="kw">Tavily 3calls</div>
    <div class="kw">Prompt Caching</div>
    <div class="kw">RelationshipScore</div>
    <div class="kw">HubSpot\u540c\u671f</div>
    <div class="kw">Global Ready</div>
  </div>
</div>

</div><!-- .main-wrap -->

<script>
const MEETINGS = {meetings_json};
const HISTORY  = {history_json};
const SECTIONS = [
  ['1','\ud83c\udfe2','\u4f1a\u793e\u6982\u8981','company_overview'],
  ['2','\ud83d\udcb0','\u8ca1\u52d9\u60c5\u5831','financial_info'],
  ['3','\ud83d\udcc8','\u5e02\u5834\u60c5\u5831','market_info'],
  ['4','\u2694\ufe0f','\u7af6\u5408\u5206\u6790','competitive_positioning'],
  ['5','\ud83d\udcf0','\u6700\u65b0\u30cb\u30e5\u30fc\u30b9','recent_news'],
  ['6','\ud83d\udca1','\u63d0\u6848\u8996\u70b9','proposal_ideas'],
  ['7','\u2615','\u30a2\u30a4\u30b9\u30d6\u30ec\u30a4\u30ad\u30f3\u30b0','icebreakers'],
  ['8','\ud83e\udde0','\u30d3\u30b8\u30cd\u30b9\u30a4\u30f3\u30b5\u30a4\u30c8','business_insight']
];

function esc(s){{return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}}
function renderMd(s){{
  const e=esc(s);
  return e.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener" class="inline-ref">$1</a>');
}}

function toggleTutorial(){{
  const b=document.getElementById('tut-body');
  b.classList.toggle('open');
  document.getElementById('tut-toggle').textContent=b.classList.contains('open')
    ?'\u2753 \u4f7f\u3044\u65b9\u30ac\u30a4\u30c9 (\u9589\u3058\u308b)'
    :'\u2753 \u4f7f\u3044\u65b9\u30ac\u30a4\u30c9 (\u30af\u30ea\u30c3\u30af\u3067\u5c55\u958b)';
}}

function fmtDate(s){{
  if(!s) return '';
  const d=new Date(s);
  if(isNaN(d)) return s;
  return d.toLocaleDateString('ja-JP',{{month:'short',day:'numeric',weekday:'short'}})
    +' '+d.toLocaleTimeString('ja-JP',{{hour:'2-digit',minute:'2-digit'}});
}}

function renderMeetingCard(m,idx){{
  const mid=m.id||'manual-'+idx;
  const company=m.summary||(m.attendees&&m.attendees[0]?m.attendees[0].split('@')[0]:'Unknown');
  return `
<div class="meeting-card" id="card-${{mid}}">
  <div class="card-header">
    <div>
      <div class="card-time">${{fmtDate(m.start)}}</div>
      <div class="card-title">${{esc(m.summary||'(無題)')}}</div>
      <div class="card-attendees">${{m.attendees&&m.attendees.length?'\u53c2\u52a0\u8005: '+esc(m.attendees.slice(0,3).join(', ')):''}}</div>
    </div>
  </div>
  <div class="card-controls">
    <input type="text" id="company-${{mid}}" value="${{esc(company)}}"
      style="background:#0a1929;border:1px solid #1e3a5f;color:#e2e8f0;padding:5px 8px;border-radius:6px;font-size:.8rem;flex:1;min-width:140px"
      placeholder="\u4f01\u696d\u540d\u3092\u5165\u529b...">
    <select id="mode-${{mid}}">
      <option value="standard">\u6a19\u6e96</option>
      <option value="short">\u7c21\u6f54</option>
      <option value="detail">\u8a73\u7d30</option>
    </select>
    <select id="lang-${{mid}}">
      <option value="ja">\u65e5\u672c\u8a9e</option>
      <option value="en">English</option>
    </select>
    <button class="btn btn-primary" id="btn-${{mid}}"
      onclick="generateBriefing('${{mid}}')">\u26a1 \u30d6\u30ea\u30fc\u30d5\u30a3\u30f3\u30b0\u751f\u6210</button>
  </div>
  <div class="progress-wrap"><div class="progress-bar" id="prog-${{mid}}"></div></div>
  <div class="status-msg" id="status-${{mid}}"></div>
  <div class="result-wrap" id="result-${{mid}}"></div>
</div>`;
}}

function renderManualCard(){{
  return `
<div class="meeting-card">
  <div class="card-header">
    <div>
      <div class="card-title">\u624b\u52d5\u3067\u30d6\u30ea\u30fc\u30d5\u30a3\u30f3\u30b0\u751f\u6210</div>
    </div>
  </div>
  <div class="card-controls">
    <input type="text" id="company-manual"
      style="background:#0a1929;border:1px solid #1e3a5f;color:#e2e8f0;padding:5px 8px;border-radius:6px;font-size:.8rem;flex:1;min-width:140px"
      placeholder="\u4f01\u696d\u540d\u3092\u5165\u529b...">
    <select id="mode-manual">
      <option value="standard">\u6a19\u6e96</option>
      <option value="short">\u7c21\u6f54</option>
      <option value="detail">\u8a73\u7d30</option>
    </select>
    <select id="lang-manual">
      <option value="ja">\u65e5\u672c\u8a9e</option>
      <option value="en">English</option>
    </select>
    <button class="btn btn-primary" id="btn-manual"
      onclick="generateBriefing('manual')">\u26a1 \u30d6\u30ea\u30fc\u30d5\u30a3\u30f3\u30b0\u751f\u6210</button>
  </div>
  <div class="progress-wrap"><div class="progress-bar" id="prog-manual"></div></div>
  <div class="status-msg" id="status-manual"></div>
  <div class="result-wrap" id="result-manual"></div>
</div>`;
}}

function renderSectionContent(key, val){{
  if(!val) return '<p style="color:#475569">\u30c7\u30fc\u30bf\u306a\u3057</p>';
  if(key==='proposal_ideas' && Array.isArray(val)){{
    const items=val.map(p=>`
      <div class="proposal-item">
        <div class="idea">${{esc(p.idea||'')}}</div>
        <div class="detail">${{esc(p.detail||'')}}</div>
        <div class="priority">\u512a\u5148\u5ea6: ${{esc(p.priority||'')}}</div>
      </div>`).join('');
    return `<div class="proposals-list">${{items}}</div>`;
  }}
  if(key==='icebreakers' && Array.isArray(val)){{
    const items=val.map(s=>`<li>${{esc(s)}}</li>`).join('');
    return `<ul>${{items}}</ul>`;
  }}
  const content=`<p style="white-space:pre-line">${{renderMd(String(val||''))}}</p>`;
  return content;
}}

function renderRefs(refs){{
  if(!refs) return '';
  let html='<div class="refs-wrap"><div class="refs-title">\u53c2\u8003\u30bd\u30fc\u30b9</div><div class="refs-links">';
  const allRefs=[];
  Object.values(refs).forEach(arr=>{{if(Array.isArray(arr)) arr.forEach(r=>allRefs.push(r))}});
  const seen=new Set();
  allRefs.forEach(r=>{{
    if(r.url&&!seen.has(r.url)){{
      seen.add(r.url);
      html+=`<a href="${{esc(r.url)}}" target="_blank" rel="noopener">${{esc(r.title||r.url.slice(0,40))}}</a>`;
    }}
  }});
  html+='</div></div>';
  return html;
}}

function renderResult(mid, data){{
  const sections=SECTIONS.map(([num,icon,label,key])=>{{
    return `<div class="section-card">
      <h4>${{icon}} Section ${{num}}: ${{label}}</h4>
      ${{renderSectionContent(key, data[key])}}
    </div>`;
  }}).join('');

  const refsHtml=renderRefs(data.references);

  const postActions=`
<div class="post-actions">
  <span style="font-size:.75rem;color:#475569;align-self:center">Post-Meeting:</span>
  <button class="btn btn-sm" onclick="showTranscript('${{mid}}')">\ud83c\udfa4 \u66f8\u304d\u8d77\u3053\u3057\u5165\u529b</button>
  <button class="btn btn-sm" onclick="showScore('${{mid}}')">\ud83d\udcca RelationshipScore <span class="score-badge">Phase2</span></button>
  <button class="btn btn-sm" onclick="showHubspot('${{mid}}')">\ud83d\udd04 HubSpot\u540c\u671f <span class="score-badge">Phase2</span></button>
  <a href="/briefing/${{mid}}" target="_blank" class="btn btn-sm">\ud83d\udd17 \u8a73\u7d30\u30da\u30fc\u30b8</a>
</div>
<div id="transcript-section-${{mid}}" style="display:none;margin-top:10px">
  <textarea class="transcript-area" id="transcript-${{mid}}" rows="5"
    placeholder="\u5546\u8ac7\u4e2d\u306e\u66f8\u304d\u8d77\u3053\u3057\u3092\u8cbc\u308a\u4ed8\u3051\u3066\u304f\u3060\u3055\u3044..."></textarea>
  <div style="display:flex;gap:8px;margin-top:6px">
    <button class="btn btn-primary btn-sm" onclick="saveTranscript('${{mid}}')">
      \u4fdd\u5b58
    </button>
    <button class="btn btn-sm" onclick="genMinutes('${{mid}}')">
      \ud83d\udcdd \u8b70\u4e8b\u9332\u751f\u6210
    </button>
  </div>
  <div id="minutes-${{mid}}" class="minutes-out" style="display:none;margin-top:8px"></div>
</div>`;

  return `<div class="sections-grid">${{sections}}</div>${{refsHtml}}${{postActions}}`;
}}

async function generateBriefing(mid){{
  const companyEl=document.getElementById('company-'+mid);
  const modeEl=document.getElementById('mode-'+mid);
  const langEl=document.getElementById('lang-'+mid);
  const btn=document.getElementById('btn-'+mid);
  const prog=document.getElementById('prog-'+mid);
  const status=document.getElementById('status-'+mid);
  const resultDiv=document.getElementById('result-'+mid);

  const company=(companyEl&&companyEl.value.trim())||'Unknown';
  const mode=modeEl?modeEl.value:'standard';
  const language=langEl?langEl.value:'ja';

  btn.disabled=true;
  prog.style.width='20%';
  status.textContent=`${{company}} \u306e\u60c5\u5831\u3092\u53ce\u96c6\u4e2d... (Tavily 3calls + Wikipedia)`;
  resultDiv.classList.remove('show');

  try{{
    prog.style.width='50%';
    const resp=await fetch('/api/briefing',{{
      method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{company,mode,language,mid}})
    }});
    prog.style.width='80%';
    if(!resp.ok){{
      const err=await resp.json();
      throw new Error(err.detail||'API error');
    }}
    const data=await resp.json();
    prog.style.width='100%';
    status.textContent=`\u2705 \u30d6\u30ea\u30fc\u30d5\u30a3\u30f3\u30b0\u5b8c\u4e86: ${{company}}`;
    resultDiv.innerHTML=renderResult(mid,data);
    resultDiv.classList.add('show');
    updateHistory();
  }} catch(e){{
    prog.style.width='0%';
    status.textContent='\u274c \u30a8\u30e9\u30fc: '+e.message;
  }} finally{{
    btn.disabled=false;
  }}
}}

function showTranscript(mid){{
  const s=document.getElementById('transcript-section-'+mid);
  if(s) s.style.display=s.style.display==='none'?'block':'none';
}}

async function saveTranscript(mid){{
  const t=document.getElementById('transcript-'+mid);
  if(!t||!t.value.trim()) return;
  await fetch(`/api/transcript/${{mid}}`,{{
    method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{transcript:t.value}})
  }});
  alert('\u4fdd\u5b58\u3057\u307e\u3057\u305f');
}}

async function genMinutes(mid){{
  const t=document.getElementById('transcript-'+mid);
  const mout=document.getElementById('minutes-'+mid);
  if(!t||!t.value.trim()){{alert('\u66f8\u304d\u8d77\u3053\u3057\u3092\u5165\u529b\u3057\u3066\u304f\u3060\u3055\u3044');return;}}
  mout.style.display='block';
  mout.textContent='\u8b70\u4e8b\u9332\u751f\u6210\u4e2d...';
  const resp=await fetch(`/api/minutes/${{mid}}`,{{
    method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{transcript:t.value}})
  }});
  const data=await resp.json();
  mout.textContent=data.minutes||'error';
}}

function showScore(mid){{alert('RelationshipScore \u306f Phase 2 \u3067\u5b9f\u88c5\u4e2d\u3067\u3059')}}
function showHubspot(mid){{alert('HubSpot\u540c\u671f\u306f Phase 2 \u3067\u5b9f\u88c5\u4e2d\u3067\u3059')}}

async function updateHistory(){{
  const resp=await fetch('/api/history');
  if(!resp.ok) return;
  const h=await resp.json();
  const sec=document.getElementById('history-section');
  const list=document.getElementById('history-list');
  const entries=Object.entries(h);
  if(entries.length===0) return;
  sec.classList.add('show');
  list.innerHTML=entries.reverse().map(([mid,d])=>`
    <div class="history-item" onclick="location.href='/briefing/${{mid}}'">
      <div>
        <div class="history-company">${{esc(d.company||mid)}}</div>
        <div class="history-meta">${{esc(d.created_at||'')}} \u30fb ${{esc(d.mode||'')}}</div>
      </div>
      <a class="history-link" href="/briefing/${{mid}}" target="_blank">\u8a73\u7d30 \u2192</a>
    </div>`).join('');
}}

// Init
(function(){{
  const container=document.getElementById('meetings-container');
  if(MEETINGS.length>0){{
    document.getElementById('cal-count').textContent=`(${{MEETINGS.length}}\u4ef6)`;
    container.innerHTML=MEETINGS.map((m,i)=>renderMeetingCard(m,i)).join('')+renderManualCard();
  }} else {{
    container.innerHTML=renderManualCard();
  }}
  updateHistory();
}})();
</script>
</body>
</html>"""
    return html

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    sid = request.cookies.get("sid", "")
    if not sid or sid not in sessions:
        return RedirectResponse(url="/")
    return HTMLResponse(build_dashboard(sid))

# ── API エンドポイント ─────────────────────────────────────────────────────────
@app.post("/api/briefing")
async def api_briefing(request: Request):
    sid = request.cookies.get("sid", "")
    if not sid or sid not in sessions:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    body = await request.json()
    company  = body.get("company", "Unknown")
    mode     = body.get("mode", "standard")
    language = body.get("language", "ja")
    mid      = body.get("mid", "unknown")

    try:
        result = call_claude(company, mode, language)
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)

    briefing_cache.setdefault(sid, {})[mid] = {
        "company":    company,
        "result":     result,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode":       mode,
        "language":   language,
    }
    return JSONResponse(result)

@app.get("/api/history")
async def api_history(request: Request):
    sid = request.cookies.get("sid", "")
    if not sid or sid not in sessions:
        return JSONResponse({})
    cache = briefing_cache.get(sid, {})
    return JSONResponse({mid: {k: v for k, v in d.items() if k != "result"}
                         for mid, d in cache.items()})

@app.post("/api/transcript/{mid}")
async def api_transcript(mid: str, request: Request):
    sid = request.cookies.get("sid", "")
    if not sid or sid not in sessions:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    body = await request.json()
    transcript = body.get("transcript", "")
    meeting_states.setdefault(sid, {}).setdefault(mid, {})["transcript"] = transcript
    return JSONResponse({"status": "ok"})

@app.post("/api/minutes/{mid}")
async def api_minutes(mid: str, request: Request):
    sid = request.cookies.get("sid", "")
    if not sid or sid not in sessions:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    body = await request.json()
    transcript = body.get("transcript", "")
    company = briefing_cache.get(sid, {}).get(mid, {}).get("company", "Unknown")
    try:
        minutes = call_claude_minutes(company, transcript)
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)
    meeting_states.setdefault(sid, {}).setdefault(mid, {})["minutes"] = minutes
    return JSONResponse({"minutes": minutes})

@app.post("/api/score/{mid}")
async def api_score(mid: str, request: Request):
    sid = request.cookies.get("sid", "")
    if not sid or sid not in sessions:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    # Phase 2 stub
    return JSONResponse({"status": "phase2_stub", "score": None,
                         "message": "RelationshipScore は Phase 2 で実装予定です"})

@app.post("/api/hubspot/{mid}")
async def api_hubspot(mid: str, request: Request):
    sid = request.cookies.get("sid", "")
    if not sid or sid not in sessions:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    # Phase 2 stub
    return JSONResponse({"status": "phase2_stub",
                         "message": "HubSpot 同期は Phase 2 で実装予定です"})

# ── ブリーフィング詳細ページ ──────────────────────────────────────────────────
@app.get("/briefing/{mid}", response_class=HTMLResponse)
async def briefing_detail(mid: str, request: Request):
    sid = request.cookies.get("sid", "")
    if not sid or sid not in sessions:
        return RedirectResponse(url="/")

    cache_entry = briefing_cache.get(sid, {}).get(mid)
    if not cache_entry:
        return HTMLResponse("<h2>ブリーフィングが見つかりません</h2><a href='/dashboard'>戻る</a>")

    company  = cache_entry.get("company", "")
    result   = cache_entry.get("result", {})
    created  = cache_entry.get("created_at", "")
    mode     = cache_entry.get("mode", "standard")

    result_json = json.dumps(result, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{company} \u30d6\u30ea\u30fc\u30d5\u30a3\u30f3\u30b0</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans JP',sans-serif;
  background:#0f172a;color:#e2e8f0;padding:20px 24px;max-width:1000px;margin:0 auto}}
h1{{font-size:1.5rem;font-weight:800;color:#e2e8f0;margin-bottom:4px}}
.meta{{font-size:.8rem;color:#64748b;margin-bottom:20px}}
a.back{{color:#60a5fa;font-size:.85rem;text-decoration:none;display:inline-block;margin-bottom:16px}}
a.back:hover{{text-decoration:underline}}
.sections-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}
.section-card{{background:rgba(30,58,95,.25);border:1px solid #1e3a5f;border-radius:12px;padding:14px}}
.section-card h4{{font-size:.82rem;color:#60a5fa;margin-bottom:8px;font-weight:700}}
.section-card p{{font-size:.8rem;color:#94a3b8;line-height:1.6;white-space:pre-wrap}}
.section-card ul{{padding-left:16px;font-size:.8rem;color:#94a3b8;line-height:1.6}}
.proposals-list{{display:flex;flex-direction:column;gap:8px}}
.proposal-item{{background:rgba(37,99,235,.1);border:1px solid rgba(37,99,235,.25);
  border-radius:8px;padding:10px 12px}}
.proposal-item .idea{{font-size:.85rem;font-weight:600;color:#60a5fa}}
.proposal-item .detail{{font-size:.78rem;color:#94a3b8;margin-top:3px}}
.proposal-item .priority{{display:inline-block;font-size:.68rem;padding:2px 7px;
  border-radius:4px;background:rgba(245,158,11,.15);color:#f59e0b;margin-top:5px}}
.refs-wrap{{margin-top:16px}}
.refs-title{{font-size:.75rem;color:#475569;margin-bottom:6px}}
.refs-links{{display:flex;flex-wrap:wrap;gap:5px}}
.refs-links a{{font-size:.72rem;color:#60a5fa;text-decoration:none;
  background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.2);
  padding:3px 8px;border-radius:4px}}
.refs-links a:hover{{background:rgba(59,130,246,.2)}}
.inline-ref{{color:#2563eb;text-decoration:none;border-bottom:1px dotted #93c5fd;
  font-size:.8em;vertical-align:super}}
.inline-ref:hover{{text-decoration:underline}}
@media(max-width:700px){{.sections-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<a class="back" href="/dashboard">\u2190 \u30c0\u30c3\u30b7\u30e5\u30dc\u30fc\u30c9\u306b\u623b\u308b</a>
<h1>{company} \u30d6\u30ea\u30fc\u30d5\u30a3\u30f3\u30b0</h1>
<div class="meta">\u751f\u6210\u65e5\u6642: {created} \u30fb \u30e2\u30fc\u30c9: {mode}</div>
<div id="sections-container"></div>
<script>
const DATA={result_json};
const SECTIONS=[
  ['1','\ud83c\udfe2','\u4f1a\u793e\u6982\u8981','company_overview'],
  ['2','\ud83d\udcb0','\u8ca1\u52d9\u60c5\u5831','financial_info'],
  ['3','\ud83d\udcc8','\u5e02\u5834\u60c5\u5831','market_info'],
  ['4','\u2694\ufe0f','\u7af6\u5408\u5206\u6790','competitive_positioning'],
  ['5','\ud83d\udcf0','\u6700\u65b0\u30cb\u30e5\u30fc\u30b9','recent_news'],
  ['6','\ud83d\udca1','\u63d0\u6848\u8996\u70b9','proposal_ideas'],
  ['7','\u2615','\u30a2\u30a4\u30b9\u30d6\u30ec\u30a4\u30ad\u30f3\u30b0','icebreakers'],
  ['8','\ud83e\udde0','\u30d3\u30b8\u30cd\u30b9\u30a4\u30f3\u30b5\u30a4\u30c8','business_insight']
];
function esc(s){{return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}}
function renderMd(s){{
  const e=esc(s);
  return e.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener" class="inline-ref">$1</a>');
}}
function renderSec(key,val){{
  if(!val) return '<p style="color:#475569">\u30c7\u30fc\u30bf\u306a\u3057</p>';
  if(key==='proposal_ideas'&&Array.isArray(val)){{
    return '<div class="proposals-list">'+val.map(p=>`<div class="proposal-item"><div class="idea">${{esc(p.idea||'')}}</div><div class="detail">${{esc(p.detail||'')}}</div><div class="priority">\u512a\u5148\u5ea6: ${{esc(p.priority||'')}}</div></div>`).join('')+'</div>';
  }}
  if(key==='icebreakers'&&Array.isArray(val)){{
    return '<ul>'+val.map(s=>`<li>${{esc(s)}}</li>`).join('')+'</ul>';
  }}
  return `<p>${{renderMd(String(val||''))}}</p>`;
}}
const grid=SECTIONS.map(([n,ic,lb,k])=>`<div class="section-card"><h4>${{ic}} Section ${{n}}: ${{lb}}</h4>${{renderSec(k,DATA[k])}}</div>`).join('');
let refs='';
if(DATA.references){{
  const all=[];const seen=new Set();
  Object.values(DATA.references).forEach(arr=>{{if(Array.isArray(arr))arr.forEach(r=>{{if(r.url&&!seen.has(r.url)){{seen.add(r.url);all.push(r)}}}});}});
  if(all.length)refs='<div class="refs-wrap"><div class="refs-title">\u53c2\u8003\u30bd\u30fc\u30b9</div><div class="refs-links">'+all.map(r=>`<a href="${{esc(r.url)}}" target="_blank">${{esc(r.title||r.url.slice(0,40))}}</a>`).join('')+'</div></div>';
}}
document.getElementById('sections-container').innerHTML='<div class="sections-grid">'+grid+'</div>'+refs;
</script>
</body>
</html>"""
    return HTMLResponse(html)

# ── Demo Mode ─────────────────────────────────────────────────────────────────
DEMO_LIMIT_PER_DAY = 5
demo_rate_limit: dict = {}  # {ip: {"count": int, "date": str}}

def _demo_remaining(ip: str) -> int:
    today = datetime.date.today().isoformat()
    info = demo_rate_limit.get(ip, {"count": 0, "date": ""})
    if info["date"] != today:
        return DEMO_LIMIT_PER_DAY
    return max(0, DEMO_LIMIT_PER_DAY - info["count"])

def _demo_consume(ip: str) -> bool:
    today = datetime.date.today().isoformat()
    if ip not in demo_rate_limit or demo_rate_limit[ip]["date"] != today:
        demo_rate_limit[ip] = {"count": 0, "date": today}
    if demo_rate_limit[ip]["count"] >= DEMO_LIMIT_PER_DAY:
        return False
    demo_rate_limit[ip]["count"] += 1
    return True

DEMO_HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MeetingBrief AI — Try Demo</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans JP',sans-serif;
  background:#0f172a;color:#e2e8f0;min-height:100vh}
.header{background:rgba(15,23,42,0.97);border-bottom:1px solid #1e3a5f;
  padding:14px 24px;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:100}
.logo{font-size:1.2rem;font-weight:800;color:#60a5fa;letter-spacing:-.01em}
.demo-badge{margin-left:10px;background:rgba(245,158,11,.15);border:1px solid rgba(245,158,11,.4);
  color:#fbbf24;font-size:.72rem;font-weight:700;padding:3px 10px;border-radius:999px;letter-spacing:.04em}
.header-right{margin-left:auto;font-size:.78rem;color:#64748b}
.hero{padding:48px 24px 32px;max-width:760px;margin:0 auto;text-align:center}
.hero h1{font-size:2rem;font-weight:800;line-height:1.2;margin-bottom:14px}
.hero h1 span{color:#60a5fa}
.hero p{font-size:.95rem;color:#94a3b8;line-height:1.7;max-width:600px;margin:0 auto 24px}
.flow-pills{display:flex;align-items:center;justify-content:center;gap:8px;flex-wrap:wrap;margin-bottom:32px}
.pill{padding:6px 16px;border-radius:999px;font-size:.8rem;font-weight:600}
.pill-blue{background:rgba(37,99,235,.2);border:1px solid rgba(37,99,235,.4);color:#93c5fd}
.pill-green{background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.3);color:#6ee7b7}
.arrow{color:#475569;font-size:.9rem}
.demo-box{background:rgba(30,58,95,.2);border:1px solid #1e3a5f;border-radius:18px;
  padding:28px 32px;max-width:760px;margin:0 auto 40px;text-align:left}
.demo-box h2{font-size:1.05rem;font-weight:700;color:#e2e8f0;margin-bottom:20px;
  display:flex;align-items:center;gap:10px}
.demo-box h2 .quota-badge{margin-left:auto;font-size:.72rem;font-weight:600;
  background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);
  color:#6ee7b7;padding:3px 10px;border-radius:999px}
.input-row{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap}
.company-input{flex:1;min-width:220px;background:#0a1929;border:1px solid #1e3a5f;
  color:#e2e8f0;padding:10px 14px;border-radius:10px;font-size:.95rem;outline:none}
.company-input:focus{border-color:#3b82f6;box-shadow:0 0 0 3px rgba(59,130,246,.15)}
.company-input::placeholder{color:#475569}
select.mode-sel{background:#0a1929;border:1px solid #1e3a5f;color:#e2e8f0;
  padding:10px 12px;border-radius:10px;font-size:.85rem;outline:none}
.btn-gen{background:#2563eb;color:#fff;border:none;padding:10px 24px;border-radius:10px;
  font-size:.9rem;font-weight:700;cursor:pointer;white-space:nowrap;transition:all .2s}
.btn-gen:hover{background:#1d4ed8;transform:translateY(-1px)}
.btn-gen:disabled{background:#1e3a5f;color:#475569;cursor:not-allowed;transform:none}
.hint{font-size:.75rem;color:#475569;margin-top:6px}
.hint span{color:#64748b;font-weight:600}
.prog-wrap{height:3px;background:#1e3a5f;border-radius:2px;margin:14px 0 8px;overflow:hidden;display:none}
.prog-bar{height:100%;background:linear-gradient(90deg,#2563eb,#60a5fa);width:0%;transition:width .5s;border-radius:2px}
.status-txt{font-size:.8rem;color:#94a3b8;min-height:20px;margin-bottom:8px}
.result-area{display:none;margin-top:20px}
.result-area.show{display:block}
.result-header{display:flex;align-items:center;gap:10px;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #1e3a5f}
.result-company{font-size:1.1rem;font-weight:700;color:#60a5fa}
.result-meta{font-size:.75rem;color:#64748b;margin-top:2px}
.sections-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:14px}
.sec-card{background:rgba(15,23,42,.7);border:1px solid #1e3a5f;border-radius:10px;padding:12px}
.sec-card h4{font-size:.78rem;color:#60a5fa;margin-bottom:8px;font-weight:700}
.sec-card p{font-size:.78rem;color:#94a3b8;line-height:1.6;white-space:pre-line}
.sec-card ul{padding-left:14px;font-size:.78rem;color:#94a3b8;line-height:1.6}
.proposal-item{background:rgba(37,99,235,.08);border:1px solid rgba(37,99,235,.2);
  border-radius:8px;padding:8px 10px;margin-bottom:6px}
.proposal-item .idea{font-size:.8rem;font-weight:600;color:#60a5fa}
.proposal-item .detail{font-size:.74rem;color:#94a3b8;margin-top:2px}
.proposal-item .pri{display:inline-block;font-size:.65rem;padding:1px 7px;
  border-radius:4px;background:rgba(245,158,11,.12);color:#f59e0b;margin-top:4px}
.refs-wrap{margin-top:12px;padding-top:12px;border-top:1px solid #1e3a5f;display:none}
.refs-title{font-size:.7rem;color:#475569;margin-bottom:6px}
.refs-links{display:flex;flex-wrap:wrap;gap:4px}
.refs-links a{font-size:.68rem;color:#60a5fa;text-decoration:none;
  background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.2);padding:2px 7px;border-radius:4px}
.refs-links a:hover{background:rgba(59,130,246,.18)}
.inline-ref{color:#3b82f6;text-decoration:none;border-bottom:1px dotted #93c5fd;font-size:.8em;vertical-align:super}
.cta-bar{background:rgba(37,99,235,.06);border:1px solid rgba(37,99,235,.2);
  border-radius:12px;padding:16px 20px;max-width:760px;margin:0 auto 48px;
  display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.cta-bar p{flex:1;font-size:.82rem;color:#94a3b8;line-height:1.5}
.error-box{background:rgba(220,38,38,.1);border:1px solid rgba(220,38,38,.3);
  border-radius:8px;padding:10px 14px;font-size:.8rem;color:#fca5a5;margin-top:8px;display:none}
.error-box.show{display:block}
@media(max-width:640px){
  .hero h1{font-size:1.5rem}
  .demo-box{padding:20px 16px}
  .sections-grid{grid-template-columns:1fr}
  .input-row{flex-direction:column}
}
</style>
</head>
<body>
<div class="header">
  <div class="logo">⚡ MeetingBrief AI</div>
  <span class="demo-badge">🎯 Demo Mode</span>
  <div class="header-right">Powered by Claude (Anthropic) × Tavily</div>
</div>

<div class="hero">
  <h1>商談前<span>30分</span>のリサーチを<br><span>5分</span>に。AIが自動生成。</h1>
  <p>会社名を入力するだけで、企業概要・財務情報・競合分析・最新ニュース・提案視点を<br>
  10項目で即時生成。BizDev担当者が自ら設計・実装・毎日使用しているツールです。</p>
  <div class="flow-pills">
    <span class="pill pill-blue">📅 Google Calendar 自動連携</span>
    <span class="arrow">→</span>
    <span class="pill pill-blue">🔍 企業名自動検知</span>
    <span class="arrow">→</span>
    <span class="pill pill-green">⚡ 5分でブリーフィング生成</span>
    <span class="arrow">→</span>
    <span class="pill pill-green">📝 商談後 議事録自動生成</span>
  </div>
</div>

<div class="demo-box">
  <h2>
    🔍 今すぐ試す — 任意の会社名を入力
    <span class="quota-badge" id="quota-badge">残り 5 回 / 今日</span>
  </h2>
  <div class="input-row">
    <input class="company-input" id="company-input" type="text"
      placeholder="例: SmartHR、Salesforce、トヨタ自動車、Apple..."
      onkeydown="if(event.key==='Enter')generate()">
    <select class="mode-sel" id="mode-sel">
      <option value="standard">標準</option>
      <option value="short">簡潔</option>
      <option value="detail">詳細</option>
    </select>
    <button class="btn-gen" id="btn-gen" onclick="generate()">⚡ ブリーフィング生成</button>
  </div>
  <p class="hint">※ <span>リアルタイムで検索 × Claude AI が10項目を自動生成</span>（約15〜30秒）。デモは1日5回まで。</p>
  <div class="prog-wrap" id="prog-wrap"><div class="prog-bar" id="prog-bar"></div></div>
  <div class="status-txt" id="status-txt"></div>
  <div class="error-box" id="error-box"></div>
  <div class="result-area" id="result-area">
    <div class="result-header">
      <div>
        <div class="result-company" id="result-company"></div>
        <div class="result-meta" id="result-meta"></div>
      </div>
    </div>
    <div class="sections-grid" id="sections-grid"></div>
    <div class="refs-wrap" id="refs-wrap">
      <div class="refs-title">参考ソース</div>
      <div class="refs-links" id="refs-links"></div>
    </div>
  </div>
</div>

<div class="cta-bar">
  <p>
    <strong>フル機能版</strong>ではGoogle Calendarと連携し、当日の商談を自動検知。
    議事録自動生成・RelationshipScore・HubSpot同期（Phase 2）・GEO分析も搭載。
  </p>
</div>

<script>
const SECTIONS=[
  ['1','🏢','会社概要','company_overview'],
  ['2','💰','財務情報','financial_info'],
  ['3','📈','市場情報','market_info'],
  ['4','⚔️','競合分析','competitive_positioning'],
  ['5','📰','最新ニュース','recent_news'],
  ['6','💡','提案視点','proposal_ideas'],
  ['7','☕','アイスブレイキング','icebreakers'],
  ['8','🧠','ビジネスインサイト','business_insight']
];
function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
function renderMd(s){return esc(s).replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener" class="inline-ref">$1</a>')}
function renderSection(key,val){
  if(!val) return '<p style="color:#475569">データなし</p>';
  if(key==='proposal_ideas'&&Array.isArray(val)){
    return val.map(p=>`<div class="proposal-item"><div class="idea">${esc(p.idea||'')}</div><div class="detail">${esc(p.detail||'')}</div><div class="pri">優先度: ${esc(p.priority||'')}</div></div>`).join('');
  }
  if(key==='icebreakers'&&Array.isArray(val)){
    return '<ul>'+val.map(s=>`<li>${esc(s)}</li>`).join('')+'</ul>';
  }
  return `<p>${renderMd(String(val))}</p>`;
}
async function fetchQuota(){
  try{
    const r=await fetch('/api/demo-quota');
    const d=await r.json();
    const badge=document.getElementById('quota-badge');
    badge.textContent=`残り ${d.remaining} 回 / 今日`;
    if(d.remaining<=0){
      document.getElementById('btn-gen').disabled=true;
      badge.style.background='rgba(220,38,38,.1)';
      badge.style.color='#fca5a5';
      badge.style.borderColor='rgba(220,38,38,.3)';
    }
  }catch(e){}
}
async function generate(){
  const company=document.getElementById('company-input').value.trim();
  if(!company){document.getElementById('company-input').focus();return;}
  const mode=document.getElementById('mode-sel').value;
  const btn=document.getElementById('btn-gen');
  const progWrap=document.getElementById('prog-wrap');
  const progBar=document.getElementById('prog-bar');
  const statusTxt=document.getElementById('status-txt');
  const errorBox=document.getElementById('error-box');
  const resultArea=document.getElementById('result-area');
  btn.disabled=true;
  errorBox.classList.remove('show');
  resultArea.classList.remove('show');
  progWrap.style.display='block';
  progBar.style.width='15%';
  statusTxt.textContent=`${company} の情報を収集中... (Tavily + Wikipedia)`;
  const ticker=setInterval(()=>{
    const cur=parseFloat(progBar.style.width)||15;
    if(cur<72) progBar.style.width=(cur+2.5)+'%';
  },700);
  try{
    const resp=await fetch('/api/demo-briefing',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({company,mode,language:'ja'})
    });
    clearInterval(ticker);
    progBar.style.width='92%';
    if(!resp.ok){
      const err=await resp.json();
      errorBox.textContent=resp.status===429?'デモ利用上限（1日5回）に達しました。明日またお試しください。':'エラー: '+(err.detail||'不明なエラー');
      errorBox.classList.add('show');
      progBar.style.width='0%';
      statusTxt.textContent='';
      return;
    }
    const data=await resp.json();
    progBar.style.width='100%';
    statusTxt.textContent=`✅ ブリーフィング完了: ${company}`;
    document.getElementById('result-company').textContent=company+' ブリーフィング';
    document.getElementById('result-meta').textContent=new Date().toLocaleString('ja-JP')+' · モード: '+mode+' · claude-haiku-4-5 + Tavily';
    document.getElementById('sections-grid').innerHTML=SECTIONS.map(([n,ic,lb,k])=>`<div class="sec-card"><h4>${ic} Section ${n}: ${lb}</h4>${renderSection(k,data[k])}</div>`).join('');
    if(data.references){
      const all=[];const seen=new Set();
      Object.values(data.references).forEach(arr=>{if(Array.isArray(arr))arr.forEach(r=>{if(r.url&&!seen.has(r.url)){seen.add(r.url);all.push(r)}})});
      if(all.length){
        document.getElementById('refs-links').innerHTML=all.map(r=>`<a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title||r.url.slice(0,40))}</a>`).join('');
        document.getElementById('refs-wrap').style.display='block';
      }
    }
    resultArea.classList.add('show');
    resultArea.scrollIntoView({behavior:'smooth',block:'start'});
    fetchQuota();
  }catch(e){
    clearInterval(ticker);
    errorBox.textContent='ネットワークエラー: '+e.message;
    errorBox.classList.add('show');
    progBar.style.width='0%';
    statusTxt.textContent='';
  }finally{
    btn.disabled=false;
    setTimeout(()=>{progWrap.style.display='none';},1200);
  }
}
fetchQuota();
document.getElementById('company-input').focus();
</script>
</body>
</html>"""

@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    return HTMLResponse(DEMO_HTML)

@app.get("/api/demo-quota")
async def api_demo_quota(request: Request):
    ip = request.client.host
    return JSONResponse({"remaining": _demo_remaining(ip), "limit": DEMO_LIMIT_PER_DAY})

@app.post("/api/demo-briefing")
async def api_demo_briefing(request: Request):
    ip = request.client.host
    if not _demo_consume(ip):
        return JSONResponse(
            {"detail": f"デモ制限: 1日{DEMO_LIMIT_PER_DAY}回まで利用可能です"},
            status_code=429
        )
    body = await request.json()
    company = body.get("company", "").strip()
    mode     = body.get("mode", "standard")
    language = body.get("language", "ja")
    if not company:
        return JSONResponse({"detail": "会社名を入力してください"}, status_code=400)
    try:
        result = call_claude(company, mode, language)
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)
    return JSONResponse(result)

# ── health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "anthropic": True, "model": CLAUDE_MODEL})
