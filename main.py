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
        "business_insight": [], "product_reviews": [], "wiki_text": "", "wiki_url": "",
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

            # Call 4: 製品評判 G2/Gartner/Capterra (basic, 6件)
            resp4 = tc.search(
                query=f"{company} G2 Gartner Capterra review rating score pros cons evaluation",
                search_depth="basic",
                max_results=6,
            )
            for r in resp4.get("results", []):
                item = {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")[:300]}
                bundle["product_reviews"].append(item)

        except Exception as e:
            print(f"[Tavily error] {e}")

    # ── 5. 重複排除・上限設定 ─────────────────────────────────────────────────
    limits = {
        "company_overview": 3, "financial_info": 4, "market_info": 3,
        "competitive_positioning": 3, "recent_news": 5, "sns_official": 2,
        "business_insight": 5, "product_reviews": 5,
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
    if sources["product_reviews"]:
        context_block += f"[製品評判・G2/Gartner参考]\n{_fmt_refs(sources['product_reviews'], 5)}\n\n"

    all_refs = {
        "company_overview":        sources["company_overview"],
        "financial_info":          sources["financial_info"],
        "market_info":             sources["market_info"],
        "competitive_positioning": sources["competitive_positioning"],
        "recent_news":             sources["recent_news"],
        "business_insight":        sources["business_insight"],
        "product_reviews":         sources["product_reviews"],
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
        "8.business_insightはsummary（総括文）・pain_points（主要課題リスト）・"
        "tech_stack（技術スタックリスト）・opportunities（営業機会リスト）の4フィールドで返すこと。"
        "9.product_reviewsはG2/Gartner/Capterra参考情報からsummary・g2_score・gartner_score・"
        "g2_reviews・pros（強みリスト）・cons（課題リスト）・sales_tipを生成。"
        "スコア不明の場合は空文字。prosは3〜5件、consは2〜4件。"
        "10.geo_analysisはAI検索エンジン（Claude/ChatGPT/Gemini/Perplexity）での"
        "企業言及率をWeb情報から推定（0〜100%）し、geo_score（0〜100の総合AIブランド評点）・"
        "summary・ai_engines（engine/mention_rate/trend）・top_topics・sales_insightを返す。"
        "推定値であることをsummaryに明記すること。"
    )

    tool_schema = {
        "name": "generate_briefing",
        "description": "商談前ブリーフィングを10セクションのJSON形式で生成する",
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
                "icebreakers": {"type": "array", "items": {"type": "string"}},
                "business_insight": {
                    "type": "object",
                    "properties": {
                        "summary":      {"type": "string"},
                        "pain_points":  {"type": "array", "items": {"type": "string"}},
                        "tech_stack":   {"type": "array", "items": {"type": "string"}},
                        "opportunities":{"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["summary", "pain_points", "tech_stack", "opportunities"],
                },
                "product_reviews": {
                    "type": "object",
                    "properties": {
                        "summary":       {"type": "string"},
                        "g2_score":      {"type": "string"},
                        "g2_reviews":    {"type": "string"},
                        "gartner_score": {"type": "string"},
                        "pros":          {"type": "array", "items": {"type": "string"}},
                        "cons":          {"type": "array", "items": {"type": "string"}},
                        "sales_tip":     {"type": "string"},
                    },
                    "required": ["summary", "pros", "cons", "sales_tip"],
                },
                "geo_analysis": {
                    "type": "object",
                    "properties": {
                        "geo_score":  {"type": "integer"},
                        "summary":    {"type": "string"},
                        "ai_engines": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "engine":       {"type": "string"},
                                    "mention_rate": {"type": "integer"},
                                    "trend":        {"type": "string"},
                                },
                                "required": ["engine", "mention_rate", "trend"],
                            },
                        },
                        "top_topics":    {"type": "array", "items": {"type": "string"}},
                        "sales_insight": {"type": "string"},
                    },
                    "required": ["geo_score", "summary", "ai_engines", "sales_insight"],
                },
            },
            "required": [
                "company_overview", "financial_info", "market_info",
                "competitive_positioning", "recent_news",
                "proposal_ideas", "icebreakers", "business_insight",
                "product_reviews", "geo_analysis",
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
        max_tokens=6000,
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
<title>MeetingBrief AI — Demo</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans JP',sans-serif;
  background:#080f1e;color:#e2e8f0;min-height:100vh}
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
/* Stepper */
.stepper{display:flex;align-items:flex-start;justify-content:center;gap:0;margin-bottom:32px;flex-wrap:wrap;padding:0 8px}
.step{display:flex;flex-direction:column;align-items:center;text-align:center}
.step-circle{width:38px;height:38px;border-radius:50%;border:2px solid #1e3a5f;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.8rem;margin-bottom:5px;background:rgba(255,255,255,.03);color:#475569}
.step-active .step-circle{background:linear-gradient(135deg,#2563eb,#3b82f6);border-color:#3b82f6;color:#fff;box-shadow:0 0 0 4px rgba(59,130,246,.18),0 4px 14px rgba(37,99,235,.4)}
.step-label{font-size:.7rem;color:#475569;max-width:72px;line-height:1.3}
.step-active .step-label{color:#93c5fd;font-weight:600}
.step-tag{font-size:.6rem;background:rgba(245,158,11,.15);color:#fbbf24;border:1px solid rgba(245,158,11,.25);padding:1px 7px;border-radius:999px;margin-top:4px;white-space:nowrap}
.step-connector{flex:1;height:2px;background:#1e3a5f;margin-top:18px;min-width:16px;max-width:44px;opacity:.5}
.step-connector.act{background:linear-gradient(90deg,#1e3a5f,#2563eb);opacity:1}
/* Demo box */
.demo-box{background:rgba(8,20,44,.5);border:1px solid rgba(59,130,246,.15);border-radius:18px;
  padding:28px 32px;max-width:760px;margin:0 auto 40px;text-align:left;
  box-shadow:0 0 0 1px rgba(59,130,246,.07),0 8px 40px rgba(0,0,0,.5),0 0 80px rgba(37,99,235,.05)}
.demo-box h2{font-size:1.05rem;font-weight:700;color:#e2e8f0;margin-bottom:6px;display:flex;align-items:center;gap:10px}
.quota-row{display:flex;align-items:center;gap:8px;margin-bottom:18px}
.quota-label{font-size:.7rem;color:#64748b}
.quota-track{width:80px;height:5px;background:rgba(255,255,255,.05);border-radius:3px;overflow:hidden}
.quota-fill{height:100%;background:linear-gradient(90deg,#22c55e,#16a34a);border-radius:3px;transition:width .5s}
.quota-fill.warn{background:linear-gradient(90deg,#f97316,#ea580c)}
.quota-fill.empty{background:linear-gradient(90deg,#ef4444,#dc2626)}
.quota-num{font-size:.72rem;font-weight:700;color:#6ee7b7}
/* Input */
.input-row{display:flex;gap:10px;margin-bottom:6px;flex-wrap:wrap}
.company-input{flex:1;min-width:220px;background:#050c1a;border:1px solid #1e3a5f;
  color:#e2e8f0;padding:11px 14px;border-radius:10px;font-size:.95rem;outline:none;transition:border-color .2s,box-shadow .2s}
.company-input:focus{border-color:#3b82f6;box-shadow:0 0 0 3px rgba(59,130,246,.15)}
.company-input.err{border-color:rgba(239,68,68,.5)}
.company-input::placeholder{color:#334155}
select.mode-sel{background:#050c1a;border:1px solid #1e3a5f;color:#e2e8f0;
  padding:11px 12px;border-radius:10px;font-size:.85rem;outline:none}
.btn-gen{background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;border:none;padding:11px 22px;border-radius:10px;
  font-size:.9rem;font-weight:700;cursor:pointer;white-space:nowrap;transition:all .2s;display:flex;align-items:center;gap:7px}
.btn-gen:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 4px 18px rgba(37,99,235,.45)}
.btn-gen:disabled{background:#1e3a5f;color:#475569;cursor:not-allowed;transform:none;box-shadow:none}
.btn-spin{display:none;width:14px;height:14px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;flex-shrink:0}
.btn-gen.loading .btn-spin{display:block}
.btn-gen.loading .btn-lbl{opacity:.7}
@keyframes spin{to{transform:rotate(360deg)}}
.inline-error{font-size:.74rem;color:#fca5a5;margin-bottom:8px;display:none;
  padding:7px 11px;background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.2);border-radius:8px;line-height:1.5}
.inline-error.show{display:block}
.hint{font-size:.75rem;color:#475569;margin-top:4px}
.hint span{color:#64748b;font-weight:600}
.quick-btns{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 6px;align-items:center}
.quick-label{font-size:.72rem;color:#475569;margin-right:2px;white-space:nowrap}
.quick-btn{background:rgba(30,58,95,.4);border:1px solid #1e3a5f;color:#94a3b8;
  padding:4px 11px;border-radius:8px;font-size:.78rem;cursor:pointer;transition:all .15s;white-space:nowrap}
.quick-btn:hover{background:rgba(59,130,246,.15);border-color:rgba(59,130,246,.4);color:#93c5fd}
.prog-wrap{height:4px;background:#0d1f3c;border-radius:2px;margin:14px 0 6px;overflow:hidden;display:none}
.prog-bar{height:100%;background:linear-gradient(90deg,#2563eb,#60a5fa,#818cf8);background-size:200% 100%;width:0%;transition:width .6s ease;border-radius:2px;animation:shimmer 2s linear infinite}
@keyframes shimmer{0%{background-position:0% 0}100%{background-position:200% 0}}
.status-wrap{display:flex;align-items:center;gap:7px;min-height:20px;margin-bottom:8px}
.status-dot{width:6px;height:6px;border-radius:50%;background:#3b82f6;flex-shrink:0;display:none;animation:sdot 1.2s ease-in-out infinite}
.status-dot.show{display:block}
@keyframes sdot{0%,100%{opacity:.2;transform:scale(.7)}50%{opacity:1;transform:scale(1.3)}}
.status-txt{font-size:.78rem;color:#64748b}
.result-area{display:none;margin-top:20px}
.result-area.show{display:block}
.result-header{display:flex;align-items:center;gap:10px;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #1e3a5f}
.result-company{font-size:1.1rem;font-weight:700;color:#60a5fa}
.result-meta{font-size:.75rem;color:#64748b;margin-top:2px}
.sections-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:14px}
.sec-card{background:rgba(8,15,30,.8);border:1px solid #1e3a5f;border-left-width:3px;border-radius:10px;padding:14px}
.sec-hd{display:flex;align-items:center;gap:6px;margin-bottom:10px}
.sec-n{font-size:.65rem;background:rgba(255,255,255,.06);color:#64748b;width:18px;height:18px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0}
.sec-lb{font-size:.8rem;font-weight:700;color:#e2e8f0}
.body-txt{font-size:.78rem;color:#94a3b8;line-height:1.65;white-space:pre-line}
.proposal-item{background:rgba(37,99,235,.07);border:1px solid rgba(37,99,235,.18);
  border-radius:8px;padding:8px 10px;margin-bottom:6px}
.proposal-item .idea{font-size:.8rem;font-weight:600;color:#60a5fa}
.proposal-item .detail{font-size:.74rem;color:#94a3b8;margin-top:2px}
.proposal-item .pri{display:inline-block;font-size:.65rem;padding:1px 7px;border-radius:4px;margin-top:4px}
.pri-h{background:rgba(239,68,68,.12);color:#fca5a5}
.pri-m{background:rgba(245,158,11,.12);color:#fbbf24}
.pri-l{background:rgba(34,197,94,.12);color:#86efac}
.score-pill{display:inline-flex;align-items:center;gap:4px;font-size:.75rem;font-weight:700;padding:4px 10px;border-radius:8px;background:rgba(234,179,8,.1);border:1px solid rgba(234,179,8,.3);color:#fde68a}
.pros-sec,.cons-sec{margin-top:6px}
.pros-sec ul,.cons-sec ul{padding-left:14px;font-size:.75rem;color:#94a3b8;line-height:1.5}
.ins-cols{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:8px}
.ins-col{background:rgba(255,255,255,.02);border:1px solid #1e3a5f;border-radius:8px;padding:8px}
.geo-circle{width:62px;height:62px;border-radius:50%;border:3px solid;display:flex;flex-direction:column;align-items:center;justify-content:center;flex-shrink:0}
.ai-row{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.refs-wrap{margin-top:12px;padding-top:12px;border-top:1px solid #1e3a5f;display:none}
.refs-title{font-size:.7rem;color:#475569;margin-bottom:6px}
.refs-links{display:flex;flex-wrap:wrap;gap:4px}
.refs-links a{font-size:.68rem;color:#60a5fa;text-decoration:none;
  background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.2);padding:2px 7px;border-radius:4px}
.refs-links a:hover{background:rgba(59,130,246,.18)}
.inline-ref{color:#3b82f6;text-decoration:none;border-bottom:1px dotted #93c5fd;font-size:.8em;vertical-align:super}
.cta-bar{border:1px solid rgba(59,130,246,.1);border-radius:14px;padding:20px 22px;max-width:760px;margin:0 auto 48px;background:rgba(8,15,30,.7)}
.cta-bar-title{font-size:.7rem;color:#475569;text-align:center;margin-bottom:12px;text-transform:uppercase;letter-spacing:.08em}
.cta-cards{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.cta-card{background:rgba(37,99,235,.05);border:1px solid rgba(59,130,246,.1);border-radius:10px;padding:12px 10px;text-align:center;transition:border-color .15s}
.cta-card:hover{border-color:rgba(59,130,246,.25)}
.cta-card-icon{font-size:1.25rem;margin-bottom:5px}
.cta-card-ttl{font-size:.73rem;font-weight:700;color:#93c5fd;margin-bottom:3px}
.cta-card-desc{font-size:.67rem;color:#475569;line-height:1.3}
.cta-card-tag{font-size:.6rem;background:rgba(168,85,247,.1);color:#c084fc;border:1px solid rgba(168,85,247,.2);padding:1px 6px;border-radius:999px;display:inline-block;margin-top:3px}
@media(max-width:600px){.cta-cards{grid-template-columns:repeat(2,1fr)}}
@media(max-width:700px){.ins-cols{grid-template-columns:1fr}}
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
  <div class="header-right">Powered by Claude (Anthropic) × Tavily × G2/Gartner</div>
</div>

<div class="hero">
  <h1>商談前<span>30分</span>のリサーチを<br><span>5分</span>に。AIが自動生成。</h1>
  <p>会社名を入力するだけで、企業概要・財務情報・競合分析・最新ニュース・提案視点を<br>
  10項目で即時生成。BizDev担当者が自ら設計・実装・毎日使用しているツールです。</p>
  <div class="stepper">
    <div class="step">
      <div class="step-circle">①</div>
      <div class="step-label">Google Calendar<br>自動連携</div>
    </div>
    <div class="step-connector act"></div>
    <div class="step">
      <div class="step-circle">②</div>
      <div class="step-label">企業名<br>自動検知</div>
    </div>
    <div class="step-connector act"></div>
    <div class="step step-active">
      <div class="step-circle">③</div>
      <div class="step-label">ブリーフィング<br>自動生成</div>
      <div class="step-tag">← 今ここ（体験可）</div>
    </div>
    <div class="step-connector"></div>
    <div class="step">
      <div class="step-circle">④</div>
      <div class="step-label">議事録<br>自動生成</div>
    </div>
  </div>
</div>

<div class="demo-box">
  <h2>🔍 今すぐ試す</h2>
  <div class="quota-row">
    <span class="quota-label">本日の残り回数</span>
    <div class="quota-track"><div class="quota-fill" id="quota-fill" style="width:100%"></div></div>
    <span class="quota-num" id="quota-num">5 / 5回</span>
  </div>
  <div class="input-row">
    <input class="company-input" id="company-input" type="text"
      placeholder="調べたい会社名を入力（例: トヨタ自動車、Salesforce）"
      onkeydown="if(event.key==='Enter')generate()">
    <select class="mode-sel" id="mode-sel">
      <option value="standard">標準</option>
      <option value="short">簡潔</option>
      <option value="detail">詳細</option>
    </select>
    <button class="btn-gen" id="btn-gen" onclick="generate()">
      <span class="btn-spin"></span><span class="btn-lbl">⚡ ブリーフィング生成</span>
    </button>
  </div>
  <div class="inline-error" id="inline-error"></div>
  <div class="quick-btns">
    <span class="quick-label">例:</span>
    <button class="quick-btn" onclick="quickSelect('トヨタ自動車')">🚗 トヨタ自動車</button>
    <button class="quick-btn" onclick="quickSelect('SoftBank')">📱 SoftBank</button>
    <button class="quick-btn" onclick="quickSelect('Salesforce')">☁️ Salesforce</button>
    <button class="quick-btn" onclick="quickSelect('SmartHR')">👥 SmartHR</button>
    <button class="quick-btn" onclick="quickSelect('freee')">💼 freee</button>
  </div>
  <p class="hint">※ <span>リアルタイムで検索 × Claude AI が10項目を自動生成</span>（Tavily × Wikipedia × G2/Gartner、約15〜30秒）。デモは1日5回まで。</p>
  <div class="prog-wrap" id="prog-wrap"><div class="prog-bar" id="prog-bar"></div></div>
  <div class="status-wrap"><div class="status-dot" id="status-dot"></div><div class="status-txt" id="status-txt"></div></div>
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
  <div class="cta-bar-title">フル機能版でさらに</div>
  <div class="cta-cards">
    <div class="cta-card">
      <div class="cta-card-icon">📅</div>
      <div class="cta-card-ttl">Calendar連携</div>
      <div class="cta-card-desc">当日の商談を自動検知・ブリーフィング配信</div>
    </div>
    <div class="cta-card">
      <div class="cta-card-icon">📝</div>
      <div class="cta-card-ttl">議事録自動生成</div>
      <div class="cta-card-desc">書き起こし → 構造化議事録に即時変換</div>
    </div>
    <div class="cta-card">
      <div class="cta-card-icon">🔗</div>
      <div class="cta-card-ttl">HubSpot同期</div>
      <div class="cta-card-desc">商談メモをCRMへ自動転記</div>
      <div class="cta-card-tag">Phase 2</div>
    </div>
    <div class="cta-card">
      <div class="cta-card-icon">📊</div>
      <div class="cta-card-ttl">Relationship Score</div>
      <div class="cta-card-desc">顧客関係深度の定量化</div>
      <div class="cta-card-tag">Phase 2</div>
    </div>
  </div>
</div>

<script>
const SECTIONS=[
  ['1','🏢','会社概要','company_overview','#3b82f6'],
  ['2','💰','財務情報','financial_info','#8b5cf6'],
  ['3','📈','市場情報','market_info','#06b6d4'],
  ['4','⚔️','競合分析','competitive_positioning','#f97316'],
  ['5','⭐','製品レビュー','product_reviews','#eab308'],
  ['6','📰','最新ニュース','recent_news','#14b8a6'],
  ['7','💡','提案視点','proposal_ideas','#ef4444'],
  ['8','☕','アイスブレイキング','icebreakers','#22c55e'],
  ['9','🧠','ビジネスインサイト','business_insight','#3b82f6'],
  ['10','🌍','GEO分析','geo_analysis','#a855f7']
];
const FULL=['business_insight','geo_analysis'];
function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
function renderMd(s){return esc(s).replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener" class="inline-ref">$1</a>')}
function renderReviews(d){
  if(!d||typeof d!=='object')return '<p style="color:#475569">データなし</p>';
  let h='';
  if(d.g2_score||d.gartner_score){
    h+='<div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap">';
    if(d.g2_score)h+=`<span class="score-pill">G2 ⭐ ${esc(String(d.g2_score))}</span>`;
    if(d.gartner_score)h+=`<span class="score-pill" style="background:rgba(139,92,246,.1);border-color:rgba(139,92,246,.3);color:#c4b5fd">Gartner ⭐ ${esc(String(d.gartner_score))}</span>`;
    h+='</div>';
  }
  if(d.summary)h+=`<p class="body-txt" style="margin-bottom:8px">${renderMd(d.summary)}</p>`;
  if(Array.isArray(d.pros)&&d.pros.length)h+='<div class="pros-sec"><div style="font-size:.7rem;color:#22c55e;font-weight:700;margin-bottom:3px">👍 PROS</div><ul>'+d.pros.map(x=>`<li>${esc(x)}</li>`).join('')+'</ul></div>';
  if(Array.isArray(d.cons)&&d.cons.length)h+='<div class="cons-sec"><div style="font-size:.7rem;color:#ef4444;font-weight:700;margin-bottom:3px">👎 CONS</div><ul>'+d.cons.map(x=>`<li>${esc(x)}</li>`).join('')+'</ul></div>';
  if(d.sales_tip)h+=`<div style="margin-top:8px;padding:8px;background:rgba(234,179,8,.08);border:1px solid rgba(234,179,8,.2);border-radius:6px;font-size:.75rem;color:#fde68a">💡 ${esc(d.sales_tip)}</div>`;
  return h||'<p style="color:#475569">データなし</p>';
}
function renderGeo(d){
  if(!d||typeof d!=='object')return '<p style="color:#475569">データなし</p>';
  let h='<div>';
  if(d.geo_score!=null){
    const score=parseInt(d.geo_score)||0;
    const col=score>=70?'#22c55e':score>=40?'#eab308':'#ef4444';
    h+=`<div style="display:flex;align-items:center;gap:14px;margin-bottom:12px">`;
    h+=`<div class="geo-circle" style="border-color:${col}"><span style="color:${col};font-size:1.1rem;font-weight:800">${score}</span><span style="font-size:.6rem;color:#64748b">/ 100</span></div>`;
    h+=`<div><div style="font-size:.82rem;font-weight:700;color:#e2e8f0">AI可視性スコア</div>`;
    if(d.summary)h+=`<div style="font-size:.75rem;color:#94a3b8;margin-top:2px">${esc(d.summary)}</div>`;
    h+=`</div></div>`;
  }
  if(Array.isArray(d.ai_engines)&&d.ai_engines.length){
    h+='<div style="margin-bottom:10px"><div style="font-size:.7rem;color:#64748b;margin-bottom:5px">AIエンジン別カバレッジ</div>';
    d.ai_engines.forEach(e=>{
      const pct=Math.min(100,parseInt(e.mention_rate)||0);
      h+=`<div class="ai-row"><span style="font-size:.72rem;color:#94a3b8;width:64px;flex-shrink:0">${esc(e.engine||'')}</span><div style="flex:1;height:6px;background:#1e3a5f;border-radius:3px"><div style="width:${pct}%;height:100%;background:#a855f7;border-radius:3px"></div></div><span style="font-size:.7rem;color:#c084fc;width:30px;text-align:right">${pct}%</span></div>`;
    });
    h+='</div>';
  }
  if(Array.isArray(d.top_topics)&&d.top_topics.length){
    h+='<div style="margin-bottom:8px"><div style="font-size:.7rem;color:#64748b;margin-bottom:4px">検索クエリカバー</div><div style="display:flex;flex-wrap:wrap;gap:4px">'+d.top_topics.map(t=>`<span style="background:rgba(168,85,247,.1);border:1px solid rgba(168,85,247,.2);color:#c084fc;font-size:.68rem;padding:2px 8px;border-radius:999px">${esc(t)}</span>`).join('')+'</div></div>';
  }
  if(d.sales_insight)h+=`<div style="padding:8px;background:rgba(168,85,247,.08);border:1px solid rgba(168,85,247,.2);border-radius:6px;font-size:.75rem;color:#e9d5ff">💡 ${esc(d.sales_insight)}</div>`;
  h+='</div>';
  return h;
}
function renderInsight(d){
  if(!d||typeof d!=='object')return`<p class="body-txt">${renderMd(String(d||''))}</p>`;
  let h='';
  if(d.summary)h+=`<p class="body-txt" style="margin-bottom:10px">${renderMd(d.summary)}</p>`;
  h+='<div class="ins-cols">';
  const fmtArr=(v)=>Array.isArray(v)?'<ul style="padding-left:14px;font-size:.75rem;color:#94a3b8;line-height:1.5">'+v.map(x=>`<li>${esc(x)}</li>`).join('')+'</ul>':`<p class="body-txt">${renderMd(String(v||''))}</p>`;
  if(d.pain_points)h+=`<div class="ins-col"><div style="font-size:.7rem;color:#ef4444;font-weight:700;margin-bottom:5px">⚠️ 主要課題</div>${fmtArr(d.pain_points)}</div>`;
  if(d.tech_stack)h+=`<div class="ins-col"><div style="font-size:.7rem;color:#60a5fa;font-weight:700;margin-bottom:5px">🔧 技術スタック</div>${fmtArr(d.tech_stack)}</div>`;
  if(d.opportunities)h+=`<div class="ins-col"><div style="font-size:.7rem;color:#22c55e;font-weight:700;margin-bottom:5px">🚀 営業機会</div>${fmtArr(d.opportunities)}</div>`;
  h+='</div>';
  return h;
}
function renderSec(k,v){
  if(!v)return '<p style="color:#475569">データなし</p>';
  if(k==='product_reviews')return renderReviews(v);
  if(k==='geo_analysis')return renderGeo(v);
  if(k==='business_insight')return typeof v==='object'?renderInsight(v):`<p class="body-txt">${renderMd(String(v))}</p>`;
  if(k==='proposal_ideas'&&Array.isArray(v)){
    const pc={high:'pri-h',medium:'pri-m',low:'pri-l'};
    return v.map(p=>`<div class="proposal-item"><div class="idea">${esc(p.idea||'')}</div><div class="detail">${esc(p.detail||'')}</div><div class="pri ${pc[(p.priority||'').toLowerCase()]||'pri-m'}">優先度: ${esc(p.priority||'')}</div></div>`).join('');
  }
  if(k==='icebreakers'&&Array.isArray(v)){
    return '<div style="display:flex;flex-wrap:wrap;gap:6px">'+v.map(s=>`<div style="background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.2);border-radius:8px;padding:6px 10px;font-size:.78rem;color:#86efac">☕ ${esc(s)}</div>`).join('')+'</div>';
  }
  return `<p class="body-txt">${renderMd(String(v))}</p>`;
}
async function fetchQuota(){
  try{
    const r=await fetch('/api/demo-quota');
    const d=await r.json();
    const rem=d.remaining,lim=d.limit||5;
    const fill=document.getElementById('quota-fill');
    const num=document.getElementById('quota-num');
    if(fill){fill.style.width=(rem/lim*100)+'%';fill.className='quota-fill'+(rem<=1?' empty':rem<=2?' warn':'');}
    if(num){num.textContent=`${rem} / ${lim}回`;num.style.color=rem<=1?'#fca5a5':rem<=2?'#fbbf24':'#6ee7b7';}
    if(rem<=0)document.getElementById('btn-gen').disabled=true;
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
  const statusDot=document.getElementById('status-dot');
  const inlineErr=document.getElementById('inline-error');
  const compInput=document.getElementById('company-input');
  const resultArea=document.getElementById('result-area');
  const MSGS=[
    `${company} の基本情報・Wikipedia を検索中...`,
    `${company} のニュース・財務情報を収集中...`,
    `製品レビュー・競合情報を分析中...`,
    `Claude AI がブリーフィングを生成中...`
  ];
  btn.disabled=true;
  btn.classList.add('loading');
  inlineErr.classList.remove('show');
  compInput.classList.remove('err');
  resultArea.classList.remove('show');
  progWrap.style.display='block';
  progBar.style.width='12%';
  statusDot.classList.add('show');
  statusTxt.textContent=MSGS[0];
  let msgIdx=0;
  const ticker=setInterval(()=>{
    const cur=parseFloat(progBar.style.width)||12;
    if(cur<74)progBar.style.width=(cur+1.8)+'%';
    if(msgIdx<MSGS.length-1){msgIdx++;statusTxt.textContent=MSGS[msgIdx];}
  },4500);
  try{
    const resp=await fetch('/api/demo-briefing',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({company,mode,language:'ja'})
    });
    clearInterval(ticker);
    progBar.style.width='92%';
    statusDot.classList.remove('show');
    if(!resp.ok){
      const err=await resp.json().catch(()=>({}));
      const msg=resp.status===429?'デモ利用上限（1日5回）に達しました。明日またお試しください。'
        :resp.status>=500?'サーバーエラーが発生しました。しばらく後でお試しください。（'+( err.detail||'unknown')+'）'
        :(err.detail||'エラーが発生しました。会社名を確認して再試行してください。');
      inlineErr.textContent=msg;
      inlineErr.classList.add('show');
      compInput.classList.add('err');
      progBar.style.width='0%';
      statusTxt.textContent='';
      return;
    }
    const data=await resp.json();
    progBar.style.width='100%';
    statusTxt.textContent=`✅ ブリーフィング完了: ${company}`;
    document.getElementById('result-company').textContent=company+' ブリーフィング';
    document.getElementById('result-meta').textContent=new Date().toLocaleString('ja-JP')+' · モード: '+mode+' · claude-haiku-4-5 + Tavily × G2/Gartner';
    document.getElementById('sections-grid').innerHTML=SECTIONS.map(([n,ic,lb,k,col])=>{
      const fw=FULL.includes(k)?'grid-column:1/-1;':'';
      return `<div class="sec-card" style="border-left-color:${col};${fw}"><div class="sec-hd"><span class="sec-n">${n}</span><span class="sec-lb">${ic} ${lb}</span></div>${renderSec(k,data[k])}</div>`;
    }).join('');
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
    statusDot.classList.remove('show');
    inlineErr.textContent='接続エラー: ネットワーク環境をご確認ください。';
    inlineErr.classList.add('show');
    compInput.classList.add('err');
    progBar.style.width='0%';
    statusTxt.textContent='';
  }finally{
    btn.disabled=false;
    btn.classList.remove('loading');
    setTimeout(()=>{progWrap.style.display='none';},1200);
  }
}
function quickSelect(name){
  document.getElementById('company-input').value=name;
  generate();
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
