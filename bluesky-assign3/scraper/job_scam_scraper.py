import os, csv, re, time
from dotenv import load_dotenv
from atproto import Client

load_dotenv()
USERNAME = os.getenv("USERNAME")
PW = os.getenv("PW")

client = Client()
client.login(USERNAME, PW)

GENERAL_JOB_KEYWORDS = [
    "hiring", "job opening", "career opportunity", "join our team",
    "we are recruiting", "software engineer role", "remote developer",
    "internship", "graduate program", "talent acquisition",
]

SCAM_PATTERNS = {
    # A. pay-to-earn / social-task grifts
    "social_tasks": [
        r"earn\s+\$?\d+\s*(/|per)?\s*(day|week|hour|hr|month)\b",
        r"make\s+\$?\d+\s*(/|per)?\s*(day|week|hour|hr|month)\b",
        r"\$?\d+\s+per\s+(day|week|hour|hr|month)\b",
        r"get paid\s+\$?\d+\s+(daily|weekly|hourly)\b",
        r"like (tweets|posts) (and|&) get paid",
        r"get paid to (retweet|like|share)",
    ],

    # B. zero-barrier promises
    "no_barrier": [
        r"no experience (needed|required)",
        r"work from home.*start (today|now)",
        r"quick cash", r"easy money", r"instant payout",
        r"daily payout", r"same-day pay",
    ],

    "funnel": [
        r"telegram\.me/", r"t\.me/", r"join our telegram",
        r"whatsapp.*\+?\d{6,}",
    ],
}

SCAM_KEYWORDS_FOR_SEARCH = [

    "work 2 hours AND day", "work 3 hours AND home",
    "earn AND no experience", "earn AND no degree", "earn AND basic knowledge",
    "easy money", "quick cash", "instant payout",

    "like AND get paid", "like tweets AND earn", "click ads AND get paid",
    "surveys AND get paid", "data entry AND $300",

    "DM for details", "WhatsApp AND earn", "telegram AND earn",
    "t.me", "telegram.me",
]

CTA_KEYWORDS = [
    "apply now", "apply today", "sign up", "sign-up",
    "join our telegram", "join telegram", "join our whatsapp", "join whatsapp",
    "dm me", "dm us", "message me", "contact us", "inbox me",
    "register here", "click the link", "start today",
]
CTA_REGEX = re.compile("|".join(CTA_KEYWORDS), re.IGNORECASE)

def is_scam_by_regex(text: str) -> bool:
    """True if text matches *any* scam pattern."""
    tl = text.lower()
    for bucket in SCAM_PATTERNS.values():
        for pattern in bucket:
            if re.search(pattern, tl, flags=re.IGNORECASE):
                return True
    return False


def keyword_gate(text: str, kw: str) -> bool:
    """Case-insensitive substring match for broad search."""
    return kw.lower() in text.lower()

def search_posts(keywords, max_posts, label, apply_scam_filter=False):
    """
    Collect up to max_posts posts that contain any keyword in `keywords`.
    If `apply_scam_filter` is True, a post must satisfy:
        (money-bait regex) AND (CTA present)
    """
    collected = []
    for kw in keywords:
        cursor = None
        while len(collected) < max_posts:
            if apply_scam_filter:
                print(len(collected))
            res = client.app.bsky.feed.search_posts(
                {"q": kw, "limit": 25, "cursor": cursor,
                 "sort": "latest", "lang": "en"}
            )
            for p in res.posts or []:
                rec = getattr(p, "record", None)
                if not rec or not hasattr(rec, "text"):
                    continue
                text = rec.text

                # --- scam decision -------------------------------
                if apply_scam_filter:
                    money_bait = is_scam_by_regex(text)
                    has_cta    = CTA_REGEX.search(text) is not None
                    if not (money_bait):
                        continue  # not a scam
                # --------------------------------------------------

                # --- in the general bucket, keep only job-relevant posts
                if not apply_scam_filter and not any(
                        kw_job in text.lower() for kw_job in GENERAL_JOB_KEYWORDS):
                    continue

                # --- pull a few useful fields
                links = []
                if getattr(rec, "facets", None):
                    for facet in rec.facets:
                        for feat in facet.features:
                            if hasattr(feat, "uri"):
                                links.append(feat.uri)

                collected.append({
                    "text": text,
                    "creator": p.author.handle,
                    "likes": getattr(p, "like_count", 0),
                    "reposts": getattr(p, "repost_count", 0),
                    "responses": getattr(p, "reply_count", 0),
                    "links": ", ".join(links),
                    "label": label,
                })

                if len(collected) >= max_posts:
                    break

            cursor = getattr(res, "cursor", None)
            if not cursor:
                break
            time.sleep(1)  # gentle rate-limit
    return collected


GENERAL_TARGET = 75
SCAM_TARGET    = 75

print("🔎 Collecting general job posts …")
general_posts = search_posts(GENERAL_JOB_KEYWORDS, GENERAL_TARGET,
                             label="general_job", apply_scam_filter=False)

print("🔎 Collecting suspected job-scam posts …")
scam_posts = search_posts(SCAM_KEYWORDS_FOR_SEARCH, SCAM_TARGET,
                          label="job_scam", apply_scam_filter=True)

print(f"✅ Harvested {len(general_posts)} general + {len(scam_posts)} scam posts")

csv_name = "job_offer_scam_dataset.csv"
with open(csv_name, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "text", "creator", "likes", "reposts", "responses",
            "links", "label",
        ],
    )
    writer.writeheader()
    writer.writerows(general_posts + scam_posts)

print(f"📄  Wrote dataset →  {csv_name}")
