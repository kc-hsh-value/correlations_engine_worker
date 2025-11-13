from datetime import datetime, timezone
from typing import Optional
from supabase_client import get_supabase



def find_expired_markets(now_utc: Optional[datetime] = None):
    sb = get_supabase()
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    # We’ll just SELECT here, no updates
    query = """
        SELECT id, question, end_date_utc
        FROM markets
        WHERE is_active
          AND (
               end_date_utc < now()
            OR (full_data_json->>'closed')::boolean IS TRUE
            OR full_data_json->>'umaResolutionStatus' = 'resolved'
            OR (full_data_json->'market'->>'closed')::boolean IS TRUE
            OR full_data_json->'market'->>'umaResolutionStatus' = 'resolved'
            OR jsonb_path_exists(full_data_json, '$.events[*] ? (@.closed == true)')
            OR jsonb_path_exists(full_data_json, '$.events[*] ? (@.umaResolutionStatus == "resolved")')
          )
        ORDER BY end_date_utc DESC
        LIMIT 50
    """
    res = sb.rpc("exec_sql", {"q": query}).execute()
    return res.data

if __name__ == "__main__":
    rows = find_expired_markets()
    print(f"Found {len(rows)} expired markets")
    for r in rows[:10]:
        print(r)