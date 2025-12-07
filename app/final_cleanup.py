"""
Final cleanup script: Keep only rows where prompt_response_id exists in ALL 3 tables
Deletes all other rows from prompts_responses_structured, response_brand_mentions, and citations
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import Set, List, Dict

load_dotenv()

SUPABASE_URL="https://eakywgrtiadiegjzjezd.supabase.co"
SUPABASE_SERVICE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVha3l3Z3J0aWFkaWVnanpqZXpkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzM5NTg3NCwiZXhwIjoyMDc4OTcxODc0fQ.gTqwnx0E4DMqzztK779Tc1zXkEBAYCbn0BMkcG3keqo"


supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

print("=" * 80)
print("🧹 FINAL CLEANUP: KEEP ONLY COMPLETE RECORDS")
print("=" * 80)
print("Keep only rows where prompt_response_id exists in ALL 3 tables")
print("=" * 80)


def get_all_ids_from_table(table_name: str) -> Set[str]:
    """Get all unique prompt_response_ids from a table with pagination"""
    print(f"\n📊 Getting IDs from '{table_name}'...")
    
    all_ids = set()
    page_size = 1000
    page = 0
    
    while True:
        result = supabase.table(table_name)\
            .select("prompt_response_id")\
            .range(page * page_size, (page + 1) * page_size - 1)\
            .execute()
        
        if not result.data:
            break
        
        for row in result.data:
            all_ids.add(row["prompt_response_id"])
        
        if len(result.data) < page_size:
            break
        
        page += 1
    
    print(f"   ✅ Found {len(all_ids)} unique prompt_response_ids")
    
    # Show samples
    if all_ids:
        samples = list(all_ids)[:3]
        print(f"   Sample: {', '.join([id[:16] + '...' for id in samples])}")
    
    return all_ids


def calculate_deletions(structured_ids: Set[str], mention_ids: Set[str], citation_ids: Set[str]) -> Dict:
    """
    Calculate which IDs to keep (common in all 3) and which to delete
    """
    print("\n" + "=" * 80)
    print("📊 CALCULATING COMMON AND UNIQUE IDs")
    print("=" * 80)
    
    # Find common IDs (exist in ALL 3 tables)
    common_ids = structured_ids & mention_ids & citation_ids
    
    # Find IDs to delete from each table (NOT in common)
    structured_to_delete = structured_ids - common_ids
    mentions_to_delete = mention_ids - common_ids
    citations_to_delete = citation_ids - common_ids
    
    print(f"\n✅ Common IDs (in all 3 tables):               {len(common_ids)}")
    print(f"❌ IDs to delete from structured:              {len(structured_to_delete)}")
    print(f"❌ IDs to delete from mentions:                {len(mentions_to_delete)}")
    print(f"❌ IDs to delete from citations:               {len(citations_to_delete)}")
    
    # Show sample common IDs
    if common_ids:
        print(f"\n✅ Sample common IDs (will be KEPT):")
        for i, response_id in enumerate(list(common_ids)[:5], 1):
            print(f"   {i}. {response_id}")
    
    # Show sample deletion IDs
    if structured_to_delete:
        print(f"\n❌ Sample IDs to DELETE from structured:")
        for i, response_id in enumerate(list(structured_to_delete)[:5], 1):
            print(f"   {i}. {response_id}")
    
    if mentions_to_delete:
        print(f"\n❌ Sample IDs to DELETE from mentions:")
        for i, response_id in enumerate(list(mentions_to_delete)[:5], 1):
            print(f"   {i}. {response_id}")
    
    if citations_to_delete:
        print(f"\n❌ Sample IDs to DELETE from citations:")
        for i, response_id in enumerate(list(citations_to_delete)[:5], 1):
            print(f"   {i}. {response_id}")
    
    return {
        "common": list(common_ids),
        "structured_delete": list(structured_to_delete),
        "mentions_delete": list(mentions_to_delete),
        "citations_delete": list(citations_to_delete)
    }


def get_row_ids_to_delete(table_name: str, response_ids: List[str]) -> List[str]:
    """
    Get actual row IDs for deletion (handles multiple rows per prompt_response_id)
    """
    if not response_ids:
        return []
    
    print(f"\n📊 Getting row IDs from '{table_name}' for {len(response_ids)} prompt_response_ids...")
    
    row_ids = []
    batch_size = 100
    
    for i in range(0, len(response_ids), batch_size):
        batch = response_ids[i:i + batch_size]
        
        result = supabase.table(table_name)\
            .select("id, prompt_response_id")\
            .in_("prompt_response_id", batch)\
            .execute()
        
        for row in result.data:
            row_ids.append(row["id"])
    
    print(f"   ✅ Found {len(row_ids)} actual rows to delete")
    return row_ids


def delete_rows_by_id(table_name: str, row_ids: List[str]) -> int:
    """
    Delete rows from table by their ID
    """
    if not row_ids:
        print(f"\n✅ No rows to delete from '{table_name}'")
        return 0
    
    print(f"\n🗑️  Deleting {len(row_ids)} rows from '{table_name}'...")
    
    deleted_count = 0
    batch_size = 50
    
    for i in range(0, len(row_ids), batch_size):
        batch = row_ids[i:i + batch_size]
        
        for row_id in batch:
            try:
                result = supabase.table(table_name)\
                    .delete()\
                    .eq("id", row_id)\
                    .execute()
                
                if result.data:
                    deleted_count += len(result.data)
            except Exception as e:
                print(f"   ⚠️  Failed to delete row {row_id}: {str(e)[:50]}")
        
        if (i + batch_size) % 200 == 0 or (i + batch_size) >= len(row_ids):
            print(f"   Progress: {min(i + batch_size, len(row_ids))}/{len(row_ids)} rows processed")
    
    print(f"   ✅ Successfully deleted {deleted_count} rows from '{table_name}'")
    return deleted_count


def main():
    try:
        # Step 1: Get all unique prompt_response_ids from each table
        print("\n" + "=" * 80)
        print("STEP 1: FETCHING IDs FROM ALL TABLES")
        print("=" * 80)
        
        structured_ids = get_all_ids_from_table("prompts_responses_structured")
        mention_ids = get_all_ids_from_table("response_brand_mentions")
        citation_ids = get_all_ids_from_table("citations")
        
        # Step 2: Calculate what to keep and what to delete
        deletion_plan = calculate_deletions(structured_ids, mention_ids, citation_ids)
        
        total_to_delete = (
            len(deletion_plan["structured_delete"]) +
            len(deletion_plan["mentions_delete"]) +
            len(deletion_plan["citations_delete"])
        )
        
        if total_to_delete == 0:
            print("\n🎉 All records are already complete! Nothing to delete.")
            return
        
        # Step 3: Confirmation
        print("\n" + "=" * 80)
        print("⚠️  FINAL CONFIRMATION")
        print("=" * 80)
        print(f"\n✅ Records to KEEP:   {len(deletion_plan['common'])} prompt_response_ids (exist in all 3 tables)")
        print(f"❌ Records to DELETE: {total_to_delete} prompt_response_ids (incomplete records)")
        print(f"\nBreakdown:")
        print(f"   - Delete from prompts_responses_structured: {len(deletion_plan['structured_delete'])} IDs")
        print(f"   - Delete from response_brand_mentions:      {len(deletion_plan['mentions_delete'])} IDs")
        print(f"   - Delete from citations:                    {len(deletion_plan['citations_delete'])} IDs")
        
        confirm = input("\n❓ Are you sure you want to proceed? (yes/no): ").strip().lower()
        
        if confirm != "yes":
            print("\n❌ Operation cancelled by user")
            return
        
        # Step 4: Get actual row IDs to delete
        print("\n" + "=" * 80)
        print("STEP 2: IDENTIFYING ROWS TO DELETE")
        print("=" * 80)
        
        structured_row_ids = get_row_ids_to_delete("prompts_responses_structured", deletion_plan["structured_delete"])
        mention_row_ids = get_row_ids_to_delete("response_brand_mentions", deletion_plan["mentions_delete"])
        citation_row_ids = get_row_ids_to_delete("citations", deletion_plan["citations_delete"])
        
        total_rows = len(structured_row_ids) + len(mention_row_ids) + len(citation_row_ids)
        
        print(f"\n📊 Total rows to delete: {total_rows}")
        print(f"   - prompts_responses_structured: {len(structured_row_ids)} rows")
        print(f"   - response_brand_mentions:      {len(mention_row_ids)} rows")
        print(f"   - citations:                    {len(citation_row_ids)} rows")
        
        # Step 5: Delete rows (delete children first, then parent)
        print("\n" + "=" * 80)
        print("STEP 3: DELETING ROWS")
        print("=" * 80)
        
        # Delete child tables first
        mentions_deleted = delete_rows_by_id("response_brand_mentions", mention_row_ids)
        citations_deleted = delete_rows_by_id("citations", citation_row_ids)
        
        # Delete parent table last
        structured_deleted = delete_rows_by_id("prompts_responses_structured", structured_row_ids)
        
        # Final Summary
        print("\n" + "=" * 80)
        print("📊 FINAL DELETION SUMMARY")
        print("=" * 80)
        print(f"\n✅ Successfully deleted:")
        print(f"   - prompts_responses_structured: {structured_deleted} rows")
        print(f"   - response_brand_mentions:      {mentions_deleted} rows")
        print(f"   - citations:                    {citations_deleted} rows")
        print(f"\n🎉 Total rows removed: {structured_deleted + mentions_deleted + citations_deleted}")
        print(f"\n✅ Remaining records: {len(deletion_plan['common'])} complete prompt_response_ids")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Script failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
