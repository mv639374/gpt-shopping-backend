import os
from supabase import create_client, Client
from dotenv import load_dotenv
import httpx

load_dotenv()

def test_supabase_connection():
    # Print environment variables (without sensitive data)
    print("Environment Variables:")
    print(f"SUPABASE_URL: {os.getenv('SUPABASE_URL')}")
    print(f"SUPABASE_KEY present: {'Yes' if os.getenv('SUPABASE_KEY') else 'No'}")
    print(f"SUPABASE_SERVICE_KEY present: {'Yes' if os.getenv('SUPABASE_SERVICE_KEY') else 'No'}\n")
    
    # Try with both regular and service role keys
    keys_to_try = [
        ("anon/public", os.getenv("SUPABASE_KEY")),
        ("service role", os.getenv("SUPABASE_SERVICE_KEY"))
    ]
    
    for key_type, supabase_key in keys_to_try:
        if not supabase_key:
            print(f"Skipping {key_type} key (not set)")
            continue
            
        print(f"\nTrying with {key_type} key...")
        
        try:
            # Test basic connection first
            print("Testing basic connection...")
            supabase: Client = create_client(
                os.getenv("SUPABASE_URL"),
                supabase_key
            )
            
            # Test fetching tables
            print("Testing tables...")
            tables = supabase.table('pg_tables')\
                .select('*')\
                .execute()
            print(f"Available tables: {[t['tablename'] for t in tables.data if t['schemaname'] == 'public']}")
            
            # Try a simple query on product_analysis
            print("\nTrying to fetch from product_analysis...")
            result = supabase.table('product_analysis')\
                .select('*', count='exact')\
                .limit(1)\
                .execute()
            
            print(f"Query successful! Found {result.count} rows in product_analysis")
            if result.data:
                print("First row columns:", list(result.data[0].keys()))
            return True
                
        except Exception as e:
            print(f"Error with {key_type} key:")
            print(f"Type: {type(e).__name__}")
            print(f"Error: {str(e)}")
            if hasattr(e, 'args') and len(e.args) > 1:
                print("Additional error details:", e.args[1])
    
    return False

if __name__ == "__main__":
    test_supabase_connection()