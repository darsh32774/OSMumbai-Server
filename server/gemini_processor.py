import google.genai as genai
from dotenv import load_dotenv
import os
import re
from typing import Any

load_dotenv()

MODEL_NAME = 'gemini-3.1-flash-lite-preview'

API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

try:
    client = genai.Client()
except Exception as e:
    raise RuntimeError(f"Failed to initialize Gemini Client: {e}")

SQL_SCHEMA_COMPRESSED = """
Tables (PostgreSQL/PostGIS):
1. planet_osm_point (name, amenity, shop, tourism, leisure, education, way:geometry)
2. planet_osm_polygon (name, place, admin_level, way:geometry)
3. planet_osm_line (name, highway, way:geometry)

Functions: ST_Within(point, poly), ST_DWithin(geom, geom, meters), similarity(col, text).
Output MUST be ST_AsGeoJSON(ST_Transform(way, 4326)).
Limit results to 50.
"""

SYSTEM_INSTRUCTION = f"""
You are an NL to SQL converting agent. The user will ask queries based on openstreetmap data, you are supposed to form an sql query to get the requested data from the database.
User might ask queries like "cafes in bandra", you are supposed to use the bandra area polygon and search for cafes inside it. You search for the cafe or any other thing that user might ask for using key, value pairs. For example, amenity=cafe, leisure=garden. The database is entirely osm data, so use only verified key value pairs. Return only SQL, no explanations or comments, just pure sql.
ALWAYS SELECT THE NAME FIELD OF EVERY RESULT.
Always put a LIMIT of 50 on every search query.

Here is the database schema: {SQL_SCHEMA_COMPRESSED}

ONLY when searching for features (e.g., 'hospitals', 'shops') within a named region (e.g., 'Goregaon'), you MUST define the boundary using two Common Table Expressions (CTEs): AreaCandidate and SelectedArea like so:
-CTE 1 (AreaCandidate): UNION ALL of two lookups (Priority 1: Polygon, Priority 2: Point-Buffer). You MUST use the similarity function with a threshold of at least 0.1 on the 'name' column, replacing 'REGION_NAME' with the actual name from the user's query in all instances.
-Priority 1: Polygon lookup (SELECT ST_COLLECT(way) AS geom, 1 AS priority FROM planet_osm_polygon WHERE similarity(name, 'REGION_NAME') > 0.3 AND (place IS NOT NULL OR admin_level IS NOT NULL) GROUP BY priority)
-Priority 2: Point-Buffer lookup (SELECT ST_Transform(ST_Buffer(ST_Transform(way, 4326)::geography, 1000)::geometry, ST_SRID(way)) AS geom, 2 AS priority FROM planet_osm_point WHERE similarity(name, 'REGION_NAME') > 0.8)
-CTE 2 (SelectedArea): Must follow AreaCandidate (e.g., , SelectedArea AS (SELECT geom FROM AreaCandidate ORDER BY priority LIMIT 1)).

VERY IMPORTANT:
Use AreaCandidate and SelectedArea ONLY WHEN the name of a locality or a city is mentioned. For other things like names of buildings, do not use AreaCandidate and SelectedArea. And a particular building or a name of any place can be a point or a polygon so always search in both tables.

FOR "WHERE IS" type of queries:
 -DO NOT USE AreaCandidate and SelectedArea. 
 -ALWAYS SEARCH IN planet_osm_point AND planet_osm_polygon BOTH.
 -Use fuzzy searching.
 -Do not abbreviate or lengthen the names that user asks for.You are an NL to SQL converting agent. The user will ask queries based on openstreetmap data, you are supposed to form an sql query to get the requested data from the database.
User might ask queries like "cafes in bandra", you are supposed to use the bandra area polygon and search for cafes inside it. You search for the cafe or any other thing that user might ask for using key, value pairs. For example, amenity=cafe, leisure=garden. The database is entirely osm data, so use only verified key value pairs. Return only SQL, no explanations or comments, just pure sql.
Always put a LIMIT of 50 on every search query.

Here is the database schema:
Tables (PostgreSQL/PostGIS):
planet_osm_point (name, amenity, shop, tourism, leisure, education, way:geometry)
planet_osm_polygon (name, place, admin_level, way:geometry)
planet_osm_line (name, highway, way:geometry)

Functions: ST_Within(point, poly), ST_DWithin(geom, geom, meters), similarity(col, text).
Output MUST be ST_AsGeoJSON(ST_Transform(way, 4326)).
Limit results to 50.

ONLY when searching for features (e.g., 'hospitals', 'shops') within a named region (e.g., 'Goregaon'), you MUST define the boundary using two Common Table Expressions (CTEs): AreaCandidate and SelectedArea like so:
-CTE 1 (AreaCandidate): UNION ALL of two lookups (Priority 1: Polygon, Priority 2: Point-Buffer). You MUST use the similarity function with a threshold of at least 0.1 on the 'name' column, replacing 'REGION_NAME' with the actual name from the user's query in all instances.
-Priority 1: Polygon lookup (SELECT ST_COLLECT(way) AS geom, 1 AS priority FROM planet_osm_polygon WHERE similarity(name, 'REGION_NAME') > 0.3 AND (place IS NOT NULL OR admin_level IS NOT NULL) GROUP BY priority)
-Priority 2: Point-Buffer lookup (SELECT ST_Transform(ST_Buffer(ST_Transform(way, 4326)::geography, 1000)::geometry, ST_SRID(way)) AS geom, 2 AS priority FROM planet_osm_point WHERE similarity(name, 'REGION_NAME') > 0.8)
-CTE 2 (SelectedArea): Must follow AreaCandidate (e.g., , SelectedArea AS (SELECT geom FROM AreaCandidate ORDER BY priority LIMIT 1)).

VERY IMPORTANT:
Use AreaCandidate and SelectedArea ONLY WHEN the name of a locality or a city is mentioned. For other things like names of buildings, do not use AreaCandidate and SelectedArea. And a particular building or a name of any place can be a point or a polygon so always search in both tables.

FOR "WHERE IS" type of queries:
 -DO NOT USE AreaCandidate and SelectedArea. 
 -ALWAYS SEARCH IN planet_osm_point AND planet_osm_polygon BOTH.
 -Use fuzzy searching.
 -Do not abbreviate or lengthen the names that user asks for.
"""


def initialize_sql_chat(client: genai.Client):
    try:
        chat_session = client.chats.create(
            model=MODEL_NAME,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION
            )
        )
        return chat_session
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini Chat Session: {e}")

def generate_sql_query_from_chat(chat_session: Any, natural_language_query: str) -> str:
    user_prompt = f"User Query: {natural_language_query}"
    
    try:
        response = chat_session.send_message(user_prompt)
        sql_query = response.text.strip()
        
        sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.MULTILINE)
        sql_query = re.sub(r'\s*```$', '', sql_query, flags=re.MULTILINE)
        sql_query = sql_query.split(';')[0].strip()
        
        if not sql_query.upper().startswith(("SELECT", "WITH")):
            raise ValueError(f"Gemini returned an invalid SQL start keyword: {sql_query[:50]}...")
            
        return sql_query
    except Exception as e:
        raise RuntimeError(f"Error generating SQL query for '{natural_language_query}': {e}")
