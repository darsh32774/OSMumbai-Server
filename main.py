import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from server.gemini_processor import initialize_sql_chat, generate_sql_query_from_chat, client
from server.database_processor import execute_query_raw, ensure_extensions
from server.map_processor import create_folium_map

chat_sessions: dict = {
    'accuracy': None,
    'speed': None,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize DB extensions and both Gemini chat sessions."""
    ensure_extensions()

    for mode in ('accuracy', 'speed'):
        try:
            chat_sessions[mode] = initialize_sql_chat(client, mode=mode)
            print(f"Gemini chat session initialized: mode='{mode}'")
        except Exception as e:
            print(f"WARNING: Could not initialize chat session for mode='{mode}': {e}")

    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mumbai OSM NL-to-Map API"}

@app.post("/nl-to-map")
async def nl_to_map(request: dict):
    query = request.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="'query' field is required.")

    mode = request.get("mode", "accuracy")
    if mode not in ("accuracy", "speed"):
        mode = "accuracy"

    chat_session = chat_sessions.get(mode)
    if chat_session is None:
        raise HTTPException(
            status_code=503,
            detail=f"Gemini service unavailable for mode='{mode}'. Check server logs."
        )

    try:
        sql = generate_sql_query_from_chat(chat_session, query)
        print(f"[mode={mode}] Generated SQL: {sql}")

        headers, db_rows = execute_query_raw(sql)

        geom_col_index = None
        for i, h in enumerate(headers):
            h_lower = h.lower()
            if h_lower in ("geojson", "st_asgeojson", "coordinates") or "st_asgeojson(" in h_lower:
                geom_col_index = i
                break

        display_rows = []
        geojson_features = []

        for row in db_rows:
            display_row = list(row)

            if geom_col_index is not None:
                geojson_str = row[geom_col_index]
                if geojson_str:
                    try:
                        geojson_obj = json.loads(geojson_str)
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"GeoJSON decode error: {e} | data: {str(geojson_str)[:100]}")
                        continue

                    properties = {
                        headers[i]: value
                        for i, value in enumerate(row)
                        if i != geom_col_index
                    }
                    geojson_features.append({
                        "type": "Feature",
                        "geometry": geojson_obj,
                        "properties": properties,
                    })

                display_row.pop(geom_col_index)

            display_rows.append(display_row)

        display_headers = (
            [h for i, h in enumerate(headers) if i != geom_col_index]
            if geom_col_index is not None
            else headers
        )

        final_geojson = create_folium_map(geojson_features)
        print(f"[mode={mode}] Returning {len(geojson_features)} features.")

        return JSONResponse(content={
            "sql":               sql,
            "rows_count":        len(db_rows),
            "headers":           display_headers,
            "display_rows":      display_rows,
            "geo_json_features": final_geojson,
            "map_html":          final_geojson,
        })

    except RuntimeError as e:
        print(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    except ValueError as e:
        print(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid query: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
