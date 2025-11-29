from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
import json
import os
import traceback

from quiz_solver import AutoSolver

from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="QuizSolver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Payload(BaseModel):
    email: str
    secret: str
    url: str

SECRET = os.getenv("SECRET")

def run_solver_sync(payload: dict) -> None:
    """Run AutoSolver synchronously and persist the run log to a JSON file."""
    try:
        solver = AutoSolver(payload)
        solver.run()

        # Persist the scraped_data_log for later inspection
        outname = f"last_run_{int(time.time())}.json"
        with open(outname, "w", encoding="utf-8") as f:
            json.dump(solver.scraped_data_log, f, indent=2)
    except Exception:
        # Write a traceback file so failures are visible when running in background
        tb = traceback.format_exc()
        with open("last_run_error.log", "w", encoding="utf-8") as f:
            f.write(tb)


@app.post("/solve")
def solve(payload: Payload, background_tasks: BackgroundTasks, wait: Optional[bool] = False):
    """Accepts the initial payload and dispatches an AutoSolver run.

    - If `wait=true` the endpoint will run the solver synchronously and return
      the persisted run filename or an error message.
    - Otherwise the solver runs in the background and the endpoint returns
      an acknowledgement immediately.
    """
    payload_dict = payload.model_dump()
    
    print("Received payload:",payload_dict['secret'], payload_dict['url'])
    
    if SECRET and payload_dict["secret"] != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret key")

    # Basic validation
    if not payload_dict["email"] or not payload_dict["secret"] or not payload_dict["url"]:
        raise HTTPException(status_code=400, detail="Missing required payload fields")

    if wait:
        # Run synchronously and return the output filename (or error log)
        try:
            run_solver_sync(payload_dict)
            # find most recent last_run_*.json file
            candidates = [f for f in os.listdir(".") if f.startswith("last_run_") and f.endswith(".json")]
            latest = max(candidates, key=lambda p: os.path.getmtime(p)) if candidates else None
            return {"status": "completed", "output": latest}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    # Launch as a background task and return accepted
    background_tasks.add_task(run_solver_sync, payload_dict)
    return {"status": "accepted", "message": "Solver started in background"}


@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)