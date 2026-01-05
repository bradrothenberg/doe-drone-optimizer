"""
nTop launch endpoint for opening designs in nTopology software
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import subprocess
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter()

# Configuration - paths to nTop executable and file
NTOP_EXE = Path(r"C:\Program Files\nTopology\nTopology\ntop.exe")
NTOP_FILE = Path(r"c:\Users\bradrothenberg\OneDrive - nTop\OUT\parts\Group3_Test\RHEL\dependencies\nTopGrp3_v3.ntop")


class LaunchNtopRequest(BaseModel):
    """Request body for launching nTop with design parameters."""
    run_id: Optional[str] = None
    loa: float
    span: float
    le_sweep_p1: float
    le_sweep_p2: float
    te_sweep_p1: float
    te_sweep_p2: float
    panel_break: float


class LaunchNtopResponse(BaseModel):
    """Response from nTop launch request."""
    status: str
    message: str


@router.post("/launch-ntop", response_model=LaunchNtopResponse, tags=["ntop"])
async def launch_ntop(request: LaunchNtopRequest):
    """
    Launch nTop with the provided design parameters.

    Creates an input.json file with the design parameters and launches
    nTop with the pre-configured ntop file.
    """
    try:
        run_label = request.run_id or "Design"
        logger.info(f"Launching nTop for {run_label}")

        # Values should already be in inches (from the optimizer)
        loa_value = request.loa
        span_value = request.span

        # If values look like feet (< 50), convert to inches
        if loa_value < 50:
            loa_value = loa_value * 12
            logger.info(f"Converting LOA: {request.loa:.2f} ft -> {loa_value:.2f} in")
        if span_value < 50:
            span_value = span_value * 12
            logger.info(f"Converting Span: {request.span:.2f} ft -> {span_value:.2f} in")

        # Build input.json structure for nTop
        input_json = {
            "inputs": [
                {
                    "name": "LOA In",
                    "type": "real",
                    "value": loa_value,
                    "units": "in"
                },
                {
                    "name": "Span",
                    "type": "real",
                    "value": span_value,
                    "units": "in"
                },
                {
                    "name": "LE Sweep P1",
                    "type": "real",
                    "value": request.le_sweep_p1,
                    "units": "deg"
                },
                {
                    "name": "LE Sweep P2",
                    "type": "real",
                    "value": request.le_sweep_p2,
                    "units": "deg"
                },
                {
                    "name": "TE Sweep P1",
                    "type": "real",
                    "value": request.te_sweep_p1,
                    "units": "deg"
                },
                {
                    "name": "TE Sweep P2",
                    "type": "real",
                    "value": request.te_sweep_p2,
                    "units": "deg"
                },
                {
                    "name": "Panel Break Span %",
                    "type": "real",
                    "value": request.panel_break
                },
                {
                    "name": "MAIN PATH",
                    "type": "file_path",
                    "value": str(Path(tempfile.gettempdir()) / "ntop_viewer_output/")
                }
            ]
        }

        # Create output directory
        output_dir = Path(tempfile.gettempdir()) / "ntop_viewer_output"
        output_dir.mkdir(exist_ok=True)

        # Write input.json to temp file
        input_path = Path(tempfile.gettempdir()) / "ntop_viewer_input.json"
        input_path.write_text(json.dumps(input_json, indent=2))

        logger.info(f"Input JSON written to: {input_path}")
        logger.info("Design parameters:")
        for inp in input_json["inputs"]:
            if inp["name"] != "MAIN PATH":
                units = inp.get("units", "")
                logger.info(f"  {inp['name']}: {inp['value']} {units}")

        # Check if nTop executable exists
        if not NTOP_EXE.exists():
            error_msg = f"nTop executable not found at: {NTOP_EXE}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        # Check if ntop file exists
        if not NTOP_FILE.exists():
            error_msg = f"nTop file not found at: {NTOP_FILE}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        # Launch nTop
        cmd = [str(NTOP_EXE), "--trustnotebook", "-j", str(input_path), str(NTOP_FILE)]
        logger.info(f"Launching: {' '.join(cmd)}")

        subprocess.Popen(cmd, shell=False)

        logger.info("nTop launched successfully!")

        return LaunchNtopResponse(
            status="ok",
            message=f"Launched nTop for {run_label}"
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error launching nTop: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
